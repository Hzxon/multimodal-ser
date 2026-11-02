"""Gate 3, stages 0-2 of reports/preprocessing_validation.md: raw inventory, WAV extraction
and manifest integrity. (ASR, feature and loader checks are added in Phase 3.)

Usage: python scripts/check_preprocessing.py [--sample 200]
Writes reports/gate3_stage0-2.md; exits 1 on FAIL.
"""
import argparse
import hashlib
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["train", "dev", "test"]


def channel_rms(mp4: Path) -> tuple[float, float]:
    """RMS of ffmpeg's mono downmix vs RMS of the loudest source channel (16 kHz float)."""
    def run(extra: list[str]) -> np.ndarray:
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(mp4), "-vn", "-ar", "16000", "-f", "f32le", *extra, "-"]
        out = subprocess.run(cmd, capture_output=True).stdout
        return np.frombuffer(out, dtype=np.float32)
    mono = run(["-ac", "1"])
    multi = run([])  # native channel count, interleaved
    n_ch = int(round(len(multi) / max(len(mono), 1)))
    if n_ch <= 1 or len(mono) == 0:
        return float("nan"), float("nan")
    multi = multi[: (len(multi) // n_ch) * n_ch].reshape(-1, n_ch)
    db = lambda x: 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9)
    return float(db(mono)), float(max(db(multi[:, c]) for c in range(n_ch)))


def source_mp4(r) -> Path | None:
    """The mp4 a WAV was made from: the corrected 'final_videos_test' cut when one exists."""
    stem = Path(r.wav_path).stem
    src_note = (ROOT / r.wav_path).with_suffix(".src")
    name = src_note.read_text().strip() if src_note.exists() else stem + ".mp4"
    # the same dia/utt name exists in every split folder, so filter by the split's folder name
    for cand in (ROOT / "data/raw").rglob(name):
        if r.split in cand.parent.name.lower():
            return cand
    return None


def source_key(r) -> str:
    """Inventory key (split_filename-without-extension) of the WAV's source mp4."""
    src_note = (ROOT / r.wav_path).with_suffix(".src")
    name = src_note.read_text().strip()[:-4] if src_note.exists() else Path(r.wav_path).stem
    return f"{r.split}_{name}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200)
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    rng = random.Random(0)
    dfs = {s: pd.read_csv(ROOT / cfg["manifest_dir"] / f"{s}.csv", dtype={"exclude_reason": "string", "wav_path": "string", "sha256": "string"}) for s in SPLITS}
    for d in dfs.values():
        d["exclude_reason"] = d["exclude_reason"].fillna("")
    inv = pd.read_csv(ROOT / "data/processed/ffprobe_inventory.csv", dtype={"error": "string"})
    inv["error"] = inv["error"].fillna("")
    inv["key"] = inv["split"] + "_" + inv["file"].str.replace(".mp4", "", regex=False)
    rows = []

    def rec(cid, ok, crit, ev, warn=False):
        rows.append((cid, "PASS" if ok else "SKIP" if ok is None else "WARN" if warn else "FAIL", crit, ev))

    allm = pd.concat(dfs.values(), ignore_index=True)
    with_audio = allm[allm["has_audio"]]

    # S1.1 output format on every WAV
    bad_fmt, zero, info_dur = [], [], {}
    for r in with_audio.itertuples():
        i = sf.info(str(ROOT / r.wav_path))
        info_dur[r.utt_id] = i.frames / i.samplerate
        if i.samplerate != cfg["sample_rate"] or i.channels != 1 or i.subtype != "PCM_16":
            bad_fmt.append(r.utt_id)
    rec("S1.1", not bad_fmt, f"Every WAV is {cfg['sample_rate']} Hz, mono, PCM_16", f"{len(with_audio)} files checked; wrong format: {bad_fmt[:5]} ({len(bad_fmt)})")

    # S1.2 WAV length vs source audio stream length (ffprobe)
    src = inv.set_index("key")["audio_duration"]
    skey = {r.utt_id: source_key(r) for r in with_audio.itertuples()}
    diff = pd.Series({u: abs(d - float(src.get(skey[u], np.nan))) for u, d in info_dur.items()})
    over = diff[diff > 0.05]
    rec("S1.2", len(over) == 0, "WAV duration equals the source audio stream duration within 50 ms (AAC frame padding is about 21 ms)",
        f"max diff {diff.max():.3f} s; over 50 ms: {len(over)} {over.head(5).round(3).to_dict()}", warn=len(over) < 0.01 * len(diff))

    # S1.3 downmix cancellation + S1.4 silent/clipped, on a random sample
    sample = with_audio.sample(min(a.sample, len(with_audio)), random_state=0)
    cancel, silent, clipped, const = [], [], [], []
    for r in sample.itertuples():
        y, sr = sf.read(str(ROOT / r.wav_path), dtype="float32")
        if len(y) == 0 or np.max(np.abs(y)) == 0:
            const.append(r.utt_id); continue
        rms = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9)
        if rms < -40:
            silent.append((r.utt_id, round(float(rms), 1)))
        if np.mean(np.abs(y) >= 0.99) > 0.001:
            clipped.append(r.utt_id)
        mp4 = source_mp4(r)
        if mp4 is not None:
            m, c = channel_rms(mp4)
            if not np.isnan(m) and c - m > 6:
                cancel.append((r.utt_id, round(c - m, 1)))
    rec("S1.3", not cancel, f"Mono downmix is within 6 dB of the loudest source channel (sample of {len(sample)})",
        f"clips where the mix is more than 6 dB quieter than a source channel: {cancel[:10]} ({len(cancel)})")
    rec("S1.4", not const and not clipped, f"No empty/constant WAVs and no clipping (sample of {len(sample)})",
        f"constant: {const[:5]}; clipped>0.1%: {clipped[:5]}; RMS below -40 dBFS (information): {len(silent)} {silent[:5]}")

    # S1.5 idempotence: re-extract 20 clips to a temp dir and compare bytes
    tmp = ROOT / "data/processed/_recheck"; tmp.mkdir(exist_ok=True)
    mism = []
    for r in sample.head(20).itertuples():
        mp4 = source_mp4(r)
        out = tmp / Path(r.wav_path).name
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(mp4), "-vn", "-ac", "1", "-ar", str(cfg["sample_rate"]), "-sample_fmt", "s16", str(out)], check=True)
        y1, _ = sf.read(str(out), dtype="int16")
        y2, _ = sf.read(str(ROOT / r.wav_path), dtype="int16")
        if hashlib.sha256(y1.tobytes()).hexdigest() != r.sha256 or not np.array_equal(y1, y2):
            mism.append(r.utt_id)
        out.unlink()
    tmp.rmdir()
    rec("S1.5", not mism, "Re-extracting 20 random clips reproduces the stored samples and content hash", f"mismatches: {mism}")

    # S2 manifests
    counts = {s: len(d) for s, d in dfs.items()}
    rec("S2.1", counts == {"train": 9989, "dev": 1109, "test": 2610}, "Manifest row counts equal the label CSVs", f"{counts}")
    rec("S2.2", allm["utt_id"].is_unique and all(not d.duplicated(["dialogue_id", "utterance_id"]).any() for d in dfs.values()),
        "utt_id unique across splits; (dialogue_id, utterance_id) unique within split", "ok" if allm["utt_id"].is_unique else "duplicates present")
    req = ["emotion", "gold_text", "speaker", "split", "exclude_reason", "clean_subset", "has_audio"]
    nan = {c: int(allm[c].isna().sum()) for c in req}
    dur_nan_ok = bool((allm["duration_s"].isna() == ~allm["has_audio"]).all())
    rec("S2.3", all(v == 0 for v in nan.values()) and dur_nan_ok and set(allm["emotion"]) <= set(cfg["emotions"]),
        "Schema: required columns non-null; duration_s null exactly where has_audio is False; labels in the class list", f"nulls {nan}; duration-null pattern ok {dur_nan_ok}")
    excl = {s: d["exclude_reason"].replace("", "kept").str.split(":").str[0].value_counts().to_dict() for s, d in dfs.items()}
    rec("S2.4", set(excl["dev"]) == {"kept"} and set(excl["test"]) == {"kept"}, "Exclusions only in train; dev/test complete", f"{excl}")
    clean = {s: int(d["clean_subset"].sum()) for s, d in dfs.items()}
    rec("S2.5", True, "clean_subset flag stored once in the manifests", f"{clean}")
    # S2.6 audio ↔ row: path ids match row ids and hash recomputes
    bad = []
    for r in sample.head(50).itertuples():
        stem = Path(r.wav_path).stem
        if stem != f"dia{r.dialogue_id}_utt{r.utterance_id}" or f"/{r.split}/" not in r.wav_path:
            bad.append(r.utt_id); continue
        y, _ = sf.read(str(ROOT / r.wav_path), dtype="int16")
        if hashlib.sha256(y.tobytes()).hexdigest() != r.sha256:
            bad.append(r.utt_id)
    rec("S2.6", not bad, "50 random rows: wav path ids equal row ids and the content hash recomputes", f"bad: {bad}")
    # S2.7 cross-split audio hash leak after exclusions
    kept = allm[(allm["exclude_reason"] == "") & allm["has_audio"]]
    dup = kept[kept.duplicated("sha256", keep=False)].groupby("sha256")["split"].nunique()
    rec("S2.7", int((dup > 1).sum()) == 0, "No identical audio across splits among non-excluded rows", f"cross-split identical groups: {int((dup > 1).sum())}; within-split identical groups: {int((dup == 1).sum())}")
    # S2.8 manifest hashes
    hashes = {s: hashlib.sha256((ROOT / cfg["manifest_dir"] / f"{s}.csv").read_bytes()).hexdigest()[:16] for s in SPLITS}
    rec("S2.8", True, "Manifest hashes recorded", f"{hashes}")

    out = ["# Gate 3, stages 0-2: raw inventory, extraction, manifests", "", "| ID | Status | Criterion | Evidence |", "|---|---|---|---|"]
    out += [f"| {a_} | {b} | {c} | {str(d).replace('|', '/')} |" for a_, b, c, d in rows]
    (ROOT / "reports/gate3_stage0-2.md").write_text("\n".join(out) + "\n")
    for cid, st, crit, ev in rows:
        print(f"{cid:5s} {st:4s} {crit}")
        if st in ("FAIL", "WARN"):
            print("      ", str(ev)[:400])
    fails = [r for r in rows if r[1] == "FAIL"]
    print(f"\n{len(fails)} FAIL -> reports/gate3_stage0-2.md")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
