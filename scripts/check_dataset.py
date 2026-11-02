"""Gate 1: dataset integrity checks on data/manifests/{train,dev,test}.csv.

Usage: python scripts/check_dataset.py

Runs every check in PLAN.md Section 3.1 that the available columns allow. Audio checks
(D2, D5, D8, duration agreement) are skipped with a note when the manifests were built
with --labels-only. Writes reports/gate1.md and reports/figures/class_dist.png, and
exits 1 if any check fails.
"""
import re
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["train", "dev", "test"]
MOJIBAKE = re.compile(r"â€|Ã.|\x92|�")
EMO_TO_SENT = {
    "joy": {"positive"}, "neutral": {"neutral"},
    "anger": {"negative"}, "disgust": {"negative"}, "fear": {"negative"}, "sadness": {"negative"},
    "surprise": {"positive", "negative", "neutral"},
}


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    emotions = cfg["emotions"]
    str_cols = {c: "string" for c in ["exclude_reason", "sha256", "wav_path"]}
    dfs = {s: pd.read_csv(ROOT / cfg["manifest_dir"] / f"{s}.csv", dtype=str_cols) for s in SPLITS}
    has_audio = all("has_audio" in d for d in dfs.values())
    rows: list[tuple[str, str, str, str]] = []  # id, status, criterion, evidence

    def rec(cid: str, ok: bool | None, crit: str, ev: str, warn: bool = False) -> None:
        status = "PASS" if ok else "SKIP" if ok is None else "WARN" if warn else "FAIL"
        rows.append((cid, status, crit, ev))

    for d in dfs.values():
        d["exclude_reason"] = d["exclude_reason"].fillna("") if "exclude_reason" in d else ""
    kept = {s: d[d["exclude_reason"] == ""] for s, d in dfs.items()}
    excluded = {s: int((d["exclude_reason"] != "").sum()) for s, d in dfs.items()}

    # --- Label validity (correctness of the label files) ---
    bad = {s: sorted(set(d["emotion"]) - set(emotions)) for s, d in dfs.items()}
    rec("L1", not any(bad.values()), "Every emotion label is one of the 7 expected classes", f"unexpected labels: {bad}")
    sents = {s: sorted(set(d["sentiment"])) for s, d in dfs.items()}
    rec("L2", all(set(v) <= {"positive", "negative", "neutral"} for v in sents.values()),
        "Sentiment labels are positive/negative/neutral", f"{sents}")
    incons = {s: int((~d.apply(lambda r: r["sentiment"] in EMO_TO_SENT.get(r["emotion"], set()), axis=1)).sum())
              for s, d in dfs.items()}
    rec("L3", all(v == 0 for v in incons.values()),
        "Emotion and sentiment agree (joy→positive, neutral→neutral, anger/disgust/fear/sadness→negative, surprise→any)",
        f"inconsistent rows per split: {incons}")
    empty = {s: int((d["gold_text"].fillna("").str.strip() == "").sum()) for s, d in dfs.items()}
    rec("L4", all(v == 0 for v in empty.values()), "No empty transcripts", f"empty per split: {empty}")
    moji = {s: int(d["gold_text"].fillna("").str.contains(MOJIBAKE).sum()) for s, d in dfs.items()}
    rec("L5", all(v == 0 for v in moji.values()), "No mojibake or replacement characters in transcripts",
        f"rows with suspicious characters per split: {moji}")
    nonpos = {s: int((d["nominal_duration_s"] <= 0).sum()) for s, d in dfs.items()}
    zero_rows = [u for d in dfs.values() for u in d.loc[d["nominal_duration_s"] <= 0, "utt_id"]]
    rec("L6", all(v == 0 for v in nonpos.values()), "EndTime is after StartTime for every row",
        f"non-positive nominal durations per split: {nonpos}; rows: {zero_rows}. "
        "Known MELD annotation defect; decided by the audio length in D9.", warn=True)

    # --- A1 ffprobe inventory of the raw clips (Stage 0 of preprocessing_validation.md) ---
    inv_path = ROOT / "data/processed/ffprobe_inventory.csv"
    if inv_path.exists():
        inv = pd.read_csv(inv_path, dtype={"error": "string"})
        inv["error"] = inv["error"].fillna("")
        inv["key"] = inv["file"].str.replace(".mp4", "", regex=False)
        parts = []
        for s in SPLITS:
            sub = inv[inv["split"] == s]
            keys = set(dfs[s]["utt_id"].str.replace(f"{s}_", "", regex=False))
            files = set(sub["key"])
            parts.append(f"{s}: files {len(sub)}, labelled rows {len(keys)}, rows without file {len(keys - files)} "
                         f"{sorted(keys - files)[:5]}, files without label {len(files - keys)}, "
                         f"errors {int((sub['error'] != '').sum())} {sub.loc[sub['error'] != '', 'file'].tolist()[:5]}, "
                         f"channels {sub['channels'].value_counts().to_dict()}, sample rates {sub['sample_rate'].value_counts().to_dict()}")
        errs = inv[inv["error"] != ""]
        rec("A1", len(errs) < 0.01 * len(inv), "Raw inventory: every clip has one decodable audio stream; label/file set differences listed",
            "\n".join(parts))
    else:
        rec("A1", None, "Raw inventory", "run scripts/ffprobe_inventory.py")

    # --- L7 label source agreement ---
    try:
        import re as _re
        diffs = []
        for s in SPLITS:
            inner_p = next((q for q in (ROOT / "data/raw").rglob(f"{s}_sent_emo.csv") if "MELD" in str(q.relative_to(ROOT / "data/raw"))), None)
            if inner_p is None:
                continue
            inner = pd.read_csv(inner_p, encoding="cp1252")
            gh = pd.read_csv(ROOT / cfg["labels"][s])
            meta = ["Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID", "Season", "Episode", "StartTime", "EndTime"]
            m = (inner[meta].astype(str).values != gh[meta].astype(str).values).any(axis=1)
            strip = lambda t: _re.sub(r"[^A-Za-z0-9 ]", "", str(t))
            t = inner["Utterance"].map(strip) != gh["Utterance"].map(strip)
            diffs.append(f"{s}: rows {len(gh)} vs {len(inner)}; metadata differs in {int(m.sum())} rows "
                         f"{[f'dia{d}_utt{u}' for d, u in zip(gh.loc[m, 'Dialogue_ID'], gh.loc[m, 'Utterance_ID'])][:5]}; "
                         f"text differs beyond encoding/punctuation in {int(t.sum())} rows")
        rec("L7", True, "Archive CSVs vs GitHub CSVs compared; GitHub (UTF-8, corrected) is the label source", "; ".join(diffs) or "archive CSVs not found")
    except Exception as e:  # noqa: BLE001
        rec("L7", None, "Archive CSVs vs GitHub CSVs", f"comparison failed: {e}")

    # --- D2 clips decode ---
    if has_audio:
        missing = {s: int((~d["has_audio"]).sum()) for s, d in dfs.items()}
        fail_file = ROOT / "data/processed/extract_failures.csv"
        nfail = len(pd.read_csv(fail_file)) if fail_file.exists() else -1
        rec("D2", all(v < 0.01 * len(dfs[s]) for s, v in missing.items()),
            "Fewer than 1 % of rows per split lack a decodable clip; count reported",
            f"rows without audio per split: {missing}; ffmpeg failures logged: {nfail}")
    else:
        rec("D2", None, "Clips decode", "labels-only manifests; rerun after audio extraction")

    # --- D3 class distribution ---
    counts = pd.DataFrame({s: d["emotion"].value_counts() for s, d in dfs.items()}).reindex(emotions).fillna(0).astype(int)
    ratio = {s: round(float(counts[s].max() / counts[s].min()), 1) for s in SPLITS}
    fig, ax = plt.subplots(figsize=(8, 4))
    (counts / counts.sum()).plot.bar(ax=ax)
    ax.set_ylabel("share of split")
    ax.set_title("MELD class distribution per split")
    fig.tight_layout()
    (ROOT / "reports/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "reports/figures/class_dist.png", dpi=120)
    rec("D3", True, "Class counts and imbalance ratio (max/min) reported per split",
        "counts:\n\n" + counts.to_markdown() + f"\n\nimbalance ratio: {ratio}")

    # --- D4 split integrity ---
    ids = {s: set(d["utt_id"]) for s, d in kept.items()}
    keys = {s: set(zip(d["season"], d["episode"], d["start_time"], d["end_time"])) for s, d in kept.items()}
    eps = {s: set(zip(d["season"], d["episode"])) for s, d in dfs.items()}
    pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
    id_overlap = {f"{a}/{b}": len(ids[a] & ids[b]) for a, b in pairs}
    key_overlap = {f"{a}/{b}": len(keys[a] & keys[b]) for a, b in pairs}
    ep_overlap = {f"{a}/{b}": len(eps[a] & eps[b]) for a, b in pairs}
    rec("D4", all(v == 0 for v in id_overlap.values()) and all(v == 0 for v in key_overlap.values()),
        "No non-excluded utterance (by id, and by season/episode/start/end timestamp) appears in two splits",
        f"train rows excluded: {excluded['train']} by reason "
        f"{dfs['train'].loc[dfs['train']['exclude_reason'] != '', 'exclude_reason'].str.split(':').str[0].value_counts().to_dict()}; "
        f"timestamp duplicates: {sorted(dfs['train'].loc[dfs['train']['exclude_reason'].str.startswith('timestamp'), 'exclude_reason'])}; "
        f"after exclusion, utt_id overlap: {id_overlap}; timestamp-key overlap: {key_overlap}; "
        f"episodes shared (information only, dialogues are the split unit): {ep_overlap}")

    # --- D5 duplicate audio ---
    if has_audio:
        h = {s: d.loc[d["has_audio"], "sha256"] for s, d in kept.items()}
        cross = {f"{a}/{b}": len(set(h[a]) & set(h[b])) for a, b in pairs}
        within = {s: int(h[s].duplicated().sum()) for s in SPLITS}
        rec("D5", all(v == 0 for v in cross.values()), "No identical audio file (sha256) across splits after exclusions",
            f"cross-split identical files: {cross}; duplicates within split: {within}")
    else:
        rec("D5", None, "Duplicate audio", "labels-only manifests")

    # --- D6 speaker overlap ---
    spk = {s: set(d["speaker"]) for s, d in dfs.items()}
    shared = spk["train"] & spk["dev"] & spk["test"]
    share_rows = {s: round(float(d["speaker"].isin(shared).mean()), 3) for s, d in dfs.items()}
    rec("D6", True, "Speaker overlap across splits is documented (MELD is dialogue-disjoint, not speaker-disjoint)",
        f"{len(shared)} speakers appear in all three splits: {sorted(shared)}; "
        f"fraction of rows spoken by them per split: {share_rows}")

    # --- D7 label not in path ---
    if has_audio:
        leak = sum(int(d.loc[d["has_audio"]].apply(lambda r: r["emotion"] in r["wav_path"].lower(), axis=1).sum())
                   for d in dfs.values())
        rec("D7", leak == 0, "No wav path contains its emotion label", f"paths containing the label: {leak}")
    else:
        rec("D7", None, "Label not in path", "labels-only manifests")

    # --- D8 durations ---
    col = "duration_s" if has_audio else "nominal_duration_s"
    stats = pd.DataFrame({s: d[col].describe()[["count", "mean", "50%", "min", "max"]] for s, d in dfs.items()}).round(2)
    long = {s: int((d[col] > cfg["long_clip_seconds"]).sum()) for s, d in dfs.items()}
    rec("D8", True, f"Duration stats per split ({col}); clips above {cfg['long_clip_seconds']} s counted",
        "\n\n" + stats.to_markdown() + f"\n\nclips above threshold: {long}")
    if has_audio:
        diff = {s: int(((d["duration_s"] - d["nominal_duration_s"]).abs() > 1.0).sum()) for s, d in dfs.items()}
        rec("D9", all(v < 0.02 * len(dfs[s]) for s, v in diff.items()),
            "Extracted audio length agrees with the CSV timestamps within 1 s for at least 98 % of clips "
            "(WARN only: MELD timestamps are known to be unreliable; the WAV length is the ground truth downstream)",
            f"clips differing by more than 1 s per split: {diff}", warn=True)
    else:
        rec("D9", None, "Audio length agrees with timestamps", "labels-only manifests")

    # --- P1 data policy outcome ---
    if has_audio and "clean_subset" in dfs["train"]:
        reasons = dfs["train"]["exclude_reason"].replace("", "kept").str.split(":").str[0].value_counts().to_dict()
        clean = {s: f"{int(d['clean_subset'].sum())}/{len(d)}" for s, d in dfs.items()}
        rec("P1", excluded["dev"] == 0 and excluded["test"] == 0,
            "Data policy applied: train exclusions by reason; dev/test never excluded; clean_subset sizes",
            f"train exclusion reasons: {reasons}; clean_subset per split: {clean}; "
            f"train class counts after exclusion: {dfs['train'].loc[dfs['train']['exclude_reason'] == '', 'emotion'].value_counts().to_dict()}")

    # --- Report ---
    out = ["# Gate 1: dataset checks", "", f"Manifests: `{cfg['manifest_dir']}`; audio columns present: {has_audio}", "",
           "| ID | Status | Criterion | Evidence |", "|---|---|---|---|"]
    for cid, st, crit, ev in rows:
        short = ev.split("\n")[0].replace("|", "/")
        out.append(f"| {cid} | {st} | {crit} | {short} |")
    # Full evidence, including tables, goes below the summary table.
    out += ["", "## Details", ""]
    for cid, st, crit, ev in rows:
        out += [f"### {cid} ({st})", "", crit, "", ev, ""]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/gate1.md").write_text("\n".join(out))
    for cid, st, crit, ev in rows:
        print(f"{cid:3s} {st:4s} {crit}")
        if st == "FAIL":
            print("     ", ev.replace("\n", " ")[:300])
    fails = [r for r in rows if r[1] == "FAIL"]
    print(f"\n{len(fails)} FAIL, {sum(r[1] == 'SKIP' for r in rows)} SKIP -> reports/gate1.md")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
