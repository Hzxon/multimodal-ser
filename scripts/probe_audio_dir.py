"""Audit a directory of MELD audio clips against a split's manifest (used for the dev-split
mirror probe before the official archive is available; reusable on data/processed/wav16k).

Usage: python scripts/probe_audio_dir.py <audio_dir> <split> [--out csv]
"""
import argparse
import hashlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLIP_RE = re.compile(r"dia(\d+)_utt(\d+)")


def stats(path: str) -> dict:
    import librosa
    import soundfile as sf

    p = Path(path)
    try:
        y, sr = sf.read(str(p), dtype="float32", always_2d=False)
    except Exception:
        y, sr = librosa.load(str(p), sr=None, mono=False)
        y = y.T if y.ndim > 1 else y
    channels = y.shape[1] if y.ndim > 1 else 1
    if y.ndim > 1:
        y = y.mean(axis=1)
    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    rms_db = 20 * np.log10(rms + 1e-8)
    f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr, frame_length=2048, hop_length=hop) if len(y) > 2048 else np.array([])
    voiced = rms_db[: len(f0)] > (rms_db.max() - 25) if len(f0) else np.array([], dtype=bool)
    flat = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0] if len(y) > 2048 else np.array([np.nan])
    m = CLIP_RE.search(p.stem)
    return {
        "file": p.name,
        "dialogue_id": int(m.group(1)) if m else -1,
        "utterance_id": int(m.group(2)) if m else -1,
        "sr": sr,
        "channels": channels,
        "duration_s": len(y) / sr,
        "rms_db": float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-8)) if len(y) else float("nan"),
        "f0_median_hz": float(np.median(f0[voiced])) if voiced.any() else float("nan"),
        "dyn_range_db": float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)),
        "silence_frac": float((rms_db < rms_db.max() - 40).mean()),
        "clipped_frac": float((np.abs(y) >= 0.99).mean()),
        "flatness_median": float(np.nanmedian(flat)),
        "sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir")
    ap.add_argument("split")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    files = sorted(str(p) for p in Path(a.audio_dir).rglob("*") if p.is_file() and CLIP_RE.search(p.stem) and not p.name.startswith("._"))
    print(f"{len(files)} audio files")
    with ProcessPoolExecutor(a.jobs) as ex:
        st = pd.DataFrame(list(ex.map(stats, files, chunksize=16)))
    man = pd.read_csv(ROOT / f"data/manifests/{a.split}.csv")
    df = man.merge(st, on=["dialogue_id", "utterance_id"], how="outer", indicator=True)
    extra = df[df["_merge"] == "right_only"]
    missing = df[df["_merge"] == "left_only"]
    both = df[df["_merge"] == "both"].copy()
    print(f"manifest rows {len(man)}; files matched {len(both)}; manifest rows without file {len(missing)}: {sorted(missing['utt_id'])[:10]}")
    print(f"files without manifest row {len(extra)}: {sorted(extra['file'])[:10]}")
    print("sample rates:", st["sr"].value_counts().to_dict(), "channels:", st["channels"].value_counts().to_dict())
    d = both["duration_s"]
    print(f"audio duration: median {d.median():.2f} s, mean {d.mean():.2f}, min {d.min():.2f}, max {d.max():.2f}, <0.5 s: {(d < 0.5).sum()}, >15 s: {(d > 15).sum()}")
    diff = (both["duration_s"] - both["nominal_duration_s"]).abs()
    print(f"|audio - timestamp span| > 1 s: {(diff > 1).sum()} rows ({(diff > 1).mean():.1%}); > 3 s: {(diff > 3).sum()}; "
          f"rows with span < 0.3 s whose audio is > 1 s: {((both['nominal_duration_s'] < 0.3) & (both['duration_s'] > 1)).sum()} of {(both['nominal_duration_s'] < 0.3).sum()}")
    print(f"near-silent (rms < -40 dB): {(both['rms_db'] < -40).sum()}; clipped > 0.1 %: {(both['clipped_frac'] > 0.001).sum()}; "
          f"low dynamic range (< 10 dB): {(both['dyn_range_db'] < 10).sum()}; silence_frac > 0.5: {(both['silence_frac'] > 0.5).sum()}")
    print(f"identical audio content within split (sha256 of samples): {both['sha256'].duplicated(keep=False).sum()} rows")
    print("\nper-class means:")
    print(both.groupby("emotion")[["duration_s", "rms_db", "f0_median_hz", "dyn_range_db", "flatness_median"]].mean().round(2).to_string())
    dup = both[both["sha256"].duplicated(keep=False)].sort_values("sha256")[["utt_id", "speaker", "emotion", "gold_text", "sha256"]]
    if len(dup):
        print("\nidentical-audio groups:")
        print(dup.to_string(max_colwidth=40))
    if a.out:
        both.drop(columns="_merge").to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
