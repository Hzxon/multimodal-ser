"""Write small manifests for smoke tests and CI-style runs, never for reported numbers.

Usage: python scripts/make_subset_manifests.py --n 64 --out data/processed/smoke_manifests [--seed 0]

Takes the first --n rows per split (stratified: at least 2 per class where available)
from data/manifests, fills wav_path/duration_s/has_audio from data/processed/wav16k when
the source manifest lacks them, and writes <out>/{train,dev,test}.csv. Point any stage at
it with --manifest-dir <out>. If <out>/asr/<tag>/ exists it is left alone.
"""
import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="data/processed/smoke_manifests")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    wav_dir = ROOT / cfg["wav_dir"]
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(ROOT / cfg["manifest_dir"] / f"{split}.csv", dtype={"exclude_reason": "string"})
        df["exclude_reason"] = df["exclude_reason"].fillna("") if "exclude_reason" in df else ""
        df = df[df["exclude_reason"] == ""]
        if "has_audio" not in df:
            paths = [wav_dir / split / f"dia{d}_utt{u}.wav" for d, u in zip(df["dialogue_id"], df["utterance_id"])]
            has = [p.exists() for p in paths]
            df = df.assign(
                wav_path=[str(p.relative_to(ROOT)) if h else "" for p, h in zip(paths, has)],
                has_audio=has,
            )
            df = df[df["has_audio"]]
            df["duration_s"] = [sf.info(str(ROOT / p)).duration for p in df["wav_path"]]
            df["sha256"] = ""
            df["clean_subset"] = (df["duration_s"] >= cfg["min_clip_seconds"]) & (df["duration_s"] <= cfg["max_clean_seconds"])
        else:
            df = df[df["has_audio"].astype(bool)]
        per_class = max(2, a.n // (len(cfg["emotions"]) * 2))
        head = pd.concat([g.sample(min(len(g), per_class), random_state=a.seed) for _, g in df.groupby("emotion")])
        rest = df.drop(head.index)
        need = max(0, a.n - len(head))
        sub = pd.concat([head, rest.sample(min(need, len(rest)), random_state=a.seed)]).sort_values("utt_id")
        sub.to_csv(out / f"{split}.csv", index=False)
        print(f"{split}: {len(sub)} rows -> {out / f'{split}.csv'}")


if __name__ == "__main__":
    main()
