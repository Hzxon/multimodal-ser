"""Build data/manifests/{train,dev,test}.csv from the MELD label CSVs and the extracted WAVs.

Usage: python -m src.data.build_manifest [--labels-only]

Columns: utt_id, split, dialogue_id, utterance_id, speaker, emotion, sentiment,
gold_text, season, episode, start_time, end_time, nominal_duration_s, exclude_reason, clean_subset, and, unless
--labels-only, wav_path, duration_s, sha256, has_audio.

The manifests are the single source of truth for every later stage. Rows whose clip
could not be extracted keep has_audio = False so the loss is visible, and are dropped
by the dataset loader, not here.
"""
import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]


def parse_ts(s: str) -> float:
    t = datetime.strptime(s.strip(), "%H:%M:%S,%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def sha256_of(path: Path) -> str:
    """Hash of the PCM samples, not the file bytes, so WAV header differences between
    ffmpeg runs do not change identity and identical audio is detected reliably."""
    y, _ = sf.read(str(path), dtype="int16")
    return hashlib.sha256(y.tobytes()).hexdigest()


def load_labels(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    out = pd.DataFrame({
        "utt_id": [f"{split}_dia{d}_utt{u}" for d, u in zip(df["Dialogue_ID"], df["Utterance_ID"])],
        "split": split,
        "dialogue_id": df["Dialogue_ID"].astype(int),
        "utterance_id": df["Utterance_ID"].astype(int),
        "speaker": df["Speaker"].astype(str).str.strip(),
        "emotion": df["Emotion"].astype(str).str.strip().str.lower(),
        "sentiment": df["Sentiment"].astype(str).str.strip().str.lower(),
        "gold_text": df["Utterance"].astype(str),
        "season": df["Season"].astype(int),
        "episode": df["Episode"].astype(int),
        "start_time": df["StartTime"],
        "end_time": df["EndTime"],
    })
    out["nominal_duration_s"] = [parse_ts(e) - parse_ts(s) for s, e in zip(out["start_time"], out["end_time"])]
    return out


def add_audio(df: pd.DataFrame, wav_dir: Path) -> pd.DataFrame:
    paths, durs, hashes, has = [], [], [], []
    for split, d, u in tqdm(zip(df["split"], df["dialogue_id"], df["utterance_id"]), total=len(df)):
        p = wav_dir / split / f"dia{d}_utt{u}.wav"
        if p.exists():
            info = sf.info(str(p))
            paths.append(str(p.relative_to(ROOT)))
            durs.append(info.frames / info.samplerate)
            hashes.append(sha256_of(p))
            has.append(True)
        else:
            paths.append("")
            durs.append(float("nan"))
            hashes.append("")
            has.append(False)
    df = df.copy()
    df["wav_path"], df["duration_s"], df["sha256"], df["has_audio"] = paths, durs, hashes, has
    return df


def mark_exclusions(dfs: dict[str, pd.DataFrame]) -> None:
    """Flag train rows that duplicate a dev/test row. Dev and test are never modified, so
    results stay comparable with published MELD numbers; the train copy is excluded by
    the loader. A duplicate is the same (season, episode, start, end) timestamp key, or,
    when audio is present, the same sha256."""
    key = ["season", "episode", "start_time", "end_time"]
    for df in dfs.values():
        df["exclude_reason"] = ""
    train = dfs["train"]
    for other in ["dev", "test"]:
        od = dfs[other]
        ts = {k: u for k, u in zip(zip(*[od[c] for c in key]), od["utt_id"])}
        for i, k in enumerate(zip(*[train[c] for c in key])):
            if k in ts and train.at[i, "exclude_reason"] == "":
                train.at[i, "exclude_reason"] = f"timestamp_duplicate_of:{ts[k]}"
        if "sha256" in train and "sha256" in od:
            hs = {h: u for h, u in zip(od["sha256"], od["utt_id"]) if h}
            for i, h in enumerate(train["sha256"]):
                if h and h in hs and train.at[i, "exclude_reason"] == "":
                    train.at[i, "exclude_reason"] = f"audio_duplicate_of:{hs[h]}"
    n = int((train["exclude_reason"] != "").sum())
    print(f"train rows marked excluded as duplicates of dev/test: {n}")


def apply_policy(dfs: dict[str, pd.DataFrame], min_s: float, max_clean_s: float) -> None:
    """Data policy from reports/meld_usability_audit.md Section 8. Train rows that are
    speech fragments (< min_s) or undecodable are excluded; dev/test rows are never
    excluded, but every split gets a clean_subset flag (decodable, min_s..max_clean_s)."""
    for split, df in dfs.items():
        if "has_audio" not in df:
            df["clean_subset"] = True
            continue
        df["clean_subset"] = df["has_audio"] & (df["duration_s"] >= min_s) & (df["duration_s"] <= max_clean_s)
        # Clips whose audio is byte-identical to another clip of the same split serve two
        # different utterances, so at least one is a mis-cut: out of the clean subset
        # everywhere, excluded from training.
        shared = df["has_audio"] & df["sha256"].duplicated(keep=False)
        df.loc[shared, "clean_subset"] = False
        if split == "train":
            first = df.loc[shared].groupby("sha256")["utt_id"].first()
            idx = shared & (df["exclude_reason"] == "")
            df.loc[idx, "exclude_reason"] = "shared_audio_group:" + df.loc[idx, "sha256"].map(first)
        if split == "train":
            frag = df["has_audio"] & (df["duration_s"] < min_s) & (df["exclude_reason"] == "")
            df.loc[frag, "exclude_reason"] = f"fragment_lt_{min_s}s"
            dead = ~df["has_audio"] & (df["exclude_reason"] == "")
            df.loc[dead, "exclude_reason"] = "no_audio"
        print(f"{split}: clean_subset {int(df['clean_subset'].sum())}/{len(df)}; excluded {int((df['exclude_reason'] != '').sum())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-only", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    out_dir = ROOT / cfg["manifest_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    dfs = {}
    for split in ["train", "dev", "test"]:
        df = load_labels(ROOT / cfg["labels"][split], split)
        if not args.labels_only:
            df = add_audio(df, ROOT / cfg["wav_dir"])
        dfs[split] = df
    mark_exclusions(dfs)
    apply_policy(dfs, cfg["min_clip_seconds"], cfg["max_clean_seconds"])
    for split, df in dfs.items():
        df.to_csv(out_dir / f"{split}.csv", index=False)
        msg = f"{split}: {len(df)} rows"
        if "has_audio" in df:
            msg += f", {int(df['has_audio'].sum())} with audio"
        print(msg)


if __name__ == "__main__":
    main()
