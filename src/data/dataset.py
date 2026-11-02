"""Manifest loading with the data policy applied, and the ASR-transcript join.

Every stage after Phase 1 reads the manifests through ``load_manifest`` so the policy
(``exclude_reason``, ``has_audio``) is applied in exactly one place.

ASR transcripts live next to the manifests, one file per model tag and split:
``data/manifests/asr/<asr_tag>/<split>.csv`` with columns
``utt_id, asr_text, asr_model, asr_flag``. They are joined at load time instead of
being written into the frozen manifests, so Phase 1 files never change.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common import EMOTIONS, LABEL2ID, ROOT, SPLITS, data_cfg, normalize_text

STR_COLS = {c: "string" for c in ["exclude_reason", "sha256", "wav_path", "gold_text", "speaker"]}


def manifest_dir(override: str | Path | None = None) -> Path:
    if override is not None:
        p = Path(override)
        return p if p.is_absolute() else ROOT / p
    return ROOT / data_cfg()["manifest_dir"]


def asr_path(asr_tag: str, split: str, mdir: str | Path | None = None) -> Path:
    return manifest_dir(mdir) / "asr" / asr_tag / f"{split}.csv"


def load_manifest(
    split: str,
    *,
    require_audio: bool = True,
    apply_exclusions: bool = True,
    asr_tag: str | None = None,
    text_source: str = "gold",
    manifest_dir_override: str | Path | None = None,
) -> pd.DataFrame:
    """Return the rows of one split that the data policy allows.

    - ``apply_exclusions``: drop rows with a non-empty ``exclude_reason`` (train only has them).
    - ``require_audio``: drop rows whose WAV is missing; raises if the manifest has no
      audio columns yet (built with ``--labels-only``).
    - ``asr_tag``: join ``asr_text``/``asr_model``/``asr_flag`` from the ASR file; raises if
      any kept row has no transcript.
    - ``text_source``: ``"gold"`` or ``"asr"``; sets the ``text`` column (normalised).
    Adds ``label`` (int, index into ``configs/data.yaml`` ``emotions``).
    """
    assert split in SPLITS, split
    assert text_source in ("gold", "asr"), text_source
    path = manifest_dir(manifest_dir_override) / f"{split}.csv"
    df = pd.read_csv(path, dtype=STR_COLS)
    df["exclude_reason"] = df["exclude_reason"].fillna("") if "exclude_reason" in df else ""
    if apply_exclusions:
        df = df[df["exclude_reason"] == ""]
    if require_audio:
        if "has_audio" not in df:
            raise RuntimeError(f"{path} has no audio columns; run `python -m src.data.build_manifest` first")
        df = df[df["has_audio"].astype(bool)]
    if "clean_subset" not in df:
        df = df.assign(clean_subset=True)
    df = df.reset_index(drop=True)
    assert df["utt_id"].is_unique
    assert df["split"].eq(split).all()
    unknown = set(df["emotion"]) - set(EMOTIONS)
    assert not unknown, f"unknown emotions {unknown}"
    df["label"] = df["emotion"].map(LABEL2ID).astype(int)
    df["gold_text"] = df["gold_text"].map(normalize_text)
    if asr_tag is not None:
        asr = pd.read_csv(asr_path(asr_tag, split, manifest_dir_override), dtype={"asr_text": "string", "asr_flag": "string"})
        asr["asr_text"] = asr["asr_text"].fillna("").map(normalize_text)
        asr["asr_flag"] = asr["asr_flag"].fillna("") if "asr_flag" in asr else ""
        df = df.merge(asr[["utt_id", "asr_text", "asr_model", "asr_flag"]], on="utt_id", how="left")
        missing = df["asr_text"].isna()
        if missing.any():
            raise RuntimeError(f"{int(missing.sum())} {split} rows have no ASR transcript for tag {asr_tag!r}")
    if text_source == "asr":
        if asr_tag is None:
            raise ValueError("text_source='asr' needs asr_tag")
        df["text"] = df["asr_text"]
    else:
        df["text"] = df["gold_text"]
    return df


def class_weights(labels: np.ndarray | pd.Series, scheme: str = "sqrt_inv") -> np.ndarray:
    """Loss weights per class from the *train* labels (PLAN.md Section 3.4 / 5).

    ``sqrt_inv``: (N / (K * n_c)) ** 0.5; ``inv``: N / (K * n_c); ``none``: ones.
    """
    y = np.asarray(labels)
    counts = np.bincount(y, minlength=len(EMOTIONS)).astype(float)
    counts = np.maximum(counts, 1.0)
    n, k = counts.sum(), len(EMOTIONS)
    if scheme == "none":
        return np.ones(k)
    if scheme == "inv":
        return n / (k * counts)
    if scheme == "sqrt_inv":
        return np.sqrt(n / (k * counts))
    raise ValueError(scheme)


def wav_abs_path(rel: str) -> Path:
    return ROOT / rel
