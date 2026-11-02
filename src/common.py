"""Shared constants and helpers used by every stage after Phase 1.

Single source of truth for: repository root, split names, the label index
(``configs/data.yaml`` ``emotions`` order), the text normaliser, seeding, and the run
directory layout ``runs/<name>/<seed>/`` (PLAN.md Section 2).
"""
from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "dev", "test")
DATA_CFG_PATH = ROOT / "configs/data.yaml"
SEEDS = (13, 42, 2024)


def load_yaml(path: str | os.PathLike) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return yaml.safe_load(p.read_text())


def data_cfg() -> dict:
    return load_yaml(DATA_CFG_PATH)


EMOTIONS: list[str] = list(data_cfg()["emotions"])
LABEL2ID: dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL: dict[int, str] = {i: e for e, i in LABEL2ID.items()}
NUM_CLASSES = len(EMOTIONS)


def normalize_text(s: str) -> str:
    """The one normaliser for gold and ASR text (preprocessing_validation.md, Stage 4):
    unify curly quotes and dashes, collapse whitespace, keep case and punctuation.
    Lower-casing and punctuation stripping happen only inside TF-IDF and WER."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 ("–", "-"), ("—", "-"), (" ", " ")):
        s = s.replace(a, b)
    return " ".join(s.split())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_dir(name: str, seed: int, root: Path | None = None) -> Path:
    """``runs/<name>/<seed>/``; created on demand. ``root`` overrides ``runs/`` for smoke tests."""
    d = (root or ROOT / "runs") / name / str(seed)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
