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


# paths whose uncommitted changes make ``git_commit`` report ``-dirty``: the code and its
# configuration. Pipeline outputs (reports/, the local ASR cache under data/manifests/asr/) are left out
# because every stage rewrites them.
CODE_PATHS = ("src", "scripts", "configs", "app", "tests", "pyproject.toml", "uv.lock")


def git_commit() -> str:
    """HEAD hash, with ``-dirty`` appended when ``CODE_PATHS`` have uncommitted changes
    (``git status --porcelain -- CODE_PATHS`` non-empty), so the string identifies the code that ran."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        head = out.stdout.strip()
        if not head:
            return "unknown"
        st = subprocess.run(["git", "status", "--porcelain", "--", *CODE_PATHS], cwd=ROOT, capture_output=True, text=True)
        return head + "-dirty" if st.returncode == 0 and st.stdout.strip() else head
    except Exception:
        return "unknown"


def set_nested(cfg: dict, dotted: str, value: Any) -> None:
    """``cfg["a"]["b"] = value`` for ``dotted == "a.b"``, creating the dicts on the way (``--set``)."""
    d = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def classification_scores(y: np.ndarray, pred: np.ndarray) -> dict:
    """Weighted / macro F1 and accuracy over all ``NUM_CLASSES`` labels, plus per-class F1."""
    from sklearn.metrics import accuracy_score, f1_score

    labels = list(range(NUM_CLASSES))
    per_class = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    return {
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "per_class_f1": {e: float(v) for e, v in zip(EMOTIONS, per_class)},
    }


def rel_path(p: Path) -> str:
    """``p`` relative to the repository root when it lies inside it, else ``p`` unchanged."""
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


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
