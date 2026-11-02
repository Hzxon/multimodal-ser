"""Feature cache files: one split per file, split name asserted on load (Gate F5).

Layout: ``data/processed/features/<extractor>_<version>/<split>.npz`` with arrays
``utt_id`` (str), ``X`` (float16/float32, first axis = rows in ``utt_id`` order) and the
scalar metadata ``split``, ``extractor``, ``version``. Optional extra arrays are stored
under their own keys.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common import ROOT, SPLITS

FEATURE_ROOT = ROOT / "data/processed/features"


def cache_path(extractor: str, version: str, split: str, root: Path | None = None) -> Path:
    assert split in SPLITS, split
    return (root or FEATURE_ROOT) / f"{extractor}_{version}" / f"{split}.npz"


def save_cache(path: Path, *, split: str, extractor: str, version: str,
               utt_ids: np.ndarray | list[str], X: np.ndarray, **extra: np.ndarray) -> Path:
    assert split in SPLITS, split
    utt_ids = np.asarray(utt_ids, dtype=str)
    assert len(utt_ids) == len(X), (len(utt_ids), len(X))
    assert len(set(utt_ids.tolist())) == len(utt_ids), "duplicate utt_id in cache"
    assert all(u.startswith(split + "_") for u in utt_ids), "utt_id from another split in cache"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, utt_id=utt_ids, X=X, split=np.array(split), extractor=np.array(extractor),
             version=np.array(version), **extra)
    return path


def load_cache(path: Path, *, split: str, extractor: str | None = None, version: str | None = None) -> dict:
    """Return ``{"utt_id": ..., "X": ..., <extra keys>, "meta": {...}}``; asserts the split."""
    with np.load(path, allow_pickle=False) as z:
        meta = {k: str(z[k]) for k in ("split", "extractor", "version")}
        if meta["split"] != split:
            raise AssertionError(f"{path} holds split {meta['split']!r}, requested {split!r}")
        if extractor is not None and meta["extractor"] != extractor:
            raise AssertionError(f"{path} extractor {meta['extractor']!r} != {extractor!r}")
        if version is not None and meta["version"] != version:
            raise AssertionError(f"{path} version {meta['version']!r} != {version!r}")
        out = {k: z[k] for k in z.files if k not in ("split", "extractor", "version")}
    out["utt_id"] = out["utt_id"].astype(str)
    assert all(u.startswith(split + "_") for u in out["utt_id"])
    out["meta"] = meta
    return out


def align(cache: dict, utt_ids: list[str] | np.ndarray) -> np.ndarray:
    """Rows of ``cache["X"]`` in the order of ``utt_ids``; raises on any missing id."""
    index = {u: i for i, u in enumerate(cache["utt_id"].tolist())}
    missing = [u for u in utt_ids if u not in index]
    if missing:
        raise KeyError(f"{len(missing)} utt_ids missing from cache, e.g. {missing[:3]}")
    return cache["X"][[index[u] for u in utt_ids]]
