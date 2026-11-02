"""eGeMAPSv02 functionals (88 features per clip) via openSMILE, plus the train-only scaler.

Usage: python -m src.features.egemaps --split {train,dev,test,all} [--manifest-dir DIR]
           [--feature-root DIR] [--limit N] [--jobs N] [--fit-scaler]

Cache: ``<feature-root>/egemaps_<version>/<split>.npz`` (raw, unscaled, float32) with the extra
array ``feature_names``. ``--fit-scaler`` fits mean/std on the train cache rows whose utt_id is
in ``load_manifest("train", apply_exclusions=True)`` and writes
``<feature-root>/egemaps_<version>/scaler.pkl`` (Gate F3). Normalisation is per feature over
the train set only; nothing is grouped or normalised per utterance source (Gate F4).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from src.common import ROOT, SPLITS
from src.data.dataset import load_manifest
from src.features.cache import FEATURE_ROOT, cache_path, load_cache, save_cache

EXTRACTOR = "egemaps"
N_FEATURES = 88
STD_FLOOR = 1e-8

_SMILE = None


def opensmile_version() -> str:
    import opensmile

    return opensmile.__version__


def feature_version() -> str:
    return f"v02-opensmile{opensmile_version()}"


def make_smile():
    import opensmile

    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def _init_worker() -> None:
    global _SMILE
    _SMILE = make_smile()


def extract_file(path: str, smile=None) -> np.ndarray:
    """88 functionals for one WAV, float32. ``path`` must be a 16 kHz mono file."""
    smile = smile or _SMILE
    sig, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == 16000, (path, sr)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    out = smile.process_signal(sig, sr)
    return out.to_numpy(dtype=np.float32).reshape(-1)


def _worker(path: str) -> np.ndarray:
    return extract_file(path)


def extract_split(wav_paths: list[str], jobs: int) -> np.ndarray:
    if jobs <= 1:
        smile = make_smile()
        rows = [extract_file(p, smile) for p in tqdm(wav_paths, desc="egemaps")]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(jobs, initializer=_init_worker) as pool:
            rows = list(tqdm(pool.imap(_worker, wav_paths, chunksize=4), total=len(wav_paths), desc="egemaps"))
    return np.stack(rows).astype(np.float32)


def fit_scaler_arrays(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X64 = X.astype(np.float64)
    mean = X64.mean(axis=0)
    std = X64.std(axis=0)
    std = np.where(std < STD_FLOOR, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_scaler(feature_root: Path, version: str, manifest_dir: str | None) -> Path:
    """Fit on non-excluded train rows of the train cache and write scaler.pkl."""
    cache = load_cache(cache_path(EXTRACTOR, version, "train", feature_root), split="train",
                       extractor=EXTRACTOR, version=version)
    keep = load_manifest("train", require_audio=True, apply_exclusions=True,
                         manifest_dir_override=manifest_dir)["utt_id"].tolist()
    in_cache = set(cache["utt_id"].tolist())
    missing = [u for u in keep if u not in in_cache]
    assert not missing, f"{len(missing)} non-excluded train rows missing from the cache, e.g. {missing[:3]}"
    keep_set = set(keep)
    mask = np.array([u in keep_set for u in cache["utt_id"]])
    X = cache["X"][mask]
    mean, std = fit_scaler_arrays(X)
    scaler = {
        "mean": mean,
        "std": std,
        "feature_names": [str(f) for f in cache["feature_names"]],
        "fit_split": "train",
        "n_fit": int(mask.sum()),
        "extractor": EXTRACTOR,
        "version": version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = scaler_path(feature_root, version)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    return path


def scaler_path(feature_root: Path, version: str) -> Path:
    return feature_root / f"{EXTRACTOR}_{version}" / "scaler.pkl"


def load_scaler(path: str | Path) -> dict:
    """Load scaler.pkl; asserts it was fitted on the train split (Gate F3)."""
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    assert scaler.get("fit_split") == "train", f"{path}: fit_split={scaler.get('fit_split')!r}, expected 'train'"
    return scaler


def apply_scaler(X: np.ndarray, scaler: dict) -> np.ndarray:
    return ((np.asarray(X, dtype=np.float64) - scaler["mean"]) / scaler["std"]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=[*SPLITS, "all"])
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--feature-root", default=str(FEATURE_ROOT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--fit-scaler", action="store_true")
    a = ap.parse_args()

    feature_root = Path(a.feature_root)
    if not feature_root.is_absolute():
        feature_root = ROOT / feature_root
    version = feature_version()
    splits = list(SPLITS) if a.split == "all" else [a.split]
    names: list[str] | None = None
    for split in splits:
        df = load_manifest(split, require_audio=True, apply_exclusions=False, manifest_dir_override=a.manifest_dir)
        if a.limit:
            df = df.head(a.limit)
        paths = [str(ROOT / p) for p in df["wav_path"]]
        t0 = time.time()
        X = extract_split(paths, a.jobs)
        dt = time.time() - t0
        if names is None:
            names = list(make_smile().feature_names)
        assert X.shape == (len(df), N_FEATURES) and len(names) == N_FEATURES, (X.shape, len(names))
        assert np.isfinite(X).all(), f"non-finite eGeMAPS values in {split}"
        out = save_cache(cache_path(EXTRACTOR, version, split, feature_root), split=split, extractor=EXTRACTOR,
                         version=version, utt_ids=df["utt_id"].to_numpy(), X=X, feature_names=np.array(names))
        print(f"{split}: {len(df)} clips -> {out} ({dt:.1f}s, {1000 * dt / max(len(df), 1):.0f} ms/clip, jobs={a.jobs})")
    if a.fit_scaler:
        p = fit_scaler(feature_root, version, a.manifest_dir)
        sc = load_scaler(p)
        print(f"scaler: {p} (fit_split={sc['fit_split']}, n_fit={sc['n_fit']})")


if __name__ == "__main__":
    main()
