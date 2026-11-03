"""Models 0-3 (PLAN.md 3.4): majority class, duration + token-count LR, TF-IDF + LR,
eGeMAPS + LR / MLP. scikit-learn only, trained on train, scored on dev.

Usage: python -m src.baselines --config configs/<name>.yaml [--seed 13] [--manifest-dir ...]
       [--feature-root ...] [--runs-root runs] [--limit N] [--name NAME] [--set key=value ...]

``cfg["model"]["type"]`` in {majority, shortcut_lr, tfidf_lr, egemaps_lr, egemaps_mlp}.
Writes runs/<name>/<seed>/{config.yaml, metrics.json, predictions.csv, model.pkl,
git_commit.txt}. The test split is never read here; ``src/evaluate.py`` unpickles
``model.pkl`` (a ``BaselineModel``) and calls ``predict_proba(df, split, feature_root)``.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.common import EMOTIONS, NUM_CLASSES, ROOT, git_commit, load_yaml, run_dir, set_seed, write_json
from src.data.dataset import class_weights, load_manifest
from src.features.cache import align, cache_path, load_cache

BASELINE_TYPES = ("majority", "shortcut_lr", "tfidf_lr", "egemaps_lr", "egemaps_mlp")
# same cache key as src.data.torch_dataset.EGEMAPS_CACHE (kept here so this module needs no torch)
EGEMAPS_CACHE = {"extractor": "egemaps", "version": "v02-opensmile2.6.0"}


def feature_root(cfg: dict) -> Path:
    p = Path(cfg["data"].get("feature_root", "data/processed/features"))
    return p if p.is_absolute() else ROOT / p


def load_split(split: str, cfg: dict, limit: int | None = None) -> pd.DataFrame:
    d = cfg["data"]
    ts = d.get("text_source", "gold")
    df = load_manifest(split, asr_tag=d.get("asr_tag") if ts == "asr" else None, text_source=ts,
                       manifest_dir_override=d.get("manifest_dir"))
    return df.iloc[:limit].reset_index(drop=True) if limit else df


def token_count(texts: pd.Series) -> np.ndarray:
    return np.array([len(str(t).split()) if not pd.isna(t) else 0 for t in texts], dtype=float)


def load_egemaps(split: str, utt_ids: list[str], root: Path, spec: dict) -> tuple[np.ndarray, dict]:
    c = load_cache(cache_path(spec["extractor"], spec["version"], split, root=root), split=split,
                   extractor=spec["extractor"], version=spec["version"])
    X = align(c, utt_ids).astype(np.float64)
    with open(root / f"{spec['extractor']}_{spec['version']}" / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    assert scaler.get("fit_split") == "train", f"scaler fit on {scaler.get('fit_split')!r} (Gate F3)"
    return (X - np.asarray(scaler["mean"])) / (np.asarray(scaler["std"]) + 1e-8), scaler


def vectorizer_dir(cfg: dict) -> Path:
    d = cfg["data"]
    ts = d.get("text_source", "gold")
    tag = d.get("asr_tag", "gold") if ts == "asr" else "gold"
    return feature_root(cfg) / f"tfidf_{ts}_{tag}"


class BaselineModel:
    """Pickled into ``model.pkl``: rebuilds the features for any split and predicts."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.type = cfg["model"]["type"]
        self.estimator: Any = None
        self.scaler: StandardScaler | None = None
        self.vectorizer: Any = None
        self.class_prior: np.ndarray | None = None
        self.egemaps_spec = {**EGEMAPS_CACHE, **cfg["data"].get("caches", {}).get("egemaps", {})}

    def features(self, df: pd.DataFrame, split: str, root: Path | None = None):
        root = root or feature_root(self.cfg)
        if self.type == "shortcut_lr":
            X = np.stack([df["duration_s"].to_numpy(float), np.log1p(df["duration_s"].to_numpy(float)),
                          token_count(df["text"]), np.log1p(token_count(df["text"]))], axis=1)
            return X
        if self.type == "tfidf_lr":
            return self.vectorizer.transform(["" if pd.isna(t) else str(t) for t in df["text"]])
        if self.type in ("egemaps_lr", "egemaps_mlp"):
            return load_egemaps(split, df["utt_id"].tolist(), root, self.egemaps_spec)[0]
        return None

    def fit(self, df: pd.DataFrame, split: str, seed: int) -> None:
        m = self.cfg["model"]
        y = df["label"].to_numpy()
        self.class_prior = np.bincount(y, minlength=NUM_CLASSES) / len(y)
        if self.type == "majority":
            return
        if self.type == "tfidf_lr":
            vpath = vectorizer_dir(self.cfg) / "vectorizer.pkl"
            with open(vpath, "rb") as f:
                obj = pickle.load(f)
            # src/features/text_embed.py pickles {"vectorizer": ..., "fit_split": ..., ...}
            meta = obj if isinstance(obj, dict) else {}
            self.vectorizer = obj["vectorizer"] if isinstance(obj, dict) else obj
            meta_path = vectorizer_dir(self.cfg) / "metadata.json"
            if meta_path.exists():
                meta = {**meta, **json.loads(meta_path.read_text())}
            assert meta.get("fit_split", "train") == "train", f"{vpath} fit on {meta.get('fit_split')!r} (Gate F3)"
        X = self.features(df, split)
        if self.type == "shortcut_lr":
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        cw = class_weights(y, m.get("class_weight", "sqrt_inv"))
        weight_dict = {c: float(w) for c, w in enumerate(cw)}
        if self.type == "egemaps_mlp":
            self.estimator = MLPClassifier(hidden_layer_sizes=tuple(m.get("hidden", [128])), alpha=float(m.get("alpha", 1e-3)),
                                           max_iter=int(m.get("max_iter", 300)), random_state=seed, early_stopping=False)
            sw = cw[y]
            if "sample_weight" in inspect.signature(self.estimator.fit).parameters:
                self.estimator.fit(X, y, sample_weight=sw)
            else:
                self.estimator.fit(X, y)
                self.cfg.setdefault("notes", []).append("MLPClassifier.fit has no sample_weight; trained unweighted")
        else:
            self.estimator = LogisticRegression(C=float(m.get("C", 1.0)), class_weight=weight_dict,
                                                max_iter=int(m.get("max_iter", 2000)), random_state=seed)
            self.estimator.fit(X, y)

    def predict_proba(self, df: pd.DataFrame, split: str, root: Path | None = None) -> np.ndarray:
        if self.type == "majority":
            return np.tile(self.class_prior, (len(df), 1))
        X = self.features(df, split, root)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        p = self.estimator.predict_proba(X)
        out = np.zeros((len(df), NUM_CLASSES))
        out[:, self.estimator.classes_] = p
        return out


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    labels = list(range(NUM_CLASSES))
    pc = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    return {"weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)), "per_class_f1": {e: float(v) for e, v in zip(EMOTIONS, pc)}}


def set_nested(cfg: dict, dotted: str, value: Any) -> None:
    d = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--manifest-dir")
    ap.add_argument("--feature-root")
    ap.add_argument("--runs-root")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--name")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = copy.deepcopy(load_yaml(args.config))
    if args.name:
        cfg["name"] = args.name
    if args.manifest_dir:
        cfg["data"]["manifest_dir"] = args.manifest_dir
    if args.feature_root:
        cfg["data"]["feature_root"] = args.feature_root
    if args.limit:
        cfg["data"]["limit"] = args.limit
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        set_nested(cfg, k, yaml.safe_load(v))
    assert cfg["model"]["type"] in BASELINE_TYPES, cfg["model"]["type"]
    splits = cfg["data"]["splits"]
    assert set(splits) == {"train", "dev"} and not any("test" in str(v) for v in splits.values()), splits
    set_seed(args.seed)
    runs_root = Path(args.runs_root) if args.runs_root else None
    if runs_root is not None and not runs_root.is_absolute():
        runs_root = ROOT / runs_root
    out = run_dir(cfg["name"], args.seed, root=runs_root)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "git_commit.txt").write_text(git_commit() + "\n")

    t0 = time.time()
    limit = cfg["data"].get("limit")
    train_df = load_split(splits["train"], cfg, limit)
    dev_df = load_split(splits["dev"], cfg, limit)
    # pickle the class under its importable name, not __main__, so evaluate.py can unpickle model.pkl
    from src.baselines import BaselineModel as ImportableBaselineModel

    model = ImportableBaselineModel(cfg)
    model.fit(train_df, splits["train"], args.seed)
    p_train = model.predict_proba(train_df, splits["train"])
    p_dev = model.predict_proba(dev_df, splits["dev"])
    y_dev = dev_df["label"].to_numpy()
    dev = scores(y_dev, p_dev.argmax(1))
    tr = scores(train_df["label"].to_numpy(), p_train.argmax(1))
    pred = pd.DataFrame({"utt_id": dev_df["utt_id"], "label": y_dev, "pred": p_dev.argmax(1)})
    for c in range(NUM_CLASSES):
        pred[f"p_{c}"] = p_dev[:, c]
    pred.to_csv(out / "predictions.csv", index=False)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    cw_scheme = cfg["model"].get("class_weight", "sqrt_inv")
    res = {"name": cfg["name"], "seed": args.seed, "type": model.type, "dev": dev, "train": tr,
           "n_train": len(train_df), "n_dev": len(dev_df), "class_weight_scheme": cw_scheme,
           "class_weights": {e: float(w) for e, w in zip(EMOTIONS, class_weights(train_df["label"].to_numpy(), cw_scheme))},
           "class_prior_train": {e: float(v) for e, v in zip(EMOTIONS, model.class_prior)},
           "notes": cfg.get("notes", []), "wall_seconds": time.time() - t0, "git_commit": git_commit()}
    if model.type == "tfidf_lr":
        res["vectorizer"] = str(vectorizer_dir(cfg) / "vectorizer.pkl")
        res["n_features"] = int(len(model.vectorizer.get_feature_names_out()))
    write_json(out / "metrics.json", res)
    print(f"{cfg['name']} seed {args.seed}: dev weighted F1 {dev['weighted_f1']:.3f} macro F1 {dev['macro_f1']:.3f} "
          f"acc {dev['accuracy']:.3f} | train weighted F1 {tr['weighted_f1']:.3f} | n_train {len(train_df)} n_dev {len(dev_df)} "
          f"({time.time() - t0:.1f}s) -> {out}")


if __name__ == "__main__":
    main()
