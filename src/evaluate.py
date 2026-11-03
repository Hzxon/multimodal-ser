"""Phase 5 evaluation (PLAN.md 3.5, Gate 5 V1-V8). The only module that reads the test manifest.

Usage: python -m src.evaluate run --run runs/<name>/<seed> [--manifest-dir DIR] [--feature-root DIR]
           [--device cuda|mps|cpu] [--batch-size 16] [--limit N] [--noise-snr 20,10]
           [--text-conditions] [--config configs/fusion_late.yaml] [--set key=value ...] [--force]
       python -m src.evaluate summary --runs-root runs [--out reports/results.md] [--fusion auto|fusion_gated]
           [--unimodal text_roberta,audio_wavlm_ft] [--literature reports/literature.yaml]
           [--manifest-dir DIR] [--n-errors 30] [--seed 0]

``run`` evaluates one run directory on the test split once and writes ``test_metrics.json``
(ISO timestamp, git commit, ``n_test_evaluations``) and ``test_predictions.csv`` next to the
run's ``best.pt`` / ``model.pkl``. A second evaluation is refused without ``--force`` (Gate V1).
``--noise-snr`` re-runs inference with additive Gaussian noise at fixed SNRs (V6; only for
models that read the waveform). ``--text-conditions`` evaluates text and fusion models on both
gold and ASR transcripts (V5). A ``fusion_late`` run directory (configs/fusion_late.yaml) is
created on demand: the weight is tuned on the two unimodal runs' dev ``predictions.csv`` and
the test score comes from their ``test_predictions.csv``, so those two runs are evaluated first.

``summary`` collects every ``test_metrics.json`` under ``--runs-root`` into a Markdown report
(results ladder with mean +- std and CIs, McNemar, transcript 2x2, noise, per-class,
class-weight ablation, dev gain and train-dev gap, literature, error analysis) plus
``gate5_eval.md`` and ``error_analysis.csv`` in the same directory and confusion-matrix figures
under ``<out dir>/figures/``. The class-weight ablation (M7) and the final-model choice read
the dev scores of every ``metrics.json`` (ablation arms are never test-evaluated).
``--fusion auto`` (default) picks the fusion family with the best mean dev weighted F1 among
the test-evaluated fusion runs, skipping families the Section 5 rule rejects (dev macro F1
below their ``_cwnone`` sibling) when another candidate survives; ``--fusion <name>`` forces
one. McNemar runs for every fusion family present; M6, M8 and the error analysis use the
chosen one.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.common import (EMOTIONS, LABEL2ID, NUM_CLASSES, ROOT, device_name, git_commit, load_yaml, rel_path, set_nested,
                        write_json)
from src.data.dataset import asr_path, load_manifest

LADDER = ["majority", "shortcut_lr", "text_tfidf", "text_tfidf_gold", "audio_egemaps_lr", "audio_egemaps_mlp",
          "text_roberta", "text_roberta_gold", "audio_wavlm_cached", "audio_wavlm_ft", "fusion_late",
          "fusion_gated", "fusion_xattn"]
TEXT_TYPES = {"text_roberta", "tfidf_lr", "shortcut_lr"}
AUDIO_TYPES = {"audio_wavlm", "egemaps_lr", "egemaps_mlp"}
FUSION_TYPES = {"fusion_gated", "fusion_xattn", "fusion_late"}
PROB_COLS = [f"p_{c}" for c in range(NUM_CLASSES)]
# run-name suffixes of the class-balance ablation arms (PLAN 5): ``<name>_cwnone`` is the
# unweighted-CE baseline every other arm is judged against
ABLATION_SUFFIXES = ("_cwnone", "_cwinv", "_cwfocal", "_undersample")
CONDITION_SUFFIXES = ("_gold",)  # transcript-condition twin, not a separate model family


# ----------------------------------------------------------------------------- metrics

def _f1_from_cm(cm: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """(per-class F1, weighted F1, macro F1, accuracy) from a K x K count matrix (rows = true)."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    predicted = cm.sum(0).astype(float)
    denom = 2 * tp + (predicted - tp) + (support - tp)
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1.0), 0.0)
    n = support.sum()
    weighted = float((f1 * support).sum() / n) if n else 0.0
    return f1, weighted, float(f1.mean()), float(tp.sum() / n) if n else 0.0


def _cm(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.bincount(y * NUM_CLASSES + pred, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_dict(y: np.ndarray, pred: np.ndarray, probs: np.ndarray | None = None) -> dict:
    """Weighted/macro F1, accuracy, per-class precision/recall/F1/support, row-normalised
    confusion matrix, fear and disgust recall (M6) and, with ``probs``, the mean NLL."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    labels = list(range(NUM_CLASSES))
    out: dict[str, Any] = {"n": int(len(y))}
    if len(y) == 0:
        return out
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = _cm(y, pred)
    row = cm.sum(1, keepdims=True).astype(float)
    out.update({
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "per_class": {e: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, e in enumerate(EMOTIONS)},
        "confusion_counts": cm.tolist(),
        "confusion_row_norm": np.where(row > 0, cm / np.where(row > 0, row, 1.0), 0.0).round(4).tolist(),
        "fear_recall": float(r[LABEL2ID["fear"]]),
        "disgust_recall": float(r[LABEL2ID["disgust"]]),
        "min_class_recall": float(min(r[i] for i in labels if s[i] > 0)) if s.sum() else 0.0,
        "classes_with_zero_recall": [e for i, e in enumerate(EMOTIONS) if s[i] > 0 and r[i] == 0],
    })
    if probs is not None:
        pr = np.clip(np.asarray(probs, dtype=float)[np.arange(len(y)), y], 1e-12, 1.0)
        out["nll"] = float(-np.log(pr).mean())
    return out


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, metric: str | Callable = "weighted_f1",
                 n: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap over utterances (V2). ``metric`` is ``weighted_f1``, ``macro_f1``,
    ``accuracy`` or a callable ``(y, pred) -> float``. Returns ``{point, lo, hi, n_resamples}``."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    rng = np.random.default_rng(seed)
    N = len(y)
    idx_metric = {"weighted_f1": 1, "macro_f1": 2, "accuracy": 3}
    if callable(metric):
        point = float(metric(y, pred))
        vals = np.array([metric(y[i], pred[i]) for i in (rng.integers(0, N, N) for _ in range(n))])
    else:
        k = idx_metric[metric]
        point = float(_f1_from_cm(_cm(y, pred))[k])
        vals = np.empty(n)
        for b in range(n):
            i = rng.integers(0, N, N)
            vals[b] = _f1_from_cm(_cm(y[i], pred[i]))[k]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_resamples": int(n)}


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """McNemar's test on paired correctness vectors (V3). Exact binomial when the number of
    discordant pairs is below 25, otherwise chi-square with continuity correction; both
    p-values are returned. ``n01`` = a correct and b wrong, ``n10`` = a wrong and b correct."""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    assert a.shape == b.shape, (a.shape, b.shape)
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    disc = n01 + n10
    if disc == 0:
        p_exact, stat, p_chi2 = 1.0, 0.0, 1.0
    else:
        p_exact = float(min(1.0, 2.0 * stats.binom.cdf(min(n01, n10), disc, 0.5)))
        stat = (abs(n01 - n10) - 1) ** 2 / disc
        p_chi2 = float(stats.chi2.sf(stat, 1))
    method = "exact" if disc < 25 else "chi2_cc"
    return {"n": int(len(a)), "n01": n01, "n10": n10, "n_discordant": disc, "acc_a": float(a.mean()) if len(a) else 0.0,
            "acc_b": float(b.mean()) if len(b) else 0.0, "statistic": float(stat), "p_exact": p_exact, "p_chi2_cc": p_chi2,
            "method": method, "p_value": p_exact if method == "exact" else p_chi2}


# ----------------------------------------------------------------------------- run helpers

def resolve_path(p: str | Path | None, default: str | Path | None = None) -> Path | None:
    if p is None:
        p = default
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def run_kind(run: Path, cfg: dict) -> str:
    if cfg["model"]["type"] == "fusion_late":
        return "late"
    if (run / "model.pkl").exists():
        return "baseline"
    if (run / "best.pt").exists():
        return "torch"
    raise FileNotFoundError(f"{run}: neither model.pkl nor best.pt")


def model_modality(cfg: dict) -> str:
    t = cfg["model"]["type"]
    if t in TEXT_TYPES:
        return "text"
    if t in AUDIO_TYPES:
        return "audio"
    if t in FUSION_TYPES:
        return "fusion"
    return "none"


def reads_waveform(cfg: dict) -> bool:
    m = cfg["model"]
    if m["type"] == "audio_wavlm":
        return m.get("mode", "cached") == "finetune"
    if m["type"] in ("fusion_gated", "fusion_xattn"):
        return m["audio"].get("mode", "cached") == "finetune"
    return False


def test_frame(cfg: dict, manifest_dir: str | Path | None, text_source: str | None = None,
               limit: int | None = None) -> pd.DataFrame:
    """Test rows with the run's ASR tag joined when the ASR file exists."""
    d = cfg["data"]
    ts = text_source or d.get("text_source", "gold")
    tag = d.get("asr_tag")
    mdir = manifest_dir if manifest_dir is not None else d.get("manifest_dir")
    if tag is not None and not asr_path(tag, "test", mdir).exists():
        if ts == "asr":
            raise FileNotFoundError(f"no test ASR file for tag {tag!r} under {mdir}")
        tag = None
    df = load_manifest("test", require_audio=True, asr_tag=tag, text_source=ts, manifest_dir_override=mdir)
    if limit:
        df = df.head(limit).reset_index(drop=True)
    return df


def predictions_frame(ids: list[str], y: np.ndarray, probs: np.ndarray, clean: np.ndarray | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"utt_id": ids, "label": np.asarray(y, dtype=int), "pred": probs.argmax(1)})
    for c in range(NUM_CLASSES):
        df[f"p_{c}"] = probs[:, c]
    if clean is not None:
        df["clean_subset"] = np.asarray(clean, dtype=bool)
    return df


def predict_torch(model, cfg: dict, df: pd.DataFrame, device: str, batch_size: int, feature_root: Path | None,
                  tokenizer, transforms: list | None = None) -> np.ndarray:
    """Softmax probabilities [N, 7] of a torch run on ``df`` (test split). ``transforms`` are
    applied to the waveform before the eval centre crop (noise robustness)."""
    import torch

    from src.data.torch_dataset import MELDDataset, collate

    ds = MELDDataset(df, "test", cfg, train=False, feature_root=feature_root)
    ds.transforms = list(transforms or [])
    max_len = int(cfg.get("train", {}).get("max_text_tokens", 64))
    out = []
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            batch = collate([ds[i] for i in range(start, min(start + batch_size, len(ds)))], tokenizer, max_len)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(batch)
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def score_block(y: np.ndarray, probs: np.ndarray, clean: np.ndarray, seed: int = 0, n_boot: int = 1000) -> dict:
    pred = probs.argmax(1)
    block = metrics_dict(y, pred, probs)
    block["bootstrap"] = {m: bootstrap_ci(y, pred, m, n=n_boot, seed=seed) for m in ("weighted_f1", "macro_f1")}
    c = np.asarray(clean, dtype=bool)
    sub = metrics_dict(y[c], pred[c], probs[c])
    if c.sum():
        sub["bootstrap"] = {m: bootstrap_ci(y[c], pred[c], m, n=n_boot, seed=seed) for m in ("weighted_f1", "macro_f1")}
    return {"test": block, "clean_subset": sub}


def load_baseline(run: Path):
    import src.baselines  # noqa: F401  (unpickling needs the class importable)

    with open(run / "model.pkl", "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------- late fusion

def tune_late_fusion(audio_run: Path, text_run: Path, grid: list[float]) -> tuple[float, list[dict], pd.DataFrame]:
    """Weight search on the two runs' dev predictions (model 6)."""
    from src.models.fusion import late_fusion_probs

    pa = pd.read_csv(audio_run / "predictions.csv")
    pt = pd.read_csv(text_run / "predictions.csv")
    m = pa.merge(pt, on="utt_id", suffixes=("_a", "_t"))
    assert len(m) == len(pa) == len(pt), f"dev predictions differ: audio {len(pa)} text {len(pt)} shared {len(m)}"
    assert (m["label_a"] == m["label_t"]).all()
    y = m["label_a"].to_numpy()
    rows = []
    for w in grid:
        p = late_fusion_probs(m[[f"p_{c}_a" for c in range(NUM_CLASSES)]].to_numpy(),
                              m[[f"p_{c}_t" for c in range(NUM_CLASSES)]].to_numpy(), float(w))
        md = metrics_dict(y, p.argmax(1))
        rows.append({"w_audio": float(w), "dev_weighted_f1": md["weighted_f1"], "dev_macro_f1": md["macro_f1"], "dev_accuracy": md["accuracy"]})
    best = max(rows, key=lambda r: (r["dev_weighted_f1"], r["dev_macro_f1"]))
    return best["w_audio"], rows, m


def combine_predictions(audio_csv: Path, text_csv: Path, w: float) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    from src.models.fusion import late_fusion_probs

    pa = pd.read_csv(audio_csv)
    pt = pd.read_csv(text_csv)
    m = pa.merge(pt, on="utt_id", suffixes=("_a", "_t"))
    assert len(m) == len(pa) == len(pt), f"test predictions differ: audio {len(pa)} text {len(pt)} shared {len(m)}"
    assert (m["label_a"] == m["label_t"]).all()
    p = late_fusion_probs(m[[f"p_{c}_a" for c in range(NUM_CLASSES)]].to_numpy(),
                          m[[f"p_{c}_t" for c in range(NUM_CLASSES)]].to_numpy(), w)
    clean = m["clean_subset_a"].to_numpy() if "clean_subset_a" in m else np.ones(len(m), dtype=bool)
    return m["utt_id"].tolist(), m["label_a"].to_numpy(), p, clean


def evaluate_late(run: Path, cfg: dict, args: argparse.Namespace) -> dict:
    runs_root = run.parent.parent
    seed = run.name
    m = cfg["model"]
    audio_run = runs_root / m["audio_run"] / seed
    text_run = runs_root / m["text_run"] / seed
    for r in (audio_run, text_run):
        assert (r / "predictions.csv").exists(), f"{r}: no dev predictions.csv"
        assert (r / "test_predictions.csv").exists(), f"{r}: evaluate this run on test first"
    w, grid_rows, dev_m = tune_late_fusion(audio_run, text_run, list(m.get("weight_grid", [i / 10 for i in range(11)])))
    from src.models.fusion import late_fusion_probs

    p_dev = late_fusion_probs(dev_m[[f"p_{c}_a" for c in range(NUM_CLASSES)]].to_numpy(),
                              dev_m[[f"p_{c}_t" for c in range(NUM_CLASSES)]].to_numpy(), w)
    y_dev = dev_m["label_a"].to_numpy()
    dev = metrics_dict(y_dev, p_dev.argmax(1))
    predictions_frame(dev_m["utt_id"].tolist(), y_dev, p_dev).to_csv(run / "predictions.csv", index=False)
    write_json(run / "metrics.json", {
        "name": cfg["name"], "seed": int(seed), "type": "fusion_late", "weight_audio": w, "weight_grid": grid_rows,
        "audio_run": rel_path(audio_run), "text_run": rel_path(text_run),
        "dev": {"weighted_f1": dev["weighted_f1"], "macro_f1": dev["macro_f1"], "accuracy": dev["accuracy"],
                "per_class_f1": {e: v["f1"] for e, v in dev["per_class"].items()}},
        "n_dev": int(len(y_dev)), "dev_evaluations": len(grid_rows), "git_commit": git_commit()})
    (run / "git_commit.txt").write_text(git_commit() + "\n")
    print(f"late fusion weight (audio) {w:.2f}: dev weighted F1 {dev['weighted_f1']:.4f} macro F1 {dev['macro_f1']:.4f}")

    ids, y, probs, clean = combine_predictions(audio_run / "test_predictions.csv", text_run / "test_predictions.csv", w)
    predictions_frame(ids, y, probs, clean).to_csv(run / "test_predictions.csv", index=False)
    res = {"weight_audio": w, **score_block(y, probs, clean, seed=args.seed)}
    text_cfg = load_yaml(text_run / "config.yaml")
    trained_on = text_cfg["data"].get("text_source", "gold")
    other = "asr" if trained_on == "gold" else "gold"
    cross = text_run / f"test_predictions_{other}.csv"
    if cross.exists():
        ids2, y2, p2, c2 = combine_predictions(audio_run / "test_predictions.csv", cross, w)
        predictions_frame(ids2, y2, p2, c2).to_csv(run / f"test_predictions_{other}.csv", index=False)
        res["transcript_conditions"] = {"trained_on": trained_on, trained_on: metrics_dict(y, probs.argmax(1)),
                                        other: metrics_dict(y2, p2.argmax(1))}
    else:
        res["transcript_conditions"] = {"skipped": f"{cross.name} missing; evaluate the text run with --text-conditions"}
    noise = {}
    for f in sorted(audio_run.glob("test_predictions_snr*.csv")):
        snr = f.stem.replace("test_predictions_snr", "")
        ids3, y3, p3, c3 = combine_predictions(f, text_run / "test_predictions.csv", w)
        predictions_frame(ids3, y3, p3, c3).to_csv(run / f.name, index=False)
        noise[snr] = metrics_dict(y3, p3.argmax(1), p3)
    res["noise"] = noise or {"skipped": "audio run has no test_predictions_snr*.csv"}
    return res


# ----------------------------------------------------------------------------- run

def evaluate_run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    run = resolve_path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "config.yaml").exists():
        cfg0 = load_yaml(args.config)
        assert cfg0["model"]["type"] == "fusion_late", f"{run} has no config.yaml and {args.config} is not a fusion_late config"
        for kv in args.set or []:
            k, v = kv.split("=", 1)
            set_nested(cfg0, k, yaml.safe_load(v))
        (run / "config.yaml").write_text(yaml.safe_dump(cfg0, sort_keys=False))
    cfg = copy.deepcopy(load_yaml(run / "config.yaml"))
    out_json = run / "test_metrics.json"
    n_prev = 0
    if out_json.exists():
        prev = json.loads(out_json.read_text())
        n_prev = int(prev.get("n_test_evaluations", 1))
        if not args.force:
            raise SystemExit(f"{out_json} exists (evaluated {n_prev}x at {prev.get('timestamp')}); Gate V1 allows one test "
                             f"pass per run. Re-run with --force to record another evaluation.")
        print(f"WARNING: re-evaluating {run} on test (previous evaluations: {n_prev})")
    kind = run_kind(run, cfg)
    if args.manifest_dir:
        cfg["data"]["manifest_dir"] = args.manifest_dir
    if args.feature_root:
        cfg["data"]["feature_root"] = args.feature_root
    feature_root = resolve_path(cfg["data"].get("feature_root"))
    device = args.device or device_name()
    trained_on = cfg["data"].get("text_source", "gold")
    modality = model_modality(cfg)
    res: dict[str, Any] = {"name": cfg["name"], "seed": int(run.name) if run.name.isdigit() else run.name,
                           "run_dir": rel_path(run), "kind": kind,
                           "model_type": cfg["model"]["type"], "modality": modality, "text_source": trained_on,
                           "asr_tag": cfg["data"].get("asr_tag"), "limit": args.limit, "device": device}
    if kind == "late":
        res.update(evaluate_late(run, cfg, args))
    else:
        df = test_frame(cfg, args.manifest_dir, limit=args.limit)
        y = df["label"].to_numpy()
        clean = df["clean_subset"].to_numpy().astype(bool)
        tokenizer = model = None
        if kind == "baseline":
            model = load_baseline(run)
            predict = lambda frame, transforms=None: model.predict_proba(frame, "test", feature_root)  # noqa: E731
        else:
            from src.models import load_run, tokenizer_name

            model, _ = load_run(run, device)
            tok = tokenizer_name(cfg)
            if tok:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(tok)
            predict = lambda frame, transforms=None: predict_torch(model, cfg, frame, device, args.batch_size, feature_root,  # noqa: E731
                                                                   tokenizer, transforms)
        t1 = time.time()
        probs = predict(df)
        res["seconds_inference"] = time.time() - t1
        predictions_frame(df["utt_id"].tolist(), y, probs, clean).to_csv(run / "test_predictions.csv", index=False)
        res.update(score_block(y, probs, clean, seed=args.seed))
        res["n_cropped_test"] = None

        snrs = [float(s) for s in args.noise_snr.split(",")] if args.noise_snr else []
        if snrs:
            if kind == "torch" and reads_waveform(cfg):
                from src.data.augment import AddNoise

                noise = {}
                for snr in snrs:
                    t1 = time.time()
                    tr = AddNoise((snr, snr), rng=np.random.default_rng(args.seed))
                    p_n = predict(df, [tr])
                    key = f"{snr:g}"
                    predictions_frame(df["utt_id"].tolist(), y, p_n, clean).to_csv(run / f"test_predictions_snr{key}.csv", index=False)
                    noise[key] = metrics_dict(y, p_n.argmax(1), p_n)
                    noise[key]["seconds"] = time.time() - t1
                    print(f"noise {snr:g} dB: weighted F1 {noise[key]['weighted_f1']:.4f} (clean {res['test']['weighted_f1']:.4f})")
                res["noise"] = noise
            elif modality in ("audio", "fusion"):
                res["noise"] = {"skipped": "model reads cached features (audio.mode: cached or eGeMAPS); noise robustness "
                                           "needs the finetune (waveform) path"}
            else:
                res["noise"] = {"skipped": "text-only model has no audio input"}
        if args.text_conditions:
            if modality in ("text", "fusion") and cfg["model"]["type"] != "majority":
                other = "asr" if trained_on == "gold" else "gold"
                try:
                    df2 = test_frame(cfg, args.manifest_dir, text_source=other, limit=args.limit)
                    t1 = time.time()
                    p2 = predict(df2)
                    predictions_frame(df2["utt_id"].tolist(), y, p2, clean).to_csv(run / f"test_predictions_{other}.csv", index=False)
                    m2 = metrics_dict(y, p2.argmax(1), p2)
                    m2["seconds"] = time.time() - t1
                    res["transcript_conditions"] = {"trained_on": trained_on, trained_on: metrics_dict(y, probs.argmax(1), probs), other: m2}
                    print(f"transcript condition {other}: weighted F1 {m2['weighted_f1']:.4f} (trained on {trained_on}: {res['test']['weighted_f1']:.4f})")
                except FileNotFoundError as e:
                    res["transcript_conditions"] = {"skipped": str(e)}
            else:
                res["transcript_conditions"] = {"skipped": "model has no transcript input"}
    res.update({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
                "n_test_evaluations": n_prev + 1, "wall_seconds": time.time() - t0})
    write_json(out_json, res)
    t = res["test"]
    b = t["bootstrap"]["weighted_f1"]
    print(f"{res['name']} seed {res['seed']} [{kind}] test n={t['n']}: weighted F1 {t['weighted_f1']:.4f} "
          f"[{b['lo']:.4f}, {b['hi']:.4f}] macro F1 {t['macro_f1']:.4f} acc {t['accuracy']:.4f} "
          f"fear recall {t['fear_recall']:.3f} disgust recall {t['disgust_recall']:.3f} | clean n={res['clean_subset']['n']} "
          f"wF1 {res['clean_subset'].get('weighted_f1', float('nan')):.4f} ({res['wall_seconds']:.1f}s) -> {out_json}")
    return res


# ----------------------------------------------------------------------------- summary

def _get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def load_run_results(runs_root: str | Path) -> pd.DataFrame:
    """One row per run directory holding a ``test_metrics.json``; dev/train numbers come from
    ``metrics.json``, the full test block is kept in the ``test_metrics`` column."""
    root = resolve_path(runs_root)
    rows = []
    for tm_path in sorted(root.glob("*/*/test_metrics.json")):
        run = tm_path.parent
        tm = json.loads(tm_path.read_text())
        mj = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").exists() else {}
        cfg = load_yaml(run / "config.yaml") if (run / "config.yaml").exists() else {"model": {}, "data": {}, "train": {}}
        t = tm.get("test", {})
        cs = tm.get("clean_subset", {})
        dev_w = _get(mj, "best", "dev_weighted_f1", default=_get(mj, "dev", "weighted_f1"))
        dev_m = _get(mj, "best", "dev_macro_f1", default=_get(mj, "dev", "macro_f1"))
        tr_w = _get(mj, "best", "train_weighted_f1", default=_get(mj, "train", "weighted_f1"))
        rows.append({
            "name": tm.get("name", run.parent.name), "seed": tm.get("seed", run.name), "model_type": tm.get("model_type"),
            "kind": tm.get("kind"), "modality": tm.get("modality", model_modality(cfg) if cfg.get("model", {}).get("type") else "none"),
            "text_source": tm.get("text_source"),
            "class_weight_scheme": mj.get("class_weight_scheme", _get(cfg, "train", "class_weight", default=_get(cfg, "model", "class_weight"))),
            "loss": mj.get("loss", _get(cfg, "train", "loss")),
            "weighted_f1": t.get("weighted_f1"), "weighted_f1_lo": _get(t, "bootstrap", "weighted_f1", "lo"),
            "weighted_f1_hi": _get(t, "bootstrap", "weighted_f1", "hi"),
            "macro_f1": t.get("macro_f1"), "macro_f1_lo": _get(t, "bootstrap", "macro_f1", "lo"), "macro_f1_hi": _get(t, "bootstrap", "macro_f1", "hi"),
            "accuracy": t.get("accuracy"), "fear_recall": t.get("fear_recall"), "disgust_recall": t.get("disgust_recall"),
            "min_class_recall": t.get("min_class_recall"), "n_test": t.get("n"),
            "clean_weighted_f1": cs.get("weighted_f1"), "clean_macro_f1": cs.get("macro_f1"), "n_clean": cs.get("n"),
            "dev_weighted_f1": dev_w, "dev_macro_f1": dev_m, "train_weighted_f1": tr_w,
            "n_test_evaluations": tm.get("n_test_evaluations"), "timestamp": tm.get("timestamp"), "git_commit": tm.get("git_commit"),
            "run_dir": str(run), "test_metrics": tm, "metrics": mj, "config": cfg,
        })
    df = pd.DataFrame(rows)
    if len(df):
        order = {n: i for i, n in enumerate(LADDER)}
        df["ladder"] = df["name"].map(lambda n: order.get(n, len(order)))
        df = df.sort_values(["ladder", "name", "seed"]).reset_index(drop=True)
    return df


def base_name(name: str) -> str:
    """``text_roberta_cwnone`` -> ``text_roberta``; ``fusion_gated_gold`` -> ``fusion_gated``."""
    for suf in ABLATION_SUFFIXES + CONDITION_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def load_dev_results(runs_root: str | Path) -> pd.DataFrame:
    """One row per ``<runs-root>/<name>/<seed>/metrics.json`` (every trained run, whether or
    not it was test-evaluated): dev / train scores of the selected checkpoint, class-weight
    scheme, loss and the config's ``model.type``. Selection and ablation tables read this."""
    root = resolve_path(runs_root)
    rows = []
    for mj_path in sorted(root.glob("*/*/metrics.json")):
        run = mj_path.parent
        mj = json.loads(mj_path.read_text())
        cfg = load_yaml(run / "config.yaml") if (run / "config.yaml").exists() else {"model": {}, "data": {}, "train": {}}
        model_type = _get(cfg, "model", "type", default=mj.get("type"))
        rows.append({
            "name": mj.get("name", run.parent.name), "seed": mj.get("seed", run.name), "model_type": model_type,
            "modality": model_modality(cfg) if model_type and cfg.get("model", {}).get("type") else "none",
            "class_weight_scheme": mj.get("class_weight_scheme", _get(cfg, "train", "class_weight", default=_get(cfg, "model", "class_weight"))),
            "loss": mj.get("loss", _get(cfg, "train", "loss")),
            "dev_weighted_f1": _get(mj, "best", "dev_weighted_f1", default=_get(mj, "dev", "weighted_f1")),
            "dev_macro_f1": _get(mj, "best", "dev_macro_f1", default=_get(mj, "dev", "macro_f1")),
            "train_weighted_f1": _get(mj, "best", "train_weighted_f1", default=_get(mj, "train", "weighted_f1")),
            "has_test": (run / "test_metrics.json").exists(), "run_dir": str(run), "config": cfg,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["base"] = df["name"].map(base_name)
        order = {n: i for i, n in enumerate(LADDER)}
        df["ladder"] = df["base"].map(lambda n: order.get(n, len(order)))
        df = df.sort_values(["ladder", "name", "seed"]).reset_index(drop=True)
    return df


def _fmt_ci(mean, lo, hi) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "-"
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return f"{mean:.3f}"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +- std over seeds; the CI column is the mean of the per-seed bootstrap bounds."""
    rows = []
    for name, g in df.groupby("name", sort=False):
        def ms(col):
            v = g[col].astype(float)
            return f"{v.mean():.3f} +- {v.std(ddof=0):.3f}" if len(v) > 1 else f"{v.mean():.3f}"
        rows.append({
            "model": name, "seeds": len(g), "n_test": int(g["n_test"].iloc[0]) if g["n_test"].notna().any() else None,
            "weighted F1": ms("weighted_f1"), "wF1 95% CI": _fmt_ci(g["weighted_f1"].mean(), g["weighted_f1_lo"].mean(), g["weighted_f1_hi"].mean()),
            "macro F1": ms("macro_f1"), "mF1 95% CI": _fmt_ci(g["macro_f1"].mean(), g["macro_f1_lo"].mean(), g["macro_f1_hi"].mean()),
            "accuracy": ms("accuracy"), "fear recall": ms("fear_recall"), "disgust recall": ms("disgust_recall"),
            "clean wF1": ms("clean_weighted_f1") if g["clean_weighted_f1"].notna().any() else "-",
            "dev wF1": ms("dev_weighted_f1") if g["dev_weighted_f1"].notna().any() else "-",
        })
    return pd.DataFrame(rows)


def _paired(run_a: Path, run_b: Path, fname: str = "test_predictions.csv") -> tuple[np.ndarray, np.ndarray] | None:
    fa, fb = run_a / fname, run_b / fname
    if not (fa.exists() and fb.exists()):
        return None
    a = pd.read_csv(fa)
    b = pd.read_csv(fb)
    m = a.merge(b, on="utt_id", suffixes=("_a", "_b"))
    if len(m) == 0:
        return None
    assert (m["label_a"] == m["label_b"]).all()
    return (m["pred_a"] == m["label_a"]).to_numpy(), (m["pred_b"] == m["label_b"]).to_numpy()


def mcnemar_table(df: pd.DataFrame, runs_root: Path, fusion: str | list[str], unimodal: list[str]) -> tuple[pd.DataFrame, str | None]:
    """Every fusion family in ``fusion`` (one name or a list) vs each unimodal run per seed. The
    "best" unimodal is the one with the highest mean dev weighted F1 (selection on dev, never
    on test)."""
    fusions = [fusion] if isinstance(fusion, str) else list(fusion)
    present = [u for u in unimodal if (df["name"] == u).any()]
    best = None
    if present:
        dev = {u: df.loc[df["name"] == u, "dev_weighted_f1"].astype(float).mean() for u in present}
        best = max(dev, key=lambda u: (np.nan_to_num(dev[u], nan=-1.0)))
    rows = []
    for f in fusions:
        for u in present:
            for seed in sorted(df.loc[df["name"] == f, "seed"].unique(), key=str):
                pair = _paired(runs_root / f / str(seed), runs_root / u / str(seed))
                if pair is None:
                    continue
                r = mcnemar(*pair)
                rows.append({"fusion": f, "unimodal": u, "best_unimodal": u == best, "seed": seed, "n": r["n"],
                             "acc_fusion": round(r["acc_a"], 4), "acc_unimodal": round(r["acc_b"], 4), "n01 (fusion only right)": r["n01"],
                             "n10 (unimodal only right)": r["n10"], "method": r["method"], "p_value": r["p_value"],
                             "p_exact": r["p_exact"], "p_chi2_cc": r["p_chi2_cc"]})
    return pd.DataFrame(rows), best


def minority_recall_check(df: pd.DataFrame, fusion: str | None) -> tuple[str, str]:
    """M6: every class with test support has recall > 0 for the chosen fusion model on every
    seed; fear and disgust recall listed per seed."""
    g = df[df["name"] == fusion] if fusion else df.iloc[0:0]
    if g.empty:
        return "SKIP", f"no test-evaluated run for the chosen fusion model `{fusion}`"
    parts, zero = [], []
    for _, r in g.iterrows():
        t = r["test_metrics"].get("test", {})
        zc = t.get("classes_with_zero_recall")
        if zc is None:  # older test_metrics.json: derive from per_class
            zc = [e for e, v in t.get("per_class", {}).items() if v.get("support", 0) > 0 and v.get("recall", 0) == 0]
        parts.append(f"seed {r['seed']}: fear {t.get('fear_recall', float('nan')):.3f}, disgust {t.get('disgust_recall', float('nan')):.3f}"
                     + (f", zero recall: {zc}" if zc else ""))
        zero.extend(zc)
    status = "PASS" if not zero else "FAIL"
    return status, f"`{fusion}` test recall per class > 0 on {len(g)} seed(s): {'yes' if not zero else 'no'}; " + "; ".join(parts)


def transcript_table(df: pd.DataFrame) -> pd.DataFrame:
    """2x2 (train gold/ASR x test gold/ASR) per model family (``<name>`` and ``<name>_gold``)."""
    rows = []
    for _, r in df.iterrows():
        tc = _get(r["test_metrics"], "transcript_conditions")
        if not isinstance(tc, dict) or "trained_on" not in tc:
            continue
        family = r["name"][:-5] if r["name"].endswith("_gold") else r["name"]
        for test_cond in ("gold", "asr"):
            if test_cond in tc:
                rows.append({"family": family, "run": r["name"], "seed": r["seed"], "train": tc["trained_on"], "test": test_cond,
                             "weighted_f1": tc[test_cond]["weighted_f1"], "macro_f1": tc[test_cond]["macro_f1"]})
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    out = long.groupby(["family", "train", "test"]).agg(weighted_f1=("weighted_f1", "mean"), macro_f1=("macro_f1", "mean"),
                                                        seeds=("seed", "nunique"), run=("run", "first")).reset_index()
    return out


def present_any(df: pd.DataFrame, names: list[str]) -> bool:
    return bool(len(df)) and any((df["name"] == n).any() for n in names)


def transcript_gate(tt: pd.DataFrame, text_family: str, fusion_family: str | None) -> tuple[str, str, str]:
    """V5 verdict: PASS only when the text family and the chosen fusion family both have all
    four train x test cells; otherwise PARTIAL naming the missing cells (SKIP with no data)."""
    if tt is None or len(tt) == 0:
        return ("V5", "SKIP", "no transcript_conditions block in any test_metrics.json")
    cells = {fam: set(zip(g["train"], g["test"])) for fam, g in tt.groupby("family")}
    need = {(tr, te) for tr in ("gold", "asr") for te in ("gold", "asr")}
    required = [text_family] + ([fusion_family] if fusion_family else [])
    missing = {fam: sorted(f"train {tr}/test {te}" for tr, te in (need - cells.get(fam, set()))) for fam in required}
    complete = [fam for fam in required if not missing[fam]]
    status = "PASS" if len(complete) == len(required) and fusion_family else "PARTIAL"
    ev = (f"required families {required}; complete 2x2 for {complete or 'none'}; "
          + "; ".join(f"{fam} missing {m}" for fam, m in missing.items() if m) + ("" if fusion_family else "; no fusion model chosen")
          + "; cells present: " + ", ".join(f"{fam}:{len(c)}/4" for fam, c in cells.items()))
    return ("V5", status, ev.replace("; ; ", "; "))


def noise_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        nz = _get(r["test_metrics"], "noise")
        if not isinstance(nz, dict):
            continue
        clean = r["weighted_f1"]
        if "skipped" in nz:
            rows.append({"model": r["name"], "seed": r["seed"], "modality": r["modality"], "clean wF1": clean, "note": nz["skipped"]})
            continue
        row = {"model": r["name"], "seed": r["seed"], "modality": r["modality"], "clean wF1": clean, "note": ""}
        for snr, m in sorted(nz.items(), key=lambda kv: -float(kv[0])):
            row[f"{snr} dB wF1"] = m["weighted_f1"]
            row[f"{snr} dB delta"] = m["weighted_f1"] - clean
            row[f"{snr} dB mF1"] = m["macro_f1"]
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        num = [c for c in out.columns if c not in ("model", "seed", "modality", "note")]
        out = out.groupby(["model", "modality"], sort=False).agg({**{c: "mean" for c in num}, "seed": "nunique", "note": "first"}).reset_index()
        out = out.rename(columns={"seed": "seeds"})
    return out


def per_class_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("name", sort=False):
        row = {"model": name}
        for e in EMOTIONS:
            row[e] = np.mean([_get(tm, "test", "per_class", e, "f1", default=np.nan) for tm in g["test_metrics"]])
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_figures(df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, g in df.groupby("name", sort=False):
        mats = [np.asarray(_get(tm, "test", "confusion_row_norm")) for tm in g["test_metrics"] if _get(tm, "test", "confusion_row_norm") is not None]
        if not mats:
            continue
        cm = np.mean(mats, axis=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(NUM_CLASSES), EMOTIONS, rotation=45, ha="right")
        ax.set_yticks(range(NUM_CLASSES), EMOTIONS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}: row-normalised confusion (mean of {len(mats)} seed(s))")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        p = fig_dir / f"confusion_{name}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def class_weight_table(dev: pd.DataFrame, test: pd.DataFrame | None = None) -> pd.DataFrame:
    """M7 from dev scores: every family that has at least one ablation arm (``<name>_cwnone``,
    ``_cwinv``, ``_cwfocal``, ``_undersample``) with the main run first, mean dev weighted and
    macro F1 over seeds, test columns for the arms that happen to be test-evaluated, and the
    PLAN Section 5 rule: an arm whose dev macro F1 is below the ``_cwnone`` (unweighted CE)
    sibling's is ``REJECTED`` even if its weighted F1 is higher."""
    if dev is None or dev.empty:
        return pd.DataFrame()
    d = dev[~dev["name"].str.endswith(CONDITION_SUFFIXES)]
    is_arm = d["name"].str.endswith(ABLATION_SUFFIXES)
    bases = set(d.loc[is_arm, "base"])
    sub = d[d["base"].isin(bases)]
    if sub.empty:
        return pd.DataFrame()
    agg = sub.groupby(["base", "name"], sort=False).agg(
        class_weight_scheme=("class_weight_scheme", "first"), loss=("loss", "first"), seeds=("seed", "nunique"),
        dev_weighted_f1=("dev_weighted_f1", "mean"), dev_macro_f1=("dev_macro_f1", "mean")).reset_index()
    if test is not None and len(test):
        t = test.groupby("name").agg(test_weighted_f1=("weighted_f1", "mean"), test_macro_f1=("macro_f1", "mean"),
                                     fear_recall=("fear_recall", "mean"), disgust_recall=("disgust_recall", "mean")).reset_index()
        agg = agg.merge(t, on="name", how="left")
    else:
        for c in ("test_weighted_f1", "test_macro_f1", "fear_recall", "disgust_recall"):
            agg[c] = np.nan
    none_mf1 = {b: g.loc[g["name"] == f"{b}_cwnone", "dev_macro_f1"].iloc[0] for b, g in agg.groupby("base")
                if (g["name"] == f"{b}_cwnone").any()}
    verdict = []
    for _, r in agg.iterrows():
        ref = none_mf1.get(r["base"])
        if r["name"] == f"{r['base']}_cwnone":
            verdict.append("baseline (unweighted CE)")
        elif ref is None or pd.isna(ref) or pd.isna(r["dev_macro_f1"]):
            verdict.append("n/a (no _cwnone sibling)")
        else:
            verdict.append("REJECTED" if r["dev_macro_f1"] < ref else "ok")
    agg["M7 rule (dev mF1 >= _cwnone)"] = verdict
    order = {s: i + 1 for i, s in enumerate(ABLATION_SUFFIXES)}
    agg["_k"] = agg["name"].map(lambda n: next((order[s] for s in ABLATION_SUFFIXES if n.endswith(s)), 0))
    return agg.sort_values(["base", "_k"]).drop(columns=["_k", "base"]).reset_index(drop=True)


def rejected_families(cw: pd.DataFrame) -> set[str]:
    if cw is None or cw.empty:
        return set()
    col = "M7 rule (dev mF1 >= _cwnone)"
    return set(cw.loc[cw[col] == "REJECTED", "name"])


def select_fusion(dev: pd.DataFrame, test: pd.DataFrame, fusion_arg: str, rejected: set[str] | None = None) -> tuple[str | None, pd.DataFrame, str]:
    """Final fusion model. ``fusion_arg == "auto"``: among the test-evaluated runs whose config
    ``model.type`` starts with ``fusion`` (ablation / ``_gold`` arms excluded) the family with
    the best mean dev weighted F1 (ties: dev macro F1, then a trained fusion model before the
    ``fusion_late`` baseline, then ladder order); families the M7 rule rejects are skipped when
    a non-rejected one exists. Returns (name, candidate table, reason)."""
    rejected = rejected or set()
    tested = set(test["name"]) if test is not None and len(test) else set()
    cand = dev[dev["model_type"].astype(str).str.startswith("fusion") & dev["name"].isin(tested)
               & (dev["name"] == dev["base"])] if len(dev) else pd.DataFrame()
    tab = pd.DataFrame()
    if len(cand):
        tab = cand.groupby("name", sort=False).agg(model_type=("model_type", "first"), seeds=("seed", "nunique"),
                                                   dev_weighted_f1=("dev_weighted_f1", "mean"), dev_macro_f1=("dev_macro_f1", "mean"),
                                                   ladder=("ladder", "first")).reset_index()
        tab["M7"] = np.where(tab["name"].isin(rejected), "REJECTED", "ok")
        # ties: dev macro F1, then a trained fusion model (7 / 8) before the late-fusion baseline (6), then ladder order
        tab["_late"] = (tab["model_type"] == "fusion_late").astype(int)
        tab = tab.sort_values(["dev_weighted_f1", "dev_macro_f1", "_late", "ladder"], ascending=[False, False, True, True]).drop(columns="_late").reset_index(drop=True)
    if fusion_arg != "auto":
        present = fusion_arg in tested
        return fusion_arg, tab, (f"`--fusion {fusion_arg}` given on the command line (override; selection on dev skipped)"
                                 + ("" if present else f"; WARNING: no test_metrics.json for `{fusion_arg}` under the runs root"))
    if tab.empty:
        return None, tab, "no test-evaluated fusion run (config model.type fusion*) under the runs root"
    pool = tab[tab["M7"] != "REJECTED"]
    if pool.empty:
        pool = tab
        note = "; every candidate is REJECTED by the M7 rule, so the best dev weighted F1 wins anyway"
    else:
        note = "" if len(pool) == len(tab) else f"; skipped (M7 REJECTED): {', '.join(tab.loc[tab['M7'] == 'REJECTED', 'name'])}"
    best = pool.iloc[0]
    others = ", ".join(f"{r['name']} {r['dev_weighted_f1']:.4f}" for _, r in tab.iterrows() if r["name"] != best["name"]) or "no other candidate"
    reason = (f"auto-selected on mean dev weighted F1 over {int(best['seeds'])} seed(s): `{best['name']}` {best['dev_weighted_f1']:.4f} "
              f"(dev macro F1 {best['dev_macro_f1']:.4f}) vs {others}{note}")
    return str(best["name"]), tab.drop(columns="ladder"), reason


def gain_gap_table(df: pd.DataFrame, fusion: str, unimodal: list[str]) -> tuple[pd.DataFrame, dict]:
    """M8 (dev gain of the fusion model over the best unimodal) and M9 (train - dev weighted F1
    of the chosen checkpoint) from ``metrics.json``."""
    names = [fusion] + [u for u in unimodal if u != fusion]
    rows = []
    for name in names:
        g = df[df["name"] == name]
        if g.empty:
            continue
        dev = g["dev_weighted_f1"].astype(float)
        tr = g["train_weighted_f1"].astype(float)
        rows.append({"model": name, "seeds": len(g), "dev weighted F1": dev.mean(), "train weighted F1": tr.mean(),
                     "train - dev": (tr - dev).mean(), "M9 (< 0.15)": "PASS" if (tr - dev).mean() < 0.15 else "FAIL"})
    tab = pd.DataFrame(rows)
    info: dict[str, Any] = {}
    if len(tab) and tab["model"].iloc[0] == fusion:
        uni = tab[tab["model"] != fusion]
        if len(uni):
            best = uni.sort_values("dev weighted F1", ascending=False).iloc[0]
            gain = tab["dev weighted F1"].iloc[0] - best["dev weighted F1"]
            info = {"fusion": fusion, "best_unimodal": best["model"], "dev_gain": float(gain), "M8": "PASS" if gain >= 0 else "FAIL"}
    return tab, info


def literature_table(path: Path) -> tuple[pd.DataFrame, int]:
    if not path.exists():
        return pd.DataFrame(), 0
    lit = yaml.safe_load(path.read_text()) or {}
    entries = lit.get("results", [])
    df = pd.DataFrame(entries)
    filled = int(sum(1 for e in entries if e.get("weighted_f1") is not None and e.get("citation"))) if entries else 0
    return df, filled


def row_wer(gold: str, asr: str) -> float:
    from src.asr.quality import wer_normalize, word_errors

    e, n = word_errors(wer_normalize(gold or ""), wer_normalize(asr or ""))
    return float(e / n) if n else float("nan")


def error_analysis(df: pd.DataFrame, runs_root: Path, fusion: str, unimodal: list[str], manifest_dir: str | None,
                   out_csv: Path, n: int = 30, seed: int = 0) -> pd.DataFrame | None:
    """Sample ``n`` misclassified test utterances of the fusion run stratified by true class
    (V8) with gold/ASR text, per-row WER, modality disagreement and empty manual columns."""
    g = df[df["name"] == fusion]
    if g.empty:
        return None
    r = g.iloc[0]
    run = Path(r["run_dir"])
    preds = pd.read_csv(run / "test_predictions.csv")
    cfg = r["config"]
    tag = cfg["data"].get("asr_tag")
    mdir = manifest_dir if manifest_dir is not None else cfg["data"].get("manifest_dir")
    if tag and not asr_path(tag, "test", mdir).exists():
        tag = None
    man = load_manifest("test", require_audio=True, asr_tag=tag, text_source="gold", manifest_dir_override=mdir)
    if "asr_text" not in man:
        man["asr_text"] = ""
    m = preds.merge(man[["utt_id", "gold_text", "asr_text", "duration_s", "emotion", "speaker"]], on="utt_id", how="left")
    wrong = m[m["pred"] != m["label"]].copy()
    if wrong.empty:
        return None
    rng = np.random.default_rng(seed)
    # stratified: proportional to the class share among errors, at least one per class present
    counts = wrong["label"].value_counts()
    alloc = {c: 1 for c in counts.index}
    remaining = max(0, min(n, len(wrong)) - len(alloc))
    shares = (counts / counts.sum() * remaining).round().astype(int)
    for c in counts.index:
        alloc[c] += int(min(shares[c], counts[c] - 1))
    picks = []
    for c, k in alloc.items():
        pool = wrong[wrong["label"] == c]
        picks.append(pool.sample(n=min(k, len(pool)), random_state=int(rng.integers(0, 2**31 - 1))))
    sample = pd.concat(picks)
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    elif len(sample) < min(n, len(wrong)):
        rest = wrong.drop(sample.index)
        sample = pd.concat([sample, rest.sample(n=min(n, len(wrong)) - len(sample), random_state=seed)])
    sample = sample.sort_values(["label", "utt_id"]).reset_index(drop=True)
    text_run = audio_run = None
    for u in unimodal:
        rr = df[(df["name"] == u) & (df["seed"].astype(str) == str(r["seed"]))]
        if rr.empty:
            continue
        if rr.iloc[0]["modality"] == "text" and text_run is None:
            text_run = Path(rr.iloc[0]["run_dir"])
        elif rr.iloc[0]["modality"] == "audio" and audio_run is None:
            audio_run = Path(rr.iloc[0]["run_dir"])
    side = {}
    for key, rd in (("text", text_run), ("audio", audio_run)):
        if rd is not None and (rd / "test_predictions.csv").exists():
            side[key] = pd.read_csv(rd / "test_predictions.csv").set_index("utt_id")["pred"]
    out = pd.DataFrame({
        "utt_id": sample["utt_id"], "gold_text": sample["gold_text"], "asr_text": sample["asr_text"],
        "true": sample["label"].map(lambda i: EMOTIONS[int(i)]), "pred": sample["pred"].map(lambda i: EMOTIONS[int(i)]),
        "p_true": [row[f"p_{int(row['label'])}"] for _, row in sample.iterrows()],
        "p_pred": [row[f"p_{int(row['pred'])}"] for _, row in sample.iterrows()],
        "duration_s": sample["duration_s"], "speaker": sample["speaker"],
        "asr_wer": [row_wer(gt, at) for gt, at in zip(sample["gold_text"], sample["asr_text"])],
    })
    if "text" in side and "audio" in side:
        out["text_pred"] = sample["utt_id"].map(side["text"]).map(lambda i: EMOTIONS[int(i)] if pd.notna(i) else "")
        out["audio_pred"] = sample["utt_id"].map(side["audio"]).map(lambda i: EMOTIONS[int(i)] if pd.notna(i) else "")
        both = (out["text_pred"] != "") & (out["audio_pred"] != "")
        out["modality_disagreement"] = np.where(both, out["text_pred"] != out["audio_pred"], np.nan)
    else:
        out["text_pred"] = out["audio_pred"] = ""
        out["modality_disagreement"] = np.nan
    out["asr_error"] = ""
    out["plausibly_ambiguous"] = ""
    out["note"] = ""
    out.to_csv(out_csv, index=False)
    return out


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df is None or len(df) == 0:
        return "_none_\n"
    return df.to_markdown(index=False, floatfmt=floatfmt) + "\n"


def summarize(args: argparse.Namespace) -> Path:
    t0 = time.time()
    runs_root = resolve_path(args.runs_root)
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = out.parent / "figures"
    unimodal = [u for u in args.unimodal.split(",") if u]
    df = load_run_results(runs_root)
    if df.empty:
        raise SystemExit(f"no test_metrics.json under {runs_root}")
    dev_df = load_dev_results(runs_root)
    cw = class_weight_table(dev_df, df)
    fusion, fusion_tab, fusion_reason = select_fusion(dev_df, df, args.fusion, rejected_families(cw))
    fusion_families = sorted(set(df.loc[df["model_type"].astype(str).str.startswith("fusion"), "name"]) | ({fusion} if fusion else set()),
                             key=lambda n: ({n: i for i, n in enumerate(LADDER)}.get(base_name(n), len(LADDER)), n))
    lines = [f"# Results ({runs_root.relative_to(ROOT) if runs_root.is_relative_to(ROOT) else runs_root})\n",
             f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `src/evaluate.py summary`, "
             f"git {git_commit()[:8]}. {len(df)} evaluated run(s), {df['name'].nunique()} model(s); {len(dev_df)} trained run(s) with "
             f"metrics.json. Test rows per run: {sorted(set(int(x) for x in df['n_test'].dropna()))}.\n",
             f"**Final fusion model: `{fusion}`** ({fusion_reason}).\n"]
    if len(fusion_tab):
        lines += ["Fusion candidates (dev scores from metrics.json; selection never looks at test):\n", _md(fusion_tab, ".4f")]
    gate: list[tuple[str, str, str]] = []
    print(f"final fusion model: {fusion} ({fusion_reason})")

    tab = results_table(df)
    lines += ["## Results ladder (test split, mean +- std over seeds; CI = mean of per-seed 95 % bootstrap bounds, 1000 resamples)\n", _md(tab)]
    n_eval_bad = df[(df["n_test_evaluations"].fillna(0) != 1)]
    gate.append(("V1", "PASS" if n_eval_bad.empty else "FAIL",
                 f"{len(df)} runs with timestamped test_metrics.json; n_test_evaluations == 1 for {len(df) - len(n_eval_bad)}/{len(df)}"
                 + ("" if n_eval_bad.empty else "; re-evaluated: " + ", ".join(f"{r['name']}/{r['seed']} x{int(r['n_test_evaluations'])}" for _, r in n_eval_bad.iterrows()))))
    missing_ci = df[df["weighted_f1_lo"].isna() | df["macro_f1_lo"].isna()]
    gate.append(("V2", "PASS" if missing_ci.empty else "FAIL", f"bootstrap CIs for weighted and macro F1 on {len(df) - len(missing_ci)}/{len(df)} runs"))

    mc, best_uni = mcnemar_table(df, runs_root, fusion_families, unimodal)
    lines += [f"## McNemar: every fusion family {fusion_families} vs unimodal runs (paired test predictions, per seed; "
              f"best unimodal by dev weighted F1 = `{best_uni}`; chosen fusion = `{fusion}`)\n",
              _md(mc, ".4g") if len(mc) else "_no paired predictions found_\n"]
    mb = mc[(mc["unimodal"] == best_uni) & (mc["fusion"] == fusion)] if len(mc) and best_uni is not None else pd.DataFrame()
    if len(mb):
        others = mc[(mc["unimodal"] == best_uni) & (mc["fusion"] != fusion)]
        gate.append(("V3", "PASS", f"`{fusion}` vs `{best_uni}`: " + ", ".join(f"seed {r['seed']}: p={r['p_value']:.3g} ({r['method']})" for _, r in mb.iterrows())
                     + (f"; other fusion families: " + ", ".join(f"{r['fusion']} seed {r['seed']} p={r['p_value']:.3g}" for _, r in others.iterrows()) if len(others) else "")))
    else:
        gate.append(("V3", "SKIP" if fusion is None or not present_any(df, unimodal) else "FAIL",
                     f"no paired test predictions for chosen fusion `{fusion}` vs unimodal runs {unimodal}"))
    m6_status, m6_ev = minority_recall_check(df, fusion)
    gate.append(("M6", m6_status, m6_ev))

    pc = per_class_table(df)
    figs = confusion_figures(df, fig_dir)
    lines += ["## Per-class F1 (test, mean over seeds)\n", _md(pc),
              "## Confusion matrices (row-normalised)\n"] + [f"- `{p.relative_to(out.parent)}`" for p in figs] + [""]
    gate.append(("V4", "PASS" if len(pc) and figs else "FAIL", f"per-class table for {len(pc)} models; {len(figs)} confusion figures in {fig_dir.relative_to(ROOT) if fig_dir.is_relative_to(ROOT) else fig_dir}"))

    tt = transcript_table(df)
    lines += ["## Transcript conditions (V5): train condition x test condition, weighted F1 (mean over seeds)\n"]
    if len(tt):
        for fam, g in tt.groupby("family"):
            piv = g.pivot(index="train", columns="test", values="weighted_f1").reindex(index=["gold", "asr"], columns=["gold", "asr"])
            piv.index = [f"train {i}" for i in piv.index]
            piv.columns = [f"test {c}" for c in piv.columns]
            lines += [f"**{fam}**\n", piv.to_markdown(floatfmt=".3f") + "\n"]
        lines += ["Long form:\n", _md(tt)]
    else:
        lines += ["_no run evaluated with --text-conditions_\n"]
    gate.append(transcript_gate(tt, text_family="text_roberta", fusion_family=base_name(fusion) if fusion else None))

    nt = noise_table(df)
    lines += ["## Noise robustness (V6): weighted F1 with additive Gaussian noise (mean over seeds)\n", _md(nt)]
    nz_ok = len(nt) and any(c.endswith("dB wF1") for c in nt.columns)
    gate.append(("V6", "PASS" if nz_ok else "SKIP", f"{int((nt['note'] == '').sum()) if len(nt) else 0} model(s) with per-SNR metrics" + ("" if nz_ok else "; run `evaluate run --noise-snr 20,10` on waveform models")))

    lines += ["## Class-weight ablation (M7; dev scores of every trained arm from metrics.json, test columns only where the arm was test-evaluated)\n",
              _md(cw, ".4f") if len(cw) else "_no `<name>_cwnone` / `_cwinv` / `_cwfocal` / `_undersample` run under the runs root_\n"]
    if len(cw):
        rej = sorted(rejected_families(cw))
        lines += [f"PLAN Section 5 rule: an arm whose dev macro F1 is below its `_cwnone` sibling's is REJECTED even if its weighted F1 is higher. "
                  f"Rejected: {rej or 'none'}. Chosen scheme per family = the non-rejected arm with the best dev weighted F1: "
                  + "; ".join(f"{b}: `{g.loc[g['M7 rule (dev mF1 >= _cwnone)'] != 'REJECTED'].sort_values('dev_weighted_f1', ascending=False, kind='stable')['name'].iloc[0]}`"
                              for b, g in cw.assign(base=cw["name"].map(base_name)).groupby("base") if (g["M7 rule (dev mF1 >= _cwnone)"] != "REJECTED").any()) + "\n"]
    gg, info = gain_gap_table(df, fusion, unimodal)
    lines += ["## Dev gain (M8) and train-dev gap (M9) of the chosen checkpoints (from metrics.json)\n", _md(gg)]
    if info:
        lines += [f"M8: `{info['fusion']}` - `{info['best_unimodal']}` on dev weighted F1 = {info['dev_gain']:+.4f} -> {info['M8']}\n"]

    lit_path = resolve_path(args.literature)
    lit, filled = literature_table(lit_path)
    lines += [f"## Literature comparison (V7, from `{lit_path.relative_to(ROOT) if lit_path.is_relative_to(ROOT) else lit_path}`)\n", _md(lit, ".1f")]
    gate.append(("V7", "PASS" if filled >= 3 else "FAIL", f"{filled} cited result(s) with weighted_f1 filled in; need 3 (placeholders carry a TODO)"))

    ea = error_analysis(df, runs_root, fusion, unimodal, args.manifest_dir, out.parent / "error_analysis.csv", n=args.n_errors, seed=args.seed)
    lines += ["## Error analysis (V8)\n"]
    if ea is not None:
        counts = {"rows": len(ea), "asr_wer > 0": int((ea["asr_wer"] > 0).sum()),
                  "modality_disagreement": int(ea["modality_disagreement"].fillna(False).astype(bool).sum()),
                  "manual asr_error filled": int((ea["asr_error"].astype(str) != "").sum()),
                  "manual plausibly_ambiguous filled": int((ea["plausibly_ambiguous"].astype(str) != "").sum())}
        lines += [f"`{(out.parent / 'error_analysis.csv').relative_to(ROOT) if out.parent.is_relative_to(ROOT) else out.parent / 'error_analysis.csv'}`: {counts}\n",
                  "Per true class:\n", _md(ea.groupby("true").agg(n=("utt_id", "count"), mean_wer=("asr_wer", "mean"),
                                                                      disagreement=("modality_disagreement", lambda s: int(s.fillna(False).astype(bool).sum()))).reset_index()),
                  _md(ea[["utt_id", "true", "pred", "p_true", "p_pred", "asr_wer", "modality_disagreement", "gold_text", "asr_text"]].head(30))]
        gate.append(("V8", "PASS" if len(ea) >= min(args.n_errors, 30) else "PARTIAL",
                     f"{len(ea)} sampled errors ({counts['asr_wer > 0']} with ASR errors, {counts['modality_disagreement']} with modality disagreement); "
                     f"manual columns asr_error / plausibly_ambiguous / note still to annotate"))
    else:
        lines += [f"_fusion run `{fusion}` has no test predictions or no errors_\n"]
        gate.append(("V8", "SKIP", f"no test_predictions.csv for `{fusion}`"))

    gate_md = ["# Gate 5 (evaluation)\n", f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from `{runs_root}`; "
               f"final fusion model `{fusion}` ({fusion_reason}).\n",
               "| ID | Check | Result | Evidence |", "|---|---|---|---|"]
    names = {"V1": "Single test pass", "V2": "Confidence intervals", "V3": "Significance", "M6": "Minority recall (Gate 4, final model on test)",
             "V4": "Per-class reporting", "V5": "ASR tax", "V6": "Robustness", "V7": "Literature comparison", "V8": "Error analysis"}
    order = {k: i for i, k in enumerate(names)}
    for gid, status, ev in sorted(gate, key=lambda g: order[g[0]]):
        gate_md.append(f"| {gid} | {names[gid]} | {status} | {ev} |")
    (out.parent / "gate5_eval.md").write_text("\n".join(gate_md) + "\n")
    lines += ["## Gate 5\n", "\n".join(gate_md[3:]) + "\n"]
    out.write_text("\n".join(lines))
    print("\n".join(gate_md[3:]))
    print(f"wrote {out} ({time.time() - t0:.1f}s)")
    return out


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate one run on the test split")
    r.add_argument("--run", required=True, help="runs/<name>/<seed>")
    r.add_argument("--manifest-dir")
    r.add_argument("--feature-root")
    r.add_argument("--device")
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--limit", type=int)
    r.add_argument("--noise-snr", help="comma-separated SNRs in dB, e.g. 20,10")
    r.add_argument("--text-conditions", action="store_true")
    r.add_argument("--config", default="configs/fusion_late.yaml", help="config for a fusion_late pseudo-run without config.yaml")
    r.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a value of that config, e.g. model.text_run=text_roberta_gold")
    r.add_argument("--force", action="store_true", help="allow a repeated test evaluation (recorded in n_test_evaluations)")
    r.add_argument("--seed", type=int, default=0, help="bootstrap / noise seed")
    s = sub.add_parser("summary", help="collect test_metrics.json files into a report")
    s.add_argument("--runs-root", default="runs")
    s.add_argument("--out", default="reports/results.md")
    s.add_argument("--fusion", default="auto", help="final fusion model: 'auto' = best mean dev weighted F1 among the test-evaluated "
                                                     "fusion families (M7-rejected ones skipped), or a run name to force it")
    s.add_argument("--unimodal", default="text_roberta,audio_wavlm_ft")
    s.add_argument("--literature", default="reports/literature.yaml")
    s.add_argument("--manifest-dir")
    s.add_argument("--n-errors", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "run":
        evaluate_run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
