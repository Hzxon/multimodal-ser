"""Train models 4-8 on the train split with early stopping on dev weighted F1 (PLAN.md 3.4).

Usage: python -m src.train --config configs/<name>.yaml [--seed 13] [--manifest-dir ...]
       [--feature-root ...] [--runs-root runs] [--limit N] [--max-steps N] [--max-epochs N]
       [--batch-size N] [--device cuda|mps|cpu] [--set key.path=value ...] [--name NAME]
       [--shuffle-labels] [--overfit-batch]

Writes runs/<name>/<seed>/{config.yaml, metrics.json, predictions.csv, best.pt, git_commit.txt}.
``git_commit.txt`` holds HEAD plus ``-dirty`` when the working tree has uncommitted changes.
Config templates: every ``init_from`` string may contain ``{seed}`` (replaced by ``--seed``) and
``{runs_root}`` (``--runs-root``, default ``runs``); a templated ``runs/...`` path is re-rooted
under ``--runs-root`` when that is not ``runs``. ``train.undersample: true`` is the PLAN 5
undersampling ablation (every class cut to the smallest class's count, seeded).
Gate M3: the data config may not mention "test" anywhere and ``data.splits`` must be exactly
{train, dev}. Gate M4: dev is evaluated exactly once per epoch. ``--shuffle-labels`` is the
Gate M2 control, ``--overfit-batch`` the Gate M1 check (see scripts/sanity_checks.py).
"""
from __future__ import annotations

import argparse
import copy
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score

from src.common import EMOTIONS, NUM_CLASSES, ROOT, device_name, git_commit, load_yaml, run_dir, set_seed, write_json
from src.data.dataset import class_weights
from src.data.torch_dataset import load_split_df, make_loader
from src.models import build_model, split_param_groups, tokenizer_name


# ----------------------------------------------------------------------------- config

def set_nested(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def substitute_init_from(value: Any, seed: int, runs_root: str | None = None) -> Any:
    """Resolve ``{seed}`` / ``{runs_root}`` in an ``init_from`` value (str, or nested dict/list of str).

    A string that contained ``{seed}`` or ``{runs_root}`` and starts with ``runs/`` is re-rooted under
    ``runs_root`` when that is given and is not ``runs``, so the config template
    ``runs/audio_wavlm_cached/{seed}`` follows ``--runs-root``. Strings without a template, and
    ``None``, are returned unchanged.
    """
    if isinstance(value, dict):
        return {k: substitute_init_from(v, seed, runs_root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [substitute_init_from(v, seed, runs_root) for v in value]
    if not isinstance(value, str):
        return value
    templated = "{seed}" in value or "{runs_root}" in value
    out = value.replace("{seed}", str(seed)).replace("{runs_root}", str(runs_root) if runs_root else "runs")
    root = str(runs_root).rstrip("/") if runs_root else "runs"
    if templated and root != "runs" and out.startswith("runs/"):
        out = f"{root}/{out[len('runs/'):]}"
    return out


def resolve_init_templates(cfg: Any, seed: int, runs_root: str | None = None) -> Any:
    """Apply ``substitute_init_from`` to every ``init_from`` key at any depth of ``cfg`` (in place)."""
    if isinstance(cfg, dict):
        for k in list(cfg):
            if k == "init_from":
                cfg[k] = substitute_init_from(cfg[k], seed, runs_root)
            else:
                resolve_init_templates(cfg[k], seed, runs_root)
    elif isinstance(cfg, list):
        for v in cfg:
            resolve_init_templates(v, seed, runs_root)
    return cfg


def _init_strings(v: Any) -> list[str]:
    """String values of an ``init_from`` entry (dict values only, never the dict keys)."""
    if isinstance(v, dict):
        return [x for sub in v.values() for x in _init_strings(sub)]
    if isinstance(v, (list, tuple)):
        return [x for sub in v for x in _init_strings(sub)]
    return [str(v)] if v else []


def init_from_paths(cfg: Any) -> list[str]:
    """All non-empty ``init_from`` strings in ``cfg`` (after resolution)."""
    found: list[str] = []
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == "init_from":
                found += _init_strings(v)
            else:
                found += init_from_paths(v)
    elif isinstance(cfg, list):
        for v in cfg:
            found += init_from_paths(v)
    return found


def check_init_from(cfg: dict) -> None:
    """A missing ``init_from`` run is an error: silently training from scratch would change the experiment."""
    for p in init_from_paths(cfg):
        pp = Path(p)
        pp = pp if pp.is_absolute() else ROOT / pp
        print(f"init_from -> {p}")
        if not (pp / "best.pt").exists():
            raise FileNotFoundError(
                f"init_from run not found: {pp / 'best.pt'} (resolved from {p!r}). Train that run first with the same "
                f"--seed/--runs-root, or pass --set model.init_from=null to start from the pretrained encoders only.")


def undersample_df(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """PLAN 5 ablation: keep at most ``n_min`` rows per class, ``n_min`` = the smallest non-empty class.

    Seeded, row order preserved, index reset. Classes absent from ``df`` (possible with ``--limit``)
    do not set ``n_min``. Returns the subsample and the per-emotion counts after subsampling.
    """
    labels = df["label"].to_numpy().astype(int)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    n_min = int(counts[counts > 0].min())
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(labels == c)
        if len(idx) > n_min:
            idx = rng.choice(idx, size=n_min, replace=False)
        keep.extend(int(i) for i in idx)
    out = df.iloc[sorted(keep)].reset_index(drop=True)
    after = np.bincount(out["label"].to_numpy().astype(int), minlength=NUM_CLASSES)
    return out, {e: int(n) for e, n in zip(EMOTIONS, after)}


def resolve_config(args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(load_yaml(args.config))
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})
    if args.name:
        cfg["name"] = args.name
    if args.manifest_dir:
        cfg["data"]["manifest_dir"] = args.manifest_dir
    if args.feature_root:
        cfg["data"]["feature_root"] = args.feature_root
    if args.limit is not None:
        cfg["data"]["limit"] = args.limit
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        set_nested(cfg, k, yaml.safe_load(v))
    t = cfg["train"]
    if args.max_epochs is not None:
        t["max_epochs"] = args.max_epochs
    if args.max_steps is not None:
        t["max_steps"] = args.max_steps
    if args.batch_size is not None:
        t["batch_size"] = args.batch_size
    t["seed"] = args.seed
    t["device"] = args.device or device_name()
    t["shuffle_labels"] = bool(args.shuffle_labels)
    t["overfit_batch"] = bool(args.overfit_batch)
    t["undersample"] = bool(t.get("undersample", False))
    resolve_init_templates(cfg, int(args.seed), args.runs_root)
    return cfg


def _strings(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)
    elif obj is not None:
        yield str(obj)


def assert_no_test_in_data_config(data_cfg: dict) -> None:
    """Gate M3: training never sees a path or split name containing "test"."""
    bad = [s for s in _strings(data_cfg) if "test" in s.lower()]
    assert not bad, f"data config mentions test: {bad}"
    splits = data_cfg.get("splits")
    assert isinstance(splits, dict) and set(splits) == {"train", "dev"}, f"data.splits must be exactly train+dev, got {splits}"


# ----------------------------------------------------------------------------- helpers

def to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        lp = logp.gather(1, target[:, None]).squeeze(1)
        p = lp.exp()
        w = self.weight[target] if self.weight is not None else torch.ones_like(lp)
        return (-(w * (1 - p) ** self.gamma * lp)).sum() / w.sum().clamp_min(1e-8)


def make_loss(scheme: str, train_labels: np.ndarray, loss_type: str, device: str) -> tuple[nn.Module, np.ndarray]:
    w = class_weights(train_labels, scheme)
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    if loss_type == "focal":
        return FocalLoss(wt, gamma=2.0), w
    assert loss_type == "ce", loss_type
    return nn.CrossEntropyLoss(weight=wt), w


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def metrics_from(y: np.ndarray, pred: np.ndarray) -> dict:
    labels = list(range(NUM_CLASSES))
    per_class = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    return {
        "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "per_class_f1": {e: float(v) for e, v in zip(EMOTIONS, per_class)},
    }


class DeterminismLog:
    """Collects the names of ops that fall back to non-deterministic kernels."""

    def __init__(self) -> None:
        self.messages: set[str] = set()
        self._prev = None

    def __enter__(self):
        self._prev = warnings.showwarning
        log = self

        def hook(message, category, filename, lineno, file=None, line=None):
            text = str(message)
            if "deterministic" in text.lower():
                log.messages.add(text.split("\n")[0][:200])
            else:
                log._prev(message, category, filename, lineno, file, line)

        warnings.showwarning = hook
        warnings.filterwarnings("always", message=".*deterministic.*")
        return self

    def __exit__(self, *exc):
        warnings.showwarning = self._prev
        return False


@torch.no_grad()
def predict(model: nn.Module, loader, device: str, use_amp: bool) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    ids: list[str] = []
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(batch)
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        ids.extend(batch["utt_id"])
    return ids, np.concatenate(probs), np.concatenate(labels)


def write_predictions(path: Path, ids: list[str], labels: np.ndarray, probs: np.ndarray) -> None:
    df = pd.DataFrame({"utt_id": ids, "label": labels, "pred": probs.argmax(1)})
    for c in range(NUM_CLASSES):
        df[f"p_{c}"] = probs[:, c]
    df.to_csv(path, index=False)


# ----------------------------------------------------------------------------- training

def train(cfg: dict, out: Path, runs_root: Path | None) -> dict:
    t0 = time.time()
    tcfg, dcfg = cfg["train"], cfg["data"]
    assert_no_test_in_data_config(dcfg)
    device = tcfg["device"]
    seed = int(tcfg["seed"])
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    use_amp = device == "cuda" and bool(tcfg.get("fp16", True))

    tok_name = tokenizer_name(cfg)
    tokenizer = None
    if tok_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tok_name)

    model = build_model(cfg).to(device)
    limit = dcfg.get("limit")
    feature_root = dcfg.get("feature_root")
    train_df = load_split_df(dcfg["splits"]["train"], cfg, limit=limit)
    undersample_info: dict = {"undersample": bool(tcfg.get("undersample", False))}
    if undersample_info["undersample"]:
        before = {e: int(n) for e, n in zip(EMOTIONS, np.bincount(train_df["label"].to_numpy().astype(int), minlength=NUM_CLASSES))}
        train_df, after = undersample_df(train_df, seed)
        undersample_info.update({"undersample_counts_before": before, "undersample_counts": after})
        print(f"[undersample] train rows {sum(before.values())} -> {len(train_df)}; per class before {before} after {after}")
    if tcfg["shuffle_labels"]:
        rng = np.random.default_rng(seed)
        train_df["label"] = rng.permutation(train_df["label"].to_numpy())
        print(f"[M2] train labels permuted with seed {seed}")
    bs = int(tcfg.get("batch_size", 16))
    train_loader, train_ds = make_loader(dcfg["splits"]["train"], cfg, train=True, seed=seed, batch_size=bs,
                                         feature_root=feature_root, tokenizer=tokenizer, df=train_df,
                                         num_workers=int(tcfg.get("num_workers", 0)))
    dev_loader, dev_ds = make_loader(dcfg["splits"]["dev"], cfg, train=False, seed=seed, batch_size=bs,
                                     feature_root=feature_root, tokenizer=tokenizer, limit=limit,
                                     num_workers=int(tcfg.get("num_workers", 0)))
    print(f"train {len(train_ds)} rows, dev {len(dev_ds)} rows, device {device}, transforms {[type(t).__name__ for t in train_ds.transforms]}")

    scheme = tcfg.get("class_weight", "sqrt_inv")
    loss_fn, cw = make_loss(scheme, train_ds.labels, tcfg.get("loss", "ce"), device)
    print("class weights (%s): %s" % (scheme, {e: round(float(w), 3) for e, w in zip(EMOTIONS, cw)}))

    enc_params, head_params = split_param_groups(model)
    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": float(tcfg.get("lr_encoder", 2e-5))})
    groups.append({"params": head_params, "lr": float(tcfg.get("lr_head", 1e-3))})
    opt = torch.optim.AdamW(groups, weight_decay=float(tcfg.get("weight_decay", 0.01)))
    n_enc = sum(p.numel() for p in enc_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"trainable params: encoder {n_enc:,} head {n_head:,}")

    accum = int(tcfg.get("grad_accum", 1))
    max_epochs = int(tcfg.get("max_epochs", 15))
    patience = int(tcfg.get("patience", 3))
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    max_steps = tcfg.get("max_steps")
    total_steps = steps_per_epoch * max_epochs if max_steps is None else min(int(max_steps), steps_per_epoch * max_epochs)
    warmup = int(round(float(tcfg.get("warmup_frac", 0.1)) * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total_steps, warmup))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best = {"dev_weighted_f1": -1.0, "epoch": -1}
    best_pred = None
    bad_epochs = 0
    step = 0
    step_times: list[float] = []
    dev_evals = 0
    det = DeterminismLog()
    with det:
        for epoch in range(1, max_epochs + 1):
            model.train()
            ep0 = time.time()
            losses, y_true, y_pred = [], [], []
            opt.zero_grad(set_to_none=True)
            for i, batch in enumerate(train_loader):
                s0 = time.time()
                batch = to_device(batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(batch)
                    loss = loss_fn(logits.float(), batch["label"]) / accum
                scaler.scale(loss).backward()
                losses.append(float(loss.detach()) * accum)
                y_true.append(batch["label"].cpu().numpy())
                y_pred.append(logits.detach().argmax(-1).cpu().numpy())
                if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 1.0)))
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    step += 1
                    step_times.append(time.time() - s0)
                    if step >= total_steps:
                        break
            tr = metrics_from(np.concatenate(y_true), np.concatenate(y_pred))
            # Gate M4: exactly one dev evaluation per epoch
            ids, probs, labels = predict(model, dev_loader, device, use_amp)
            dev_evals += 1
            dv = metrics_from(labels, probs.argmax(1))
            rec = {"epoch": epoch, "step": step, "train_loss": float(np.mean(losses)),
                   "train_weighted_f1": tr["weighted_f1"], "train_macro_f1": tr["macro_f1"], "train_accuracy": tr["accuracy"],
                   "dev_weighted_f1": dv["weighted_f1"], "dev_macro_f1": dv["macro_f1"], "dev_accuracy": dv["accuracy"],
                   "epoch_seconds": time.time() - ep0}
            history.append(rec)
            print(f"epoch {epoch} step {step} loss {rec['train_loss']:.4f} train wF1 {tr['weighted_f1']:.3f} "
                  f"dev wF1 {dv['weighted_f1']:.3f} dev mF1 {dv['macro_f1']:.3f} ({rec['epoch_seconds']:.1f}s, "
                  f"{np.mean(step_times[-len(train_loader):]):.2f}s/step)")
            if dv["weighted_f1"] > best["dev_weighted_f1"]:
                best = {"dev_weighted_f1": dv["weighted_f1"], "dev_macro_f1": dv["macro_f1"], "dev_accuracy": dv["accuracy"],
                        "dev_per_class_f1": dv["per_class_f1"], "train_weighted_f1": tr["weighted_f1"], "epoch": epoch, "step": step}
                best_pred = (ids, labels, probs)
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "dev_weighted_f1": dv["weighted_f1"],
                            "model_type": cfg["model"]["type"]}, out / "best.pt")
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"early stop at epoch {epoch} (patience {patience})")
                    break
            if step >= total_steps:
                print(f"max_steps {total_steps} reached")
                break

    assert best_pred is not None
    write_predictions(out / "predictions.csv", *best_pred)
    frac = train_ds.fraction_cropped
    print(f"cropped train clips: {train_ds.n_cropped}/{train_ds.n_served_audio} ({frac:.3f}); dev {dev_ds.n_cropped}/{dev_ds.n_served_audio}")
    extra = {}
    if hasattr(model, "layer_weights"):
        extra["layer_weights"] = model.layer_weights().cpu().tolist()
    elif hasattr(model, "audio") and hasattr(model.audio, "layer_weights"):
        extra["layer_weights"] = model.audio.layer_weights().cpu().tolist()
    return {
        "name": cfg["name"], "seed": seed, "device": device, "fp16": use_amp,
        "best": best, "history": history, "dev_evaluations": dev_evals, "epochs_run": len(history),
        "class_weight_scheme": scheme, "class_weights": {e: float(w) for e, w in zip(EMOTIONS, cw)},
        "loss": tcfg.get("loss", "ce"), "n_train": len(train_ds), "n_dev": len(dev_ds),
        "n_cropped_train": train_ds.n_cropped, "fraction_cropped_train": frac,
        "n_cropped_dev": dev_ds.n_cropped, "fraction_cropped_dev": dev_ds.fraction_cropped,
        "train_transforms": [type(t).__name__ for t in train_ds.transforms], "dev_transforms": [type(t).__name__ for t in dev_ds.transforms],
        "shuffled_labels": bool(tcfg["shuffle_labels"]), "overfit_batch": False, **undersample_info,
        "trainable_params": {"encoder": n_enc, "head": n_head},
        "total_steps": total_steps, "warmup_steps": warmup, "seconds_per_step": float(np.mean(step_times)) if step_times else None,
        "nondeterministic_ops": sorted(det.messages), "wall_seconds": time.time() - t0, **extra,
    }


def overfit_batch(cfg: dict, out: Path) -> dict:
    """Gate M1: one fixed batch, up to ``max_steps`` optimiser steps, report when train
    accuracy on that batch reaches 99 %. Data augmentation and SpecAugment are off."""
    t0 = time.time()
    tcfg, dcfg = cfg["train"], cfg["data"]
    assert_no_test_in_data_config(dcfg)
    device, seed = tcfg["device"], int(tcfg["seed"])
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    use_amp = device == "cuda" and bool(tcfg.get("fp16", True))
    tokenizer = None
    if tokenizer_name(cfg):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name(cfg))
    model = build_model(cfg).to(device)
    enc = getattr(model, "encoder", None) if cfg["model"]["type"] == "audio_wavlm" else getattr(getattr(model, "audio", None), "encoder", None)
    if enc is not None and hasattr(enc, "config"):
        enc.config.apply_spec_augment = False
    bs = int(tcfg.get("batch_size", 16))
    loader, ds = make_loader(dcfg["splits"]["train"], cfg, train=True, seed=seed, batch_size=bs,
                             feature_root=dcfg.get("feature_root"), tokenizer=tokenizer, limit=dcfg.get("limit"))
    ds.transforms, ds.text_transforms = [], []
    batch = to_device(next(iter(loader)), device)
    loss_fn, _ = make_loss(tcfg.get("class_weight", "sqrt_inv"), ds.labels, tcfg.get("loss", "ce"), device)
    enc_params, head_params = split_param_groups(model)
    groups = ([{"params": enc_params, "lr": float(tcfg.get("lr_encoder", 2e-5))}] if enc_params else []) + \
             [{"params": head_params, "lr": float(tcfg.get("lr_head", 1e-3))}]
    opt = torch.optim.AdamW(groups, weight_decay=float(tcfg.get("weight_decay", 0.01)))
    max_steps = int(tcfg.get("max_steps") or 200)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    curve, reached, step_times = [], None, []
    det = DeterminismLog()
    model.train()
    with det:
        for step in range(1, max_steps + 1):
            s0 = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(batch)
                loss = loss_fn(logits.float(), batch["label"])
            acc = float((logits.argmax(-1) == batch["label"]).float().mean())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 1.0)))
            scaler.step(opt)
            scaler.update()
            step_times.append(time.time() - s0)
            curve.append({"step": step, "loss": float(loss.detach()), "train_accuracy": acc})
            if step % 10 == 0 or step == 1:
                print(f"step {step} loss {float(loss.detach()):.4f} acc {acc:.3f} ({step_times[-1]:.2f}s/step)")
            if reached is None and acc >= 0.99:
                reached = step
                print(f"[M1] train accuracy >= 0.99 at step {step}")
                break
    final_acc = curve[-1]["train_accuracy"]
    res = {"name": cfg["name"], "seed": seed, "device": device, "overfit_batch": True, "shuffled_labels": False,
           "batch_size": int(batch["label"].shape[0]), "max_steps": max_steps, "steps_run": len(curve),
           "step_reached_99": reached, "final_train_accuracy": final_acc, "best_train_accuracy": max(c["train_accuracy"] for c in curve),
           "curve": curve, "seconds_per_step": float(np.mean(step_times)), "augmentation_disabled": True,
           "nondeterministic_ops": sorted(det.messages), "wall_seconds": time.time() - t0}
    print(f"[M1] {cfg['name']}: reached 99% at step {reached}, final acc {final_acc:.3f}, {res['seconds_per_step']:.2f}s/step")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--manifest-dir")
    ap.add_argument("--feature-root")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--device")
    ap.add_argument("--name", help="override cfg name (run directory)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value, e.g. model.audio.unfreeze_top_n=1 or train.undersample=true")
    ap.add_argument("--shuffle-labels", action="store_true", help="Gate M2 control: permute train labels")
    ap.add_argument("--overfit-batch", action="store_true", help="Gate M1 check: train on one fixed batch")
    args = ap.parse_args()
    cfg = resolve_config(args)
    runs_root = Path(args.runs_root) if args.runs_root else None
    if runs_root is not None and not runs_root.is_absolute():
        runs_root = ROOT / runs_root
    check_init_from(cfg)
    out = run_dir(cfg["name"], args.seed, root=runs_root)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "git_commit.txt").write_text(git_commit() + "\n")
    print(f"run dir {out}")
    res = overfit_batch(cfg, out) if args.overfit_batch else train(cfg, out, runs_root)
    res["git_commit"] = git_commit()
    write_json(out / "metrics.json", res)
    print(f"wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
