"""Gate 4 sanity checks M1 (overfit one batch) and M2 (shuffled-label control) per model
family, through ``src.train --overfit-batch`` and ``src.train --shuffle-labels``.

Usage: python scripts/sanity_checks.py --families text_roberta,audio_wavlm_cached,fusion_gated
       [--manifest-dir ...] [--feature-root ...] [--runs-root runs/sanity] [--steps 200]
       [--batch-size 16] [--m2-epochs 3] [--m2-max-steps N] [--seed 13] [--device ...]
       [--set key=value ...] [--out reports/gate4_sanity.md]

M1 passes when the fixed batch reaches >= 99 % train accuracy within --steps optimiser
steps (``--steps`` bounds M1 only). M2 trains on permuted labels for ``--m2-epochs`` full
epochs (``--m2-max-steps`` caps the steps when given; by default the control sees every
epoch) and passes when dev macro F1 is within +-3 points of chance: chance is computed two
ways, the macro F1 of always predicting the train majority class and of uniform random
predictions on dev (mean of 20 draws), and the pass band is
``[min(chance) - 0.03, max(chance) + 0.03]``. Above the band means leakage or a shortcut;
below it means the control run did not converge to chance either. Exit status 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import NUM_CLASSES, load_yaml  # noqa: E402
from src.data.dataset import load_manifest  # noqa: E402


def set_nested(cfg: dict, dotted: str, value) -> None:
    d = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def run_train(extra: list[str], args: argparse.Namespace, name: str) -> Path:
    cmd = [sys.executable, "-m", "src.train", "--name", name, "--seed", str(args.seed), "--runs-root", args.runs_root]
    if args.manifest_dir:
        cmd += ["--manifest-dir", args.manifest_dir]
    if args.feature_root:
        cmd += ["--feature-root", args.feature_root]
    if args.device:
        cmd += ["--device", args.device]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    for kv in args.set or []:
        cmd += ["--set", kv]
    cmd += extra
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    root = Path(args.runs_root)
    root = root if root.is_absolute() else ROOT / root
    return root / name / str(args.seed)


def chance_levels(cfg: dict, args: argparse.Namespace, seed: int) -> dict:
    d = cfg["data"]
    ts = d.get("text_source", "gold")
    kw = dict(asr_tag=d.get("asr_tag") if ts == "asr" else None, text_source=ts,
              manifest_dir_override=args.manifest_dir or d.get("manifest_dir"))
    tr = load_manifest(d["splits"]["train"], **kw)
    dv = load_manifest(d["splits"]["dev"], **kw)
    if args.limit:
        tr, dv = tr.iloc[: args.limit], dv.iloc[: args.limit]
    y = dv["label"].to_numpy()
    maj = int(np.bincount(tr["label"].to_numpy(), minlength=NUM_CLASSES).argmax())
    labels = list(range(NUM_CLASSES))
    f_maj = f1_score(y, np.full_like(y, maj), labels=labels, average="macro", zero_division=0)
    rng = np.random.default_rng(seed)
    f_rand = float(np.mean([f1_score(y, rng.integers(0, NUM_CLASSES, size=len(y)), labels=labels, average="macro", zero_division=0)
                            for _ in range(20)]))
    return {"majority_macro_f1": float(f_maj), "random_macro_f1": f_rand, "majority_class": maj}


def m2_band(chance: dict, tol: float = 0.03) -> tuple[float, float]:
    """PASS band of the shuffled-label control: +-``tol`` around the two chance levels."""
    levels = (chance["majority_macro_f1"], chance["random_macro_f1"])
    return min(levels) - tol, max(levels) + tol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True, help="comma-separated config names under configs/")
    ap.add_argument("--manifest-dir")
    ap.add_argument("--feature-root")
    ap.add_argument("--runs-root", default="runs/sanity")
    ap.add_argument("--steps", type=int, default=200, help="M1 only: max optimiser steps of the overfit-one-batch run")
    ap.add_argument("--m2-epochs", type=int, default=3, help="M2: epochs of the shuffled-label control run")
    ap.add_argument("--m2-max-steps", type=int, default=None, help="M2: optional step cap of the control run (default: none, bounded by --m2-epochs)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="forwarded to src.train")
    ap.add_argument("--out", default="reports/gate4_sanity.md")
    args = ap.parse_args()

    rows = []
    any_fail = False
    for fam in [f.strip() for f in args.families.split(",") if f.strip()]:
        cfg_path = f"configs/{fam}.yaml"
        cfg = load_yaml(cfg_path)
        for kv in args.set or []:  # same overrides as the training runs (e.g. data.text_source=gold)
            k, v = kv.split("=", 1)
            set_nested(cfg, k, yaml.safe_load(v))
        m1_dir = run_train(["--config", cfg_path, "--overfit-batch", "--max-steps", str(args.steps)], args, f"{fam}_m1")
        m1 = json.loads((m1_dir / "metrics.json").read_text())
        m1_pass = m1["step_reached_99"] is not None and m1["step_reached_99"] <= args.steps
        m2_extra = ["--config", cfg_path, "--shuffle-labels", "--max-epochs", str(args.m2_epochs)]
        if args.m2_max_steps:
            m2_extra += ["--max-steps", str(args.m2_max_steps)]
        m2_dir = run_train(m2_extra, args, f"{fam}_m2")
        m2 = json.loads((m2_dir / "metrics.json").read_text())
        ch = chance_levels(cfg, args, args.seed)
        band_lo, band_hi = m2_band(ch)
        dev_mf1 = m2["best"]["dev_macro_f1"]
        m2_pass = band_lo <= dev_mf1 <= band_hi
        any_fail |= not (m1_pass and m2_pass)
        rows.append({
            "family": fam, "m1_step": m1["step_reached_99"], "m1_final_acc": m1["final_train_accuracy"],
            "m1_steps_run": m1["steps_run"], "m1_s_per_step": m1["seconds_per_step"], "m1_pass": m1_pass,
            "m2_dev_macro_f1": dev_mf1, "m2_dev_weighted_f1": m2["best"]["dev_weighted_f1"],
            "chance_majority": ch["majority_macro_f1"], "chance_random": ch["random_macro_f1"],
            "m2_band_lo": band_lo, "m2_band_hi": band_hi, "m2_epochs": m2["epochs_run"], "m2_steps": m2.get("total_steps"),
            "m2_pass": m2_pass,
        })
        print(f"{fam}: M1 {'PASS' if m1_pass else 'FAIL'} (99% at step {m1['step_reached_99']}, final acc {m1['final_train_accuracy']:.3f}); "
              f"M2 {'PASS' if m2_pass else 'FAIL'} (dev macro F1 {dev_mf1:.3f}, chance majority {ch['majority_macro_f1']:.3f} / random "
              f"{ch['random_macro_f1']:.3f}, band [{band_lo:.3f}, {band_hi:.3f}], {m2['epochs_run']} epoch(s))")

    lines = ["# Gate 4 sanity checks (M1, M2)", "",
             f"Command: `{' '.join(sys.argv)}`; runs under `{args.runs_root}`; seed {args.seed}; batch {args.batch_size}; "
             f"M1 max steps {args.steps}; M2 {args.m2_epochs} epoch(s)" + (f", max steps {args.m2_max_steps}" if args.m2_max_steps else " (no step cap)") + ".", "",
             "| Family | M1 step @99% | M1 final acc | M1 steps | s/step | M1 | M2 dev macro F1 | M2 dev wF1 | chance (majority) | chance (random) | M2 band | M2 epochs | M2 steps | M2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['family']} | {r['m1_step']} | {r['m1_final_acc']:.3f} | {r['m1_steps_run']} | {r['m1_s_per_step']:.2f} | "
                     f"{'PASS' if r['m1_pass'] else 'FAIL'} | {r['m2_dev_macro_f1']:.3f} | {r['m2_dev_weighted_f1']:.3f} | "
                     f"{r['chance_majority']:.3f} | {r['chance_random']:.3f} | [{r['m2_band_lo']:.3f}, {r['m2_band_hi']:.3f}] | {r['m2_epochs']} | "
                     f"{r['m2_steps'] if r['m2_steps'] is not None else '-'} | {'PASS' if r['m2_pass'] else 'FAIL'} |")
    lines += ["", "M1: one fixed batch, data augmentation and SpecAugment off, PASS when train accuracy >= 0.99 within the step budget.",
              "M2: train labels permuted with the seed, trained for the full --m2-epochs; PASS when dev macro F1 lies within +-3 points of chance, "
              "i.e. in [min(chance majority, chance random) - 0.03, max(...) + 0.03]. Above the band: leakage or a shortcut; below it: the control did not reach chance.", ""]
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
