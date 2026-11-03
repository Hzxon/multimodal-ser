"""Hyper-parameter grid on dev for one model family (PLAN.md 3.4: at most 12 configs per family).

Usage: python scripts/hparam_grid.py --family {text_roberta,audio_wavlm_ft,fusion_gated,fusion_xattn}
       [--grid configs/grids/<family>.yaml] [--seed 13] [--max-configs 12] [--manifest-dir DIR]
       [--feature-root DIR] [--runs-root runs/grid] [--init-root runs] [--report-dir reports]
       [--best-dir configs] [--dry-run] [--force] [--max-steps N] [--max-epochs N] [--batch-size N]
       [--device cuda|mps|cpu]

The grid file lists dotted config keys with their candidate values; the product of the lists may not
exceed 12 (error otherwise) and ``--max-configs N`` runs only the first N. Each config is one
``python -m src.train`` run named ``<family>_g<i>`` under ``--runs-root``. The report
``<report-dir>/hparam_<family>.md`` lists dev weighted F1 and macro F1 read from each run's
``metrics.json`` (never test: Gates M3, M4). The winner, by dev weighted F1 with dev macro F1 as
tie-break, is written to ``<best-dir>/<family>_best.yaml`` as the base config plus the winning values.
With ``--manifest-dir`` set, the runs root, report dir and best dir default to
data/processed/smoke_runs/grid and data/processed/smoke_reports.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_yaml, set_nested  # noqa: E402

MAX_CONFIGS = 12
FAMILIES = ("text_roberta", "audio_wavlm_ft", "fusion_gated", "fusion_xattn")
SMOKE_RUNS = "data/processed/smoke_runs/grid"
SMOKE_REPORTS = "data/processed/smoke_reports"


def yaml_scalar(v) -> str:
    """One-line YAML text of a scalar that ``yaml.safe_load`` reads back as the same value."""
    return yaml.safe_dump(v, default_flow_style=True).splitlines()[0]


def get_nested(cfg: dict, dotted: str):
    d = cfg
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of the grid lists, in file order: [{dotted_key: value}, ...]."""
    keys = list(grid)
    for k in keys:
        assert isinstance(grid[k], list) and grid[k], f"grid entry {k!r} must be a non-empty list"
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def read_dev(run_dir: Path) -> dict | None:
    p = run_dir / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    b = m.get("best", {})
    return {"dev_weighted_f1": b.get("dev_weighted_f1"), "dev_macro_f1": b.get("dev_macro_f1"),
            "best_epoch": b.get("epoch"), "epochs_run": m.get("epochs_run"), "wall_seconds": m.get("wall_seconds")}


def run_matches(run_dir: Path, overrides: dict) -> bool:
    """True when the run's stored config already carries every expected value: the grid point,
    the init_from paths and the CLI step budget / data paths."""
    p = run_dir / "config.yaml"
    if not p.exists() or not (run_dir / "metrics.json").exists():
        return False
    cfg = yaml.safe_load(p.read_text())
    return all(get_nested(cfg, k) == v for k, v in overrides.items())


def abs_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--grid", help="grid file (default configs/grids/<family>.yaml)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-configs", type=int, default=MAX_CONFIGS, help=f"run only the first N configs (at most {MAX_CONFIGS})")
    ap.add_argument("--manifest-dir")
    ap.add_argument("--feature-root")
    ap.add_argument("--runs-root")
    ap.add_argument("--init-root", help="where the unimodal runs for init_from live (default runs, or the smoke runs root)")
    ap.add_argument("--report-dir")
    ap.add_argument("--best-dir")
    ap.add_argument("--dry-run", action="store_true", help="print the configs and commands, run nothing")
    ap.add_argument("--force", action="store_true", help="retrain configs that already have a matching run")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"])
    a = ap.parse_args()

    if a.max_configs > MAX_CONFIGS:
        ap.error(f"--max-configs {a.max_configs} exceeds the PLAN.md cap of {MAX_CONFIGS}")
    smoke = a.manifest_dir is not None
    runs_root = a.runs_root or (SMOKE_RUNS if smoke else "runs/grid")
    init_root = a.init_root or ("data/processed/smoke_runs" if smoke else "runs")
    report_dir = abs_path(a.report_dir or (SMOKE_REPORTS if smoke else "reports"))
    best_dir = abs_path(a.best_dir or (SMOKE_REPORTS if smoke else "configs"))

    grid_path = a.grid or f"configs/grids/{a.family}.yaml"
    spec = load_yaml(grid_path)
    assert spec["family"] == a.family, f"{grid_path} is for family {spec['family']!r}"
    points = expand_grid(spec["grid"])
    if len(points) > MAX_CONFIGS:
        print(f"error: {grid_path} expands to {len(points)} configs; PLAN.md caps a family at {MAX_CONFIGS}", file=sys.stderr)
        return 2
    points = points[: a.max_configs]
    base_cfg = load_yaml(spec["base_config"])
    init_over = {k: v.format(init_root=init_root, seed=a.seed) for k, v in (spec.get("init_from") or {}).items()}
    # CLI values that src.train writes into config.yaml (a CLI flag beats a --set of the same key); a stored run
    # is reused only when these match too, so a run with another step budget or other manifests is retrained
    cli = {"data.manifest_dir": a.manifest_dir, "data.feature_root": a.feature_root, "train.max_steps": a.max_steps,
           "train.max_epochs": a.max_epochs, "train.batch_size": a.batch_size}
    cli_defaults = {k: get_nested(base_cfg, k) for k, v in cli.items() if v is None}
    cli_given = {k: v for k, v in cli.items() if v is not None}
    print(f"family {a.family}: {len(points)} config(s) from {grid_path}, seed {a.seed}, runs under {runs_root}")

    rows = []
    for i, point in enumerate(points):
        name = f"{a.family}_g{i}"
        cmd = [sys.executable, "-m", "src.train", "--config", spec["base_config"], "--seed", str(a.seed),
               "--name", name, "--runs-root", runs_root]
        for flag, val in (("--manifest-dir", a.manifest_dir), ("--feature-root", a.feature_root), ("--device", a.device),
                          ("--max-steps", a.max_steps), ("--max-epochs", a.max_epochs), ("--batch-size", a.batch_size)):
            if val is not None:
                cmd += [flag, str(val)]
        for k, v in {**init_over, **point}.items():
            cmd += ["--set", f"{k}={yaml_scalar(v)}"]
        run_dir = abs_path(runs_root) / name / str(a.seed)
        desc = ", ".join(f"{k.split('.', 1)[1]}={v}" for k, v in point.items())
        print(f"[g{i}] {desc}\n  $ {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        if not a.force and run_matches(run_dir, {**cli_defaults, **init_over, **point, **cli_given}):
            print(f"  reusing {run_dir / 'metrics.json'} (config matches; --force to retrain)")
        else:
            t0 = time.time()
            subprocess.run(cmd, cwd=ROOT, check=True)
            print(f"  [g{i}] finished in {time.time() - t0:.0f}s")
        dev = read_dev(run_dir)
        assert dev is not None and dev["dev_weighted_f1"] is not None, f"no dev metrics in {run_dir}"
        rows.append({"i": i, "name": name, "point": point, **dev})
        print(f"  [g{i}] dev wF1 {dev['dev_weighted_f1']:.4f}  dev mF1 {dev['dev_macro_f1']:.4f}")

    if a.dry_run:
        print(f"dry run: {len(points)} config(s), nothing executed")
        return 0

    best = max(rows, key=lambda r: (r["dev_weighted_f1"], r["dev_macro_f1"]))
    keys = list(spec["grid"])
    short = [k.split(".", 1)[1] for k in keys]
    lines = [f"# Hyper-parameter grid: {a.family}", "",
             f"Command: `{' '.join(['python scripts/hparam_grid.py'] + sys.argv[1:])}`. Grid file `{grid_path}`, base config "
             f"`{spec['base_config']}`, seed {a.seed}, {len(rows)} of at most {MAX_CONFIGS} configs. Runs under `{runs_root}`.", "",
             "Dev split only; the test split is never read by this script or by `src.train` (Gates M3, M4). "
             "Selection: highest dev weighted F1, dev macro F1 as tie-break.", "",
             "| Config | " + " | ".join(short) + " | dev weighted F1 | dev macro F1 | best epoch | epochs run |",
             "|---|" + "---|" * len(short) + "---|---|---|---|"]
    for r in rows:
        mark = " (best)" if r is best else ""
        lines.append(f"| {r['name']}{mark} | " + " | ".join(str(r["point"][k]) for k in keys) +
                     f" | {r['dev_weighted_f1']:.4f} | {r['dev_macro_f1']:.4f} | {r['best_epoch']} | {r['epochs_run']} |")
    lines += ["", f"Best: `{best['name']}` with " + ", ".join(f"{k}={best['point'][k]}" for k in keys) +
              f" (dev weighted F1 {best['dev_weighted_f1']:.4f}, dev macro F1 {best['dev_macro_f1']:.4f}). "
              f"Written to `{rel(best_dir / (a.family + '_best.yaml'))}`.", ""]
    report_dir.mkdir(parents=True, exist_ok=True)
    out_md = report_dir / f"hparam_{a.family}.md"
    out_md.write_text("\n".join(lines))

    best_cfg = json.loads(json.dumps(base_cfg))
    for k, v in best["point"].items():
        set_nested(best_cfg, k, v)
    best_cfg["name"] = f"{a.family}_best"
    best_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = best_dir / f"{a.family}_best.yaml"
    header = (f"# Winner of the dev grid in {out_md.name}: {best['name']} "
              f"(dev weighted F1 {best['dev_weighted_f1']:.4f}, dev macro F1 {best['dev_macro_f1']:.4f}).\n"
              f"# Base config {spec['base_config']} with the grid values applied; everything else is unchanged.\n")
    out_yaml.write_text(header + yaml.safe_dump(best_cfg, sort_keys=False))
    print(f"wrote {out_md}\nwrote {out_yaml}\nbest: {best['name']} dev wF1 {best['dev_weighted_f1']:.4f} mF1 {best['dev_macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
