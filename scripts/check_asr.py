"""Gate 3, ASR part (PLAN.md F1, F2; preprocessing_validation.md Stage 3).

Usage: python scripts/check_asr.py --tag whisper-small [--manifest-dir DIR] [--n 100] [--out reports/gate3_asr.md]
                                    [--device cuda|mps|cpu] [--batch-size N]

Checks: determinism (re-transcribe --n random dev clips and compare with the cached file),
pinned asr_model, coverage, hallucination rate per split, WER per split and the dev/test gap,
ASR/gold length ratio. Prints and writes a Markdown table; exits 1 on any FAIL.
With a non-default --manifest-dir the default output is data/processed/smoke_reports/gate3_asr.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import quality  # noqa: E402
from src.asr.transcribe import (Transcriber, asr_model_string, read_asr_file, resolve_model,  # noqa: E402
                                transcribe_rows)
from src.common import ROOT, SPLITS, load_yaml, set_seed  # noqa: E402
from src.data.dataset import asr_path, load_manifest  # noqa: E402

HALLUCINATION_MAX = 0.02
WER_GAP_MAX = 0.10
RATIO_RANGE = (0.8, 1.2)
MIN_ROWS_FOR_RATE = 200  # below this a single flagged clip already exceeds 2 %, so the rate check only warns


def check_determinism(cfg, tag, model_id, mdir, n, device, batch_size, seed=13) -> tuple[str, str]:
    split = "dev"
    if not asr_path(tag, split, mdir).exists():
        return "SKIP", f"no ASR file for {split}"
    cached = read_asr_file(asr_path(tag, split, mdir)).set_index("utt_id")
    rows = load_manifest(split, require_audio=True, apply_exclusions=False, manifest_dir_override=mdir)
    rows = rows[rows["utt_id"].isin(cached.index)]
    k = min(n, len(rows))
    sample = rows.sample(n=k, random_state=seed)
    try:
        tr = Transcriber(cfg, model_id, device)
    except Exception as e:  # model files unavailable, no network
        return "SKIP", f"model not loadable: {type(e).__name__}: {str(e)[:100]}"
    new = transcribe_rows(tr, sample, batch_size, cfg, asr_model_string(cfg, model_id)).set_index("utt_id")
    diff = [u for u in sample["utt_id"] if new.loc[u, "asr_text"] != cached.loc[u, "asr_text"]]
    ev = f"{k - len(diff)}/{k} identical (device {tr.device}, batch {batch_size}, cache written with its own batching)"
    if diff:
        u = diff[0]
        ev += f"; first diff {u}: cached={cached.loc[u, 'asr_text']!r} new={new.loc[u, 'asr_text']!r}"
    return ("PASS" if not diff else "FAIL"), ev


def check_pinned(cfg, tag, model_id, mdir) -> tuple[str, str]:
    expected = asr_model_string(cfg, model_id)
    bad, seen = [], []
    for s in SPLITS:
        p = asr_path(tag, s, mdir)
        if not p.exists():
            continue
        vals = set(read_asr_file(p)["asr_model"].unique())
        seen.append(s)
        if vals != {expected}:
            bad.append(f"{s}: {sorted(vals)}")
    if not seen:
        return "SKIP", "no ASR files"
    return ("PASS", f"{', '.join(seen)}: one string = `{expected}`") if not bad else ("FAIL", "; ".join(bad) + f"; expected `{expected}`")


def main() -> int:
    cfg = load_yaml("configs/asr.yaml")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=cfg["tag"], choices=[cfg["tag"], cfg["final"]["tag"]])
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--n", type=int, default=100, help="dev clips re-transcribed for the determinism check")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    a = ap.parse_args()
    set_seed(13)
    tag, model_id = resolve_model(cfg, a.tag)
    mdir = a.manifest_dir
    out = Path(a.out) if a.out else (quality.SMOKE_DIR / "gate3_asr.md" if quality.is_smoke(mdir) else ROOT / "reports/gate3_asr.md")
    out = out if out.is_absolute() else ROOT / out

    rows: list[tuple[str, str, str, str]] = []
    rows.append(("F1 determinism", f"{a.n} random dev clips re-transcribed give identical strings (100 %)",
                 *check_determinism(cfg, tag, model_id, mdir, a.n, a.device, a.batch_size)))
    rows.append(("Pinned", "asr_model is one string per file and equals the string built from configs/asr.yaml and the installed versions",
                 *check_pinned(cfg, tag, model_id, mdir)))

    df = None
    try:
        df = quality.load_rows(tag, mdir)
    except FileNotFoundError as e:
        rows.append(("Coverage", "every has_audio manifest row has an ASR row", "SKIP", str(e)))
    cov_ev, cov_bad = [], False
    for s in SPLITS:
        if not asr_path(tag, s, mdir).exists():
            cov_ev.append(f"{s}: no file")
            cov_bad = True
            continue
        man = load_manifest(s, require_audio=True, apply_exclusions=False, manifest_dir_override=mdir)
        asr = read_asr_file(asr_path(tag, s, mdir))
        missing = set(man["utt_id"]) - set(asr["utt_id"])
        extra = set(asr["utt_id"]) - set(man["utt_id"])
        empty = int((asr["asr_text"].str.strip() == "").sum())
        cov_ev.append(f"{s}: {len(man) - len(missing)}/{len(man)} covered, {len(extra)} extra, {empty} empty")
        cov_bad |= bool(missing or extra)
    if df is not None:
        rows.append(("Coverage", "every has_audio manifest row has an ASR row (empty transcripts counted)",
                     "FAIL" if cov_bad else "PASS", "; ".join(cov_ev)))

        fc = quality.flag_counts(df)
        for _, r in fc.iterrows():
            rate = r["any"] / r["clips"]
            st = "PASS" if rate < HALLUCINATION_MAX else ("WARN" if r["clips"] < MIN_ROWS_FOR_RATE else "FAIL")
            rows.append((f"Hallucination rate, {r['split']}", f"flagged share < {HALLUCINATION_MAX:.0%}", st,
                         f"{int(r['any'])}/{int(r['clips'])} = {rate:.1%} (empty {int(r['empty'])}, too_long {int(r['too_long'])}, "
                         f"repeat_3gram {int(r['repeat_3gram'])}, filler_on_silence {int(r['filler_on_silence'])})"))

        ps = quality.wer_per_split(df).set_index("split")
        rows.append(("F2 WER reported", "WER per split computed (per class in the quality report)", "PASS",
                     ", ".join(f"{s} {100 * ps.loc[s, 'WER']:.1f} %" for s in ps.index)))
        if {"dev", "test"} <= set(ps.index):
            gap = abs(ps.loc["dev", "WER"] - ps.loc["test", "WER"])
            rows.append(("F2 dev/test WER gap", f"|dev - test| <= {100 * WER_GAP_MAX:.0f} points",
                         "PASS" if gap <= WER_GAP_MAX else "FAIL", f"{100 * gap:.1f} points"))
        else:
            rows.append(("F2 dev/test WER gap", f"|dev - test| <= {100 * WER_GAP_MAX:.0f} points", "SKIP", "dev or test ASR file missing"))

        rs = quality.ratio_stats(df)
        for _, r in rs.iterrows():
            ok = RATIO_RANGE[0] <= r["median"] <= RATIO_RANGE[1]
            rows.append((f"Length ratio, {r['split']}", f"median ASR/gold words in [{RATIO_RANGE[0]}, {RATIO_RANGE[1]}]",
                         "PASS" if ok else "FAIL", f"median {r['median']:.2f}, p10 {r['p10']:.2f}, p90 {r['p90']:.2f}"))

    md = ["# Gate 3: ASR checks", "",
          f"Tag `{tag}`, model `{model_id}`, manifests `{a.manifest_dir or 'data/manifests'}`.", "",
          "| Check | Criterion | Result | Evidence |", "|---|---|---|---|"]
    esc = lambda t: t.replace("|", "\\|")
    md += [f"| {c} | {k} | {s} | {esc(e)} |" for c, k, s, e in rows]
    text = "\n".join(md) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"wrote {out}")
    return 1 if any(r[2] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
