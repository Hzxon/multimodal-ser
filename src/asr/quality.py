"""ASR quality against the gold transcripts (PLAN.md Gate 3 F2, preprocessing_validation.md Stage 3).

Usage: python -m src.asr.quality --tag whisper-small [--manifest-dir DIR] [--out reports/asr_quality_whisper-small.md]

Reports WER per split and per split x class, the ASR/gold word-count ratio, hallucination
flag counts, WER by duration bucket and the 15 highest-WER rows. Writes a Markdown report,
``reports/figures/wer_per_class_<tag>.png`` and ``data/processed/asr_quality_<tag>.csv``.
With a non-default ``--manifest-dir`` (smoke manifests) every output goes to
``data/processed/smoke_reports/`` instead.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import EMOTIONS, ROOT, SPLITS, load_yaml, normalize_text
from src.data.dataset import load_manifest, manifest_dir

DURATION_BUCKETS = [("<1 s", 0.0, 1.0), ("1-3 s", 1.0, 3.0), ("3-10 s", 3.0, 10.0), (">10 s", 10.0, float("inf"))]
SMOKE_DIR = ROOT / "data/processed/smoke_reports"


def wer_normalize(s: str) -> str:
    """Both sides of the WER: normalize_text, lower-case, hyphens to spaces, punctuation
    removed (apostrophes too, on both sides), whitespace collapsed."""
    s = normalize_text(s).lower().replace("-", " ")
    s = "".join(c for c in s if not unicodedata.category(c).startswith("P"))
    return re.sub(r"\s+", " ", s).strip()


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(word errors, reference words) of one normalised pair; (0, 0) for an empty reference."""
    import jiwer

    if not ref:
        return 0, 0
    o = jiwer.process_words([ref], [hyp])
    return o.substitutions + o.deletions + o.insertions, len(ref.split())


def is_smoke(manifest_dir_arg: str | Path | None) -> bool:
    return manifest_dir_arg is not None and manifest_dir(manifest_dir_arg) != manifest_dir()


def load_rows(tag: str, manifest_dir_arg: str | Path | None = None, splits=SPLITS) -> pd.DataFrame:
    """One row per transcribed clip (exclusions kept, flagged by ``exclude_reason``) with
    per-row error counts. Splits without an ASR file are skipped."""
    from src.data.dataset import asr_path

    frames = []
    for split in splits:
        if not asr_path(tag, split, manifest_dir_arg).exists():
            continue
        df = load_manifest(split, require_audio=True, apply_exclusions=False, asr_tag=tag,
                           manifest_dir_override=manifest_dir_arg)
        df["gold_norm"] = df["gold_text"].map(wer_normalize)
        df["asr_norm"] = df["asr_text"].map(wer_normalize)
        errs = [word_errors(r, h) for r, h in zip(df["gold_norm"], df["asr_norm"])]
        df["errors"] = [e for e, _ in errs]
        df["ref_words"] = [n for _, n in errs]
        df["asr_words"] = df["asr_norm"].map(lambda t: len(t.split()))
        df["wer"] = np.where(df["ref_words"] > 0, df["errors"] / df["ref_words"].clip(lower=1), np.nan)
        df["ratio"] = np.where(df["ref_words"] > 0, df["asr_words"] / df["ref_words"].clip(lower=1), np.nan)
        df["asr_flag"] = df["asr_flag"].fillna("")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no ASR files for tag {tag!r} under {manifest_dir(manifest_dir_arg) / 'asr' / tag}")
    return pd.concat(frames, ignore_index=True)


def corpus_wer(df: pd.DataFrame) -> float:
    n = df["ref_words"].sum()
    return float(df["errors"].sum() / n) if n else float("nan")


def wer_per_split(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        d = df[df["split"] == split]
        if d.empty:
            continue
        kept = d[d["exclude_reason"] == ""]
        rows.append({"split": split, "clips": len(d), "ref_words": int(d["ref_words"].sum()),
                     "WER": corpus_wer(d), "WER_policy_kept": corpus_wer(kept),
                     "empty_ref": int((d["ref_words"] == 0).sum())})
    return pd.DataFrame(rows)


def wer_per_class(df: pd.DataFrame) -> pd.DataFrame:
    """WER (%) per class (rows) x split (columns) plus the class mean duration and gold words."""
    out = pd.DataFrame(index=EMOTIONS)
    for split in SPLITS:
        d = df[df["split"] == split]
        if d.empty:
            continue
        out[split] = [corpus_wer(d[d["emotion"] == e]) for e in EMOTIONS]
    out["n_clips"] = [int((df["emotion"] == e).sum()) for e in EMOTIONS]
    out["mean_dur_s"] = [df.loc[df["emotion"] == e, "duration_s"].mean() for e in EMOTIONS]
    out["mean_gold_words"] = [df.loc[df["emotion"] == e, "ref_words"].mean() for e in EMOTIONS]
    return out


def ratio_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        r = df.loc[(df["split"] == split) & df["ratio"].notna(), "ratio"]
        if r.empty:
            continue
        rows.append({"split": split, "median": r.median(), "p10": r.quantile(0.1), "p90": r.quantile(0.9)})
    return pd.DataFrame(rows)


def flag_counts(df: pd.DataFrame) -> pd.DataFrame:
    names = ["empty", "too_long", "repeat_3gram", "filler_on_silence"]
    rows = []
    for split in SPLITS:
        d = df[df["split"] == split]
        if d.empty:
            continue
        flags = d["asr_flag"].str.split(",")
        row = {"split": split, "clips": len(d)}
        for n in names:
            row[n] = int(flags.map(lambda f: n in f).sum())
        anyflag = int((d["asr_flag"] != "").sum())
        row["any"] = anyflag
        row["any_%"] = 100.0 * anyflag / len(d)
        rows.append(row)
    return pd.DataFrame(rows)


def wer_by_duration(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, lo, hi in DURATION_BUCKETS:
        d = df[(df["duration_s"] >= lo) & (df["duration_s"] < hi)]
        row = {"bucket": name, "clips": len(d)}
        for split in SPLITS:
            ds = d[d["split"] == split]
            row[split] = corpus_wer(ds) if len(ds) else float("nan")
        row["all"] = corpus_wer(d) if len(d) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def top_wer_rows(df: pd.DataFrame, k: int = 15, min_ref_words: int = 3) -> pd.DataFrame:
    """Highest per-row WER among clips with at least ``min_ref_words`` gold words, so that
    one-word clips do not fill the list; ties broken by more errors."""
    d = df[df["ref_words"] >= min_ref_words].sort_values(["wer", "errors"], ascending=False).head(k)
    return d[["utt_id", "emotion", "duration_s", "wer", "gold_text", "asr_text"]]


def plot_wer_per_class(per_class: pd.DataFrame, tag: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [s for s in SPLITS if s in per_class.columns]
    colors = {"train": "#4c78a8", "dev": "#f58518", "test": "#54a24b"}
    x = np.arange(len(per_class))
    w = 0.8 / max(len(splits), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, s in enumerate(splits):
        ax.bar(x + (i - (len(splits) - 1) / 2) * w, 100 * per_class[s].to_numpy(), w, label=s, color=colors[s])
    ax.set_xticks(x)
    ax.set_xticklabels(per_class.index)
    ax.set_ylabel("WER (%)")
    ax.set_title(f"WER per class, {tag}")
    ax.legend(title="split")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _md(df: pd.DataFrame, pct_cols: tuple[str, ...] = (), floatfmt: str = ".2f") -> str:
    d = df.copy()
    for c in pct_cols:
        d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{100 * v:.1f}")
    for c in d.columns:
        if c not in pct_cols and d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    cols = [d.index.name or ""] + list(map(str, d.columns)) if d.index.name else list(map(str, d.columns))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for idx, r in d.iterrows():
        cells = ([str(idx)] if d.index.name else []) + [str(v).replace("|", "\\|") for v in r.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(df: pd.DataFrame, tag: str, fig_path: Path, manifest_dir_arg: str | Path | None) -> str:
    per_split = wer_per_split(df)
    per_class = wer_per_class(df)
    per_class.index.name = "class"
    by_dur = wer_by_duration(df)
    top = top_wer_rows(df)
    models = sorted(df["asr_model"].unique())
    rel_fig = fig_path.relative_to(ROOT) if fig_path.is_relative_to(ROOT) else fig_path
    parts = [
        f"# ASR quality: {tag}",
        "",
        f"Manifests: `{manifest_dir(manifest_dir_arg).relative_to(ROOT) if manifest_dir(manifest_dir_arg).is_relative_to(ROOT) else manifest_dir(manifest_dir_arg)}`. "
        f"ASR model string(s): `{'`, `'.join(models)}`. Decoding config: `configs/asr.yaml`.",
        "WER is corpus-level (sum of word errors over sum of gold words) after `wer_normalize` on both sides "
        "(normalize_text, lower-case, hyphens to spaces, punctuation removed). Digits are not spelled out, "
        "so numerals in the ASR text count as errors against spelled-out gold text. Values in percent. "
        "Clips with an empty normalised gold text are left out of the WER.",
        "",
        "## WER per split",
        "",
        _md(per_split, pct_cols=("WER", "WER_policy_kept")),
        "",
        "`WER_policy_kept` drops the train rows excluded by the data policy (`exclude_reason` set).",
        "",
        "## WER per split x class",
        "",
        _md(per_class, pct_cols=tuple(s for s in SPLITS if s in per_class.columns)),
        "",
        f"![WER per class]({rel_fig})",
        "",
        "## ASR / gold word-count ratio",
        "",
        _md(ratio_stats(df)),
        "",
        "## Hallucination flags per split",
        "",
        _md(flag_counts(df)),
        "",
        "## WER by clip duration",
        "",
        _md(by_dur, pct_cols=("train", "dev", "test", "all")),
        "",
        "## 15 highest-WER rows (gold text with at least 3 words)",
        "",
        "Manual plausibility check (preprocessing_validation.md, Stage 3): errors should be acoustic "
        "(overlap, laughter, shouting), not systematic (wrong language, truncated audio).",
        "",
        _md(top.assign(wer=top["wer"].round(3), duration_s=top["duration_s"].round(2)).reset_index(drop=True)),
        "",
    ]
    return "\n".join(parts)


def default_paths(tag: str, smoke: bool) -> dict[str, Path]:
    if smoke:
        return {"report": SMOKE_DIR / f"asr_quality_{tag}.md", "fig": SMOKE_DIR / f"wer_per_class_{tag}.png",
                "csv": SMOKE_DIR / f"asr_quality_{tag}.csv"}
    return {"report": ROOT / f"reports/asr_quality_{tag}.md", "fig": ROOT / f"reports/figures/wer_per_class_{tag}.png",
            "csv": ROOT / f"data/processed/asr_quality_{tag}.csv"}


def run(tag: str, manifest_dir_arg: str | Path | None = None, out: str | Path | None = None) -> Path:
    paths = default_paths(tag, is_smoke(manifest_dir_arg))
    if out is not None:
        paths["report"] = Path(out) if Path(out).is_absolute() else ROOT / out
    df = load_rows(tag, manifest_dir_arg)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    plot_wer_per_class(wer_per_class(df), tag, paths["fig"])
    keep = ["utt_id", "split", "emotion", "duration_s", "exclude_reason", "gold_text", "asr_text",
            "ref_words", "asr_words", "errors", "wer", "ratio", "asr_flag", "asr_model"]
    df[keep].to_csv(paths["csv"], index=False)
    paths["report"].write_text(build_report(df, tag, paths["fig"], manifest_dir_arg))
    print(f"wrote {paths['report']}\nwrote {paths['fig']}\nwrote {paths['csv']}")
    return paths["report"]


def main() -> None:
    cfg = load_yaml("configs/asr.yaml")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=cfg["tag"], help="ASR tag (directory under <manifest_dir>/asr/)")
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--out", default=None, help="Markdown report path")
    a = ap.parse_args()
    run(a.tag, a.manifest_dir, a.out)


if __name__ == "__main__":
    main()
