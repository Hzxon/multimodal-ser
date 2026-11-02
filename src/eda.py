"""Gate 2: exploratory data analysis on the manifests.

Usage: python -m src.eda [--jobs N]

Text and label analyses always run. Audio analyses run when the manifests carry
has_audio (built without --labels-only); per-clip audio statistics are cached in
data/processed/audio_stats.csv. Writes reports/eda_summary.md, reports/gate2.md and
figures under reports/figures/.
"""
import argparse
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
SPLITS = ["train", "dev", "test"]
MAIN_CAST = ["Chandler", "Joey", "Monica", "Phoebe", "Rachel", "Ross"]
STOP = set("""a an the and or but if of to in on at for with from by as is are was were be been being am i you he she it we
they me him her us them my your his its our their this that these those there here what which who whom when where why how
not no nor so than too very just do does did doing have has had having will would can could should may might must shall
oh uh um yeah yes okay ok well hey like get got go going gonna wanna know think mean really dont don't im i'm its it's
thats that's youre you're ive i've id i'd ill i'll he's she's we're they're isn't aren't wasn't weren't didn't doesn't
can't couldn't won't wouldn't about out up down over all some any one two come came say said see saw look now then
also into back then""".split())
TOKEN_RE = re.compile(r"[a-z']+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("’", "'"))


# ---------------------------------------------------------------- audio stats
def audio_stats(path: str) -> dict:
    import librosa
    import soundfile as sf

    y, sr = sf.read(str(ROOT / path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    frame, hop = 1024, 256
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    rms_db = 20 * np.log10(rms + 1e-8)
    f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr, frame_length=2048, hop_length=hop)
    voiced = rms_db[: len(f0)] > (rms_db.max() - 25)
    f0_med = float(np.median(f0[voiced])) if voiced.any() else float("nan")
    return {
        "rms_db": float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-8)),
        "f0_median_hz": f0_med,
        "dyn_range_db": float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)),
        "silence_frac": float((rms_db < rms_db.max() - 40).mean()),
        "clipped_frac": float((np.abs(y) >= 0.99).mean()),
    }


def load_audio_stats(df: pd.DataFrame, jobs: int) -> pd.DataFrame:
    split = df["split"].iloc[0]
    cache = ROOT / f"data/processed/audio_stats_{split}.csv"  # one cache per split (never mixed)
    if cache.exists():
        st = pd.read_csv(cache)
        if set(st["utt_id"]) >= set(df.loc[df["has_audio"], "utt_id"]):
            return df.merge(st, on="utt_id", how="left")
    sub = df[df["has_audio"]]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(audio_stats, sub["wav_path"], chunksize=32))
    st = pd.DataFrame(res)
    st.insert(0, "utt_id", sub["utt_id"].values)
    st.to_csv(cache, index=False)
    return df.merge(st, on="utt_id", how="left")


# ---------------------------------------------------------------- analyses
def shortcut_audit(train: pd.DataFrame, dev: pd.DataFrame, feats: list[str]) -> dict:
    tr, dv = train.dropna(subset=feats), dev.dropna(subset=feats)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(tr[feats], tr["emotion"])
    pred = clf.predict(dv[feats])
    maj = tr["emotion"].mode()[0]
    return {
        "features": feats,
        "shortcut_macro_f1": round(f1_score(dv["emotion"], pred, average="macro"), 4),
        "shortcut_weighted_f1": round(f1_score(dv["emotion"], pred, average="weighted"), 4),
        "majority_macro_f1": round(f1_score(dv["emotion"], [maj] * len(dv), average="macro"), 4),
        "majority_weighted_f1": round(f1_score(dv["emotion"], [maj] * len(dv), average="weighted"), 4),
    }


def top_words(train: pd.DataFrame, emotions: list[str], k: int = 12) -> pd.DataFrame:
    per = {e: Counter() for e in emotions}
    for e, t in zip(train["emotion"], train["gold_text"]):
        per[e].update(w for w in set(tokens(t)) if w not in STOP and len(w) > 1)
    total = Counter()
    for c in per.values():
        total.update(c)
    out = {}
    for e in emotions:
        n_e = (train["emotion"] == e).sum()
        n_rest = len(train) - n_e
        scores = {}
        for w, c in per[e].items():
            if total[w] < 8:
                continue
            rest = total[w] - c
            scores[w] = np.log((c + 0.5) / (n_e + 1)) - np.log((rest + 0.5) / (n_rest + 1))
        out[e] = [f"{w} ({per[e][w]})" for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]
    return pd.DataFrame(out)


def duplicate_text_ceiling(train: pd.DataFrame, dev: pd.DataFrame) -> dict:
    norm = lambda s: " ".join(tokens(s))
    tr = train.assign(key=train["gold_text"].map(norm))
    dv = dev.assign(key=dev["gold_text"].map(norm))
    maj = tr.groupby("key")["emotion"].agg(lambda x: x.mode()[0])
    seen = dv[dv["key"].isin(maj.index)]
    acc = float((seen["key"].map(maj) == seen["emotion"]).mean()) if len(seen) else float("nan")
    top = tr.groupby("key").agg(n=("emotion", "size"), labels=("emotion", lambda x: dict(Counter(x))))
    top = top[top["n"] >= 20].sort_values("n", ascending=False).head(15)
    return {"dev_rows_with_text_seen_in_train": int(len(seen)), "dev_rows": int(len(dv)),
            "majority_label_accuracy_on_seen": round(acc, 4), "frequent_texts": top}


def class_table(df: pd.DataFrame, emotions: list[str], cols: list[str]) -> pd.DataFrame:
    g = df.groupby("emotion")[cols].agg(["mean", "median"]).reindex(emotions)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g.insert(0, "n", df["emotion"].value_counts().reindex(emotions))
    return g.round(3)


def box(df: pd.DataFrame, col: str, emotions: list[str], name: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="emotion", y=col, order=emotions, ax=ax, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=120)
    plt.close(fig)


def spectrogram_grid(df: pd.DataFrame, emotions: list[str]) -> None:
    import librosa
    import librosa.display
    import soundfile as sf

    fig, axes = plt.subplots(len(emotions), 2, figsize=(10, 2.2 * len(emotions)))
    rng = np.random.default_rng(0)
    for i, e in enumerate(emotions):
        rows = df[(df["emotion"] == e) & df["has_audio"] & (df["duration_s"].between(1.5, 5))]
        for j, (_, r) in enumerate(rows.sample(2, random_state=int(rng.integers(1e6))).iterrows()):
            y, sr = sf.read(str(ROOT / r["wav_path"]), dtype="float32")
            S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64), ref=np.max)
            librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", ax=axes[i, j])
            axes[i, j].set_title(f"{e}: {r['gold_text'][:45]}", fontsize=8)
            axes[i, j].set_xlabel("")
            axes[i, j].set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG / "spectrogram_examples.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    emotions = cfg["emotions"]
    FIG.mkdir(parents=True, exist_ok=True)
    str_cols = {c: "string" for c in ["exclude_reason", "sha256", "wav_path"]}
    dfs = {s: pd.read_csv(ROOT / cfg["manifest_dir"] / f"{s}.csv", dtype=str_cols) for s in SPLITS}
    for s, d in dfs.items():
        d["exclude_reason"] = d["exclude_reason"].fillna("")
        d["n_tokens"] = d["gold_text"].map(lambda t: len(tokens(t)))
        d["main_cast"] = d["speaker"].where(d["speaker"].isin(MAIN_CAST), "Other")
    has_audio = all("has_audio" in d for d in dfs.values())
    if has_audio:
        for s in SPLITS:
            dfs[s] = load_audio_stats(dfs[s], args.jobs)
    train = dfs["train"][dfs["train"]["exclude_reason"] == ""]
    dev = dfs["dev"]
    allsplits = pd.concat(dfs.values(), ignore_index=True)

    md = ["# EDA summary (auto-generated by src/eda.py)", "",
          f"Audio analyses included: {has_audio}. Train rows used: {len(train)} (duplicates excluded).", ""]
    gate: list[tuple[str, str, str, str]] = []

    # E2 class-conditional stats
    cols = ["n_tokens", "nominal_duration_s"] + (["duration_s", "rms_db", "f0_median_hz", "dyn_range_db", "silence_frac"] if has_audio else [])
    ct = class_table(train, emotions, cols)
    md += ["## E2. Class-conditional statistics (train)", "", ct.to_markdown(), ""]
    gate.append(("E2", "PASS", "Class-conditional stats table (duration, tokens, RMS, F0)", "see eda_summary.md E2"))
    box(train, "n_tokens", emotions, "tokens_per_class.png", "tokens per utterance")
    box(train, "nominal_duration_s", emotions, "nominal_duration_per_class.png", "nominal duration (s)")
    if has_audio:
        box(train, "duration_s", emotions, "duration_per_class.png", "audio duration (s)")
        box(train, "rms_db", emotions, "rms_per_class.png", "RMS (dB)")
        box(train, "f0_median_hz", emotions, "f0_per_class.png", "median F0 (Hz)")

    # E1 shortcut audit
    feats = ["n_tokens", "nominal_duration_s"] + (["duration_s", "rms_db", "dyn_range_db"] if has_audio else [])
    sc = shortcut_audit(train, dev, feats)
    md += ["## E1. Shortcut audit (logistic regression on metadata only, train → dev)", "",
           "| model | macro F1 | weighted F1 |", "|---|---|---|",
           f"| majority class | {sc['majority_macro_f1']} | {sc['majority_weighted_f1']} |",
           f"| {' + '.join(feats)} | {sc['shortcut_macro_f1']} | {sc['shortcut_weighted_f1']} |", "",
           "Any later model must beat the shortcut row by a clear margin, otherwise it has learned length or loudness, not emotion.", ""]
    gate.append(("E1", "PASS", "Shortcut floor recorded (macro and weighted F1 on dev)",
                 f"shortcut macro F1 {sc['shortcut_macro_f1']}, majority {sc['majority_macro_f1']}"))

    # Text: top words, duplicates
    tw = top_words(train, emotions)
    md += ["## Text: most class-specific words (train, log-odds vs rest, count in parentheses)", "", tw.to_markdown(index=False), ""]
    dup = duplicate_text_ceiling(train, dev)
    md += ["## Text: repeated utterances", "",
           f"{dup['dev_rows_with_text_seen_in_train']} of {dup['dev_rows']} dev utterances have text that also occurs in train "
           f"(after lower-casing and stripping punctuation). Predicting the train majority label for those gives accuracy "
           f"{dup['majority_label_accuracy_on_seen']} on that subset. These are mostly short generic lines; their label spread "
           "shows the ceiling of text-only classification on such lines.", "",
           dup["frequent_texts"].to_markdown(), ""]

    # Speaker x emotion
    spk = pd.crosstab(train["main_cast"], train["emotion"], normalize="index").reindex(columns=emotions).round(3)
    spk.insert(0, "n", train["main_cast"].value_counts())
    md += ["## Speaker × emotion (train, row-normalised)", "", spk.to_markdown(), ""]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(spk.drop(columns="n"), annot=True, fmt=".2f", cmap="Blues", ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "speaker_emotion.png", dpi=120)
    plt.close(fig)
    spk_counts = pd.crosstab(allsplits["split"], allsplits["main_cast"]).reindex(SPLITS)
    md += ["## Speaker rows per split", "", spk_counts.to_markdown(), ""]

    # Sentiment vs emotion
    se = pd.crosstab(train["emotion"], train["sentiment"]).reindex(emotions)
    md += ["## Sentiment × emotion (train)", "", se.to_markdown(), ""]

    # Audio-only extras
    if has_audio:
        q = train[["dyn_range_db", "silence_frac", "clipped_frac", "rms_db"]].describe().round(3)
        md += ["## Audio quality (train)", "", q.to_markdown(), "",
               f"Clips with clipped samples above 0.1 %: {int((train['clipped_frac'] > 0.001).sum())}. "
               f"Clips with RMS below -40 dB (near-silent): {int((train['rms_db'] < -40).sum())}. "
               f"Clips with dynamic range below 10 dB (constant background, likely laugh track or music): "
               f"{int((train['dyn_range_db'] < 10).sum())}.", ""]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.histplot(train["dyn_range_db"].dropna(), bins=50, ax=ax)
        ax.set_xlabel("frame-RMS dynamic range (dB), 95th − 10th percentile")
        fig.tight_layout()
        fig.savefig(FIG / "dyn_range_hist.png", dpi=120)
        plt.close(fig)
        spectrogram_grid(train, emotions)
        gate.append(("E3", "PASS", "Noise / quality estimate per clip reported",
                     f"near-silent {int((train['rms_db'] < -40).sum())}, low dynamic range {int((train['dyn_range_db'] < 10).sum())}"))
    else:
        gate.append(("E3", "SKIP", "Noise estimate", "labels-only manifests; rerun after audio extraction"))
    gate.append(("E4", "TODO", "Five-bullet findings with design consequences", "written by hand in reports/gates.md"))

    (ROOT / "reports/eda_summary.md").write_text("\n".join(md))
    g = ["# Gate 2: EDA checks", "", "| ID | Status | Criterion | Evidence |", "|---|---|---|---|"]
    g += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in gate]
    (ROOT / "reports/gate2.md").write_text("\n".join(g) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
