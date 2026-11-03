"""Gate 3, feature part: scaler/TF-IDF fit split, cache isolation, feature sanity, recomputation.

Usage: python scripts/check_leakage.py [--manifest-dir DIR] [--feature-root DIR] [--n 50]
           [--device {cuda,mps,cpu}] [--out reports/gate3_features.md]

Covers PLAN.md Gate 3 items F3, F4, F5, F7, F8 and the Stage 5 checks of
reports/preprocessing_validation.md (shape, one-to-one, recomputation, correlation with the
EDA statistics, crop count). Prints the table, writes it to --out, exits 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import SPLITS, device_name  # noqa: E402
from src.data.dataset import load_manifest  # noqa: E402
from src.features.cache import FEATURE_ROOT, load_cache  # noqa: E402
from src.features.egemaps import apply_scaler, load_scaler  # noqa: E402
from src.features.text_embed import fit_tfidf, load_tfidf, train_texts  # noqa: E402

KNOWN = ("egemaps", "wavlm-base-plus")
Row = tuple[str, str, str, str, str]  # id, status, criterion, evidence, detail


def row(id_: str, status: str, criterion: str, evidence: str, detail: str = "") -> Row:
    return (id_, status, criterion, evidence, detail)


def cache_files(feature_root: Path) -> list[tuple[str, str, Path]]:
    """(directory name, split, path) for every <split>.npz under the feature root."""
    out = []
    for d in sorted(p for p in feature_root.glob("*") if p.is_dir()):
        for split in SPLITS:
            if (d / f"{split}.npz").exists():
                out.append((d.name, split, d / f"{split}.npz"))
    return out


def extractor_of(dirname: str) -> str:
    for k in KNOWN:
        if dirname.startswith(k + "_"):
            return k
    return dirname.split("_")[0]


def manifest(split: str, mdir: str | None, **kw):
    return load_manifest(split, require_audio=True, apply_exclusions=False, manifest_dir_override=mdir, **kw)


# --- F3 ---------------------------------------------------------------------------------

def check_f3(feature_root: Path, mdir: str | None) -> list[Row]:
    rows: list[Row] = []
    scalers = sorted(feature_root.glob("egemaps_*/scaler.pkl"))
    if not scalers:
        rows.append(row("F3", "SKIP", "scaler.pkl fit on train only", f"no scaler.pkl under {feature_root}"))
    for sp in scalers:
        ev, status = [], "PASS"
        try:
            sc = load_scaler(sp)
            ev.append(f"fit_split={sc['fit_split']}")
        except AssertionError as e:
            rows.append(row("F3", "FAIL", f"{sp.parent.name}/scaler.pkl fit_split == train", str(e)))
            continue
        version = sc["version"]
        tr = load_cache(sp.parent / "train.npz", split="train", extractor="egemaps", version=version)
        keep = load_manifest("train", require_audio=True, apply_exclusions=True,
                             manifest_dir_override=mdir)["utt_id"].tolist()
        mask = np.isin(tr["utt_id"], keep)
        n_expected = len(keep)
        ok_n = sc["n_fit"] == n_expected == int(mask.sum())
        ev.append(f"n_fit={sc['n_fit']} vs non-excluded train rows={n_expected} (cache rows {len(tr['utt_id'])})")
        ref_mean = tr["X"][mask].astype(np.float64).mean(axis=0)
        ok_mean = np.allclose(ref_mean, sc["mean"], rtol=1e-4, atol=1e-5)
        ev.append(f"scaler mean equals mean of non-excluded train rows: {ok_mean}")
        ztr = apply_scaler(tr["X"][mask], sc)
        ev.append(f"max |mean| of scaled train = {np.abs(ztr.mean(axis=0)).max():.2e}")
        dev_path = sp.parent / "dev.npz"
        ok_dev = True
        if dev_path.exists():
            dv = load_cache(dev_path, split="dev", extractor="egemaps", version=version)
            m = np.abs(apply_scaler(dv["X"], sc).mean(axis=0))
            ok_dev = not bool((m < 1e-3).all())
            ev.append(f"scaled dev mean: max |mean| = {m.max():.3f}, columns within 1e-3 of 0: {int((m < 1e-3).sum())}/{len(m)}")
        else:
            ev.append("no dev cache, dev-mean check skipped")
            status = "WARN"
        if not (ok_n and ok_mean and ok_dev):
            status = "FAIL"
        rows.append(row("F3", status, f"{sp.parent.name}: scaler fit on train only, n_fit matches, dev mean is not 0",
                        "; ".join(ev)))

    vecs = sorted(feature_root.glob("tfidf_*/vectorizer.pkl"))
    if not vecs:
        rows.append(row("F3-tfidf", "SKIP", "TF-IDF vectoriser fit on train only", f"no vectorizer.pkl under {feature_root}"))
    for vp in vecs:
        try:
            obj = load_tfidf(vp)
        except AssertionError as e:
            rows.append(row("F3-tfidf", "FAIL", f"{vp.parent.name} fit_split == train", str(e)))
            continue
        try:
            texts = train_texts(obj["text_source"], obj["asr_tag"], mdir)
        except (FileNotFoundError, RuntimeError) as e:
            rows.append(row("F3-tfidf", "WARN", f"{vp.parent.name}: metadata and refit",
                            f"fit_split=train, n_fit={obj['n_fit']}; text not loadable: {str(e)[:120]}"))
            continue
        same_n = obj["n_fit"] == len(texts)
        refit = fit_tfidf(texts, obj["params"])
        same_vocab = refit.vocabulary_ == obj["vectorizer"].vocabulary_
        rows.append(row("F3-tfidf", "PASS" if same_n and same_vocab else "FAIL",
                        f"{vp.parent.name}: fit_split == train, n_fit matches, vocabulary equals a refit on non-excluded train text",
                        f"fit_split=train, n_fit={obj['n_fit']} vs {len(texts)}, vocab={len(obj['vectorizer'].vocabulary_)}, "
                        f"refit vocabulary identical: {same_vocab}"))
    return rows


# --- F4 ---------------------------------------------------------------------------------

def check_f4() -> Row:
    dirs = [ROOT / "src/features", ROOT / "src/models"]
    hits, suspect = [], []
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.rglob("*.py")):
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if "speaker" in line.lower():
                    hits.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
                    code = line.split("#")[0]
                    if re.search(r"groupby|\.mean\(|\.std\(|scaler|normali[sz]", code, re.I) and "speaker" in code.lower():
                        suspect.append(hits[-1])
    scanned = ", ".join(str(d.relative_to(ROOT)) for d in dirs if d.exists())
    detail = f"grep -rni speaker {scanned}\n" + ("\n".join(hits) if hits else "(no matches)")
    if suspect:
        return row("F4", "FAIL", "no per-speaker statistics in feature or model code",
                   f"{len(suspect)} suspect lines, see details", detail)
    return row("F4", "PASS", "no per-speaker statistics in feature or model code (code-review note)",
               f"{len(hits)} mentions of 'speaker' in {scanned}, none combined with a statistic or scaler; "
               "normalisation is per feature over train rows only", detail)


# --- F5 ---------------------------------------------------------------------------------

def check_f5(feature_root: Path, mdir: str | None) -> list[Row]:
    files = cache_files(feature_root)
    if not files:
        return [row("F5", "SKIP", "cache split isolation", f"no cache files under {feature_root}")]
    bad, ok_n = [], 0
    for dname, split, path in files:
        try:
            load_cache(path, split=split)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{dname}/{split}: right split failed to load: {e}")
            continue
        for wrong in (s for s in SPLITS if s != split):
            try:
                load_cache(path, split=wrong)
                bad.append(f"{dname}/{split}: loaded with wrong split {wrong}")
            except AssertionError:
                pass
        ok_n += 1
    rows = [row("F5", "FAIL" if bad else "PASS",
                "every cache file loads with its own split and raises AssertionError for the other two",
                f"{ok_n}/{len(files)} files OK" + ("; " + "; ".join(bad) if bad else ""))]

    bad121, n121 = [], 0
    for dname, split, path in files:
        if extractor_of(dname) not in KNOWN:
            continue
        c = load_cache(path, split=split)
        ids = manifest(split, mdir)["utt_id"].to_numpy()
        n121 += 1
        if not np.array_equal(c["utt_id"], ids):
            same_set = set(c["utt_id"]) == set(ids)
            bad121.append(f"{dname}/{split}: {len(c['utt_id'])} cached vs {len(ids)} manifest rows"
                          + (", same ids, different order" if same_set else ", id sets differ"))
    if n121:
        rows.append(row("F5-1to1", "FAIL" if bad121 else "PASS",
                        "cache utt_id array equals the manifest utt_id array (rows with audio, same order)",
                        f"{n121 - len(bad121)}/{n121} caches match" + ("; " + "; ".join(bad121) if bad121 else "")))
    else:
        rows.append(row("F5-1to1", "SKIP", "cache utt_id equals manifest", "no egemaps or wavlm caches"))
    return rows


# --- F7 ---------------------------------------------------------------------------------

F7_TEXT = (
    "WavLM-base-plus is pretrained on Libri-Light (60k h), GigaSpeech (10k h) and VoxPopuli (24k h), "
    "94k hours of unlabelled English speech (Chen et al., WavLM, arXiv:2110.13900). RoBERTa-base is pretrained "
    "on BookCorpus, English Wikipedia, CC-News, OpenWebText and Stories (Liu et al., arXiv:1907.11692). "
    "Neither corpus carries MELD emotion labels, and both encoders are self-supervised, so no label leakage "
    "through pretraining. Caveat (PLAN.md 1.1): Whisper's training data may include Friends subtitles, so ASR "
    "quality on MELD can be optimistic; the gold-transcript condition bounds that effect. "
    "Pretraining sets were not checked by hashing against MELD clips; this is a documentation-level check."
)


def check_f7() -> Row:
    return row("F7", "PASS", "encoder pretraining corpora contain no MELD labels (documented)", F7_TEXT)


# --- F8 / Stage 5 shape -----------------------------------------------------------------

def check_f8(feature_root: Path) -> list[Row]:
    rows: list[Row] = []
    eg = [(d, s, p) for d, s, p in cache_files(feature_root) if extractor_of(d) == "egemaps"]
    wl = [(d, s, p) for d, s, p in cache_files(feature_root) if extractor_of(d) == "wavlm-base-plus"]
    if eg:
        bad, ev = [], []
        for d, s, p in eg:
            X = load_cache(p, split=s)["X"]
            fin = np.isfinite(X).all(axis=1)
            if X.ndim != 2 or X.shape[1] != 88 or not fin.all():
                bad.append(f"{d}/{s}: shape {X.shape}, {int((~fin).sum())} rows with NaN/inf")
            ev.append(f"{s}: {X.shape} {X.dtype}")
        rows.append(row("F8-egemaps", "FAIL" if bad else "PASS", "eGeMAPS: 88 finite values in every row",
                        "; ".join(bad) if bad else "; ".join(ev)))
    else:
        rows.append(row("F8-egemaps", "SKIP", "eGeMAPS finite", "no egemaps cache"))
    if wl:
        bad, ev, detail = [], [], []
        for d, s, p in wl:
            X = load_cache(p, split=s)["X"]
            Xf = X.astype(np.float32)
            n_bad_shape = X.ndim != 3 or X.shape[1:] != (13, 768) or X.dtype != np.float16
            n_nan = int((~np.isfinite(Xf)).any(axis=(1, 2)).sum())
            n_zero = int((~(Xf != 0).any(axis=2)).sum())
            if n_bad_shape or n_nan or n_zero:
                bad.append(f"{d}/{s}: shape {X.shape} {X.dtype}, {n_nan} rows with NaN/inf, {n_zero} all-zero (row, layer) vectors")
            norms = np.linalg.norm(Xf, axis=2)
            ev.append(f"{s}: {X.shape} {X.dtype}")
            detail.append(f"{s} L2 norm per layer (mean +- std over rows):")
            detail.append("  " + "  ".join(f"L{i}:{norms[:, i].mean():.1f}+-{norms[:, i].std():.1f}" for i in range(norms.shape[1])))
        rows.append(row("F8-wavlm", "FAIL" if bad else "PASS",
                        "WavLM: (N, 13, 768) float16, finite, no all-zero vector; per-layer L2 norm reported (sanity)",
                        "; ".join(bad) if bad else "; ".join(ev), "\n".join(detail)))
    else:
        rows.append(row("F8-wavlm", "SKIP", "WavLM finite, non-zero", "no wavlm cache"))
    return rows


# --- Stage 5 recomputation --------------------------------------------------------------

def check_recompute_egemaps(feature_root: Path, mdir: str | None, n: int, rng: np.random.Generator) -> Row:
    eg = [(d, s, p) for d, s, p in cache_files(feature_root) if extractor_of(d) == "egemaps"]
    if not eg:
        return row("S5-egemaps-recompute", "SKIP", "eGeMAPS recomputation", "no egemaps cache")
    try:
        from src.features.egemaps import make_smile
        smile = make_smile()
    except ImportError as e:
        return row("S5-egemaps-recompute", "SKIP", "eGeMAPS recomputation", f"opensmile not importable: {e}")
    tot_ok = tot = 0
    worst = 0.0
    for d, s, p in eg:
        c = load_cache(p, split=s)
        m = manifest(s, mdir).set_index("utt_id")
        idx = rng.choice(len(c["utt_id"]), size=min(n, len(c["utt_id"])), replace=False)
        for i in idx:
            new = smile.process_file(str(ROOT / m.loc[c["utt_id"][i], "wav_path"])).to_numpy(dtype=np.float32).reshape(-1)
            diff = float(np.abs(new - c["X"][i]).max())
            worst = max(worst, diff)
            tot_ok += int(np.allclose(new, c["X"][i], atol=1e-3, rtol=1e-4))
            tot += 1
    return row("S5-egemaps-recompute", "PASS" if tot_ok == tot else "FAIL",
               f"eGeMAPS recomputed from the WAV (opensmile process_file) for {n} random rows per split, allclose atol 1e-3 rtol 1e-4",
               f"{tot_ok}/{tot} rows match, max abs diff {worst:.2e}")


def check_recompute_wavlm(feature_root: Path, mdir: str | None, n: int, rng: np.random.Generator,
                          device: str | None) -> Row:
    wl = [(d, s, p) for d, s, p in cache_files(feature_root) if extractor_of(d) == "wavlm-base-plus"]
    if not wl:
        return row("S5-wavlm-recompute", "SKIP", "WavLM recomputation", "no wavlm cache")
    try:
        from src.features.ssl_embed import WavLMEmbedder, clip_samples, embed_batch, plan_batches
        emb = WavLMEmbedder(device)
    except Exception as e:  # noqa: BLE001
        return row("S5-wavlm-recompute", "SKIP", "WavLM recomputation", f"model not available: {str(e)[:150]}")
    k = min(n, 10)
    tot_ok = tot = 0
    worst = 0.0
    bs_seen = set()
    for d, s, p in wl:
        c = load_cache(p, split=s)
        max_s = float(re.search(r"mean-([\d.]+)s", c["meta"]["version"]).group(1))
        bs = int(c["batch_size"]) if "batch_size" in c else 1
        bs_seen.add(bs)
        m = manifest(s, mdir).set_index("utt_id")
        paths = [str(ROOT / m.loc[u, "wav_path"]) for u in c["utt_id"]]
        # Padding changes the output slightly, so the batch the row was computed in is rebuilt.
        batches = plan_batches([clip_samples(q, max_s) for q in paths], bs)
        batch_of = {int(i): bi for bi, idx in enumerate(batches) for i in idx}
        sample = rng.choice(len(paths), size=min(k, len(paths)), replace=False)
        for bi in sorted({batch_of[int(i)] for i in sample}):
            idx = batches[bi]
            X, nf = embed_batch([paths[j] for j in idx], emb, max_s)
            for j, row_i in enumerate(idx):
                if int(row_i) not in set(map(int, sample)):
                    continue
                a, b = X[j].astype(np.float32), c["X"][row_i].astype(np.float32)
                worst = max(worst, float(np.abs(a - b).max()))
                tot_ok += int(np.allclose(a, b, atol=2e-2, rtol=1e-2) and nf[j] == c["n_frames"][row_i])
                tot += 1
    return row("S5-wavlm-recompute", "PASS" if tot_ok == tot else "FAIL",
               f"WavLM recomputed from the WAV (device {emb.device}, original batch composition, batch size {sorted(bs_seen)}) "
               f"for {k} random rows per split, allclose atol 2e-2 rtol 1e-2 on float16, same frame count",
               f"{tot_ok}/{tot} rows match, max abs diff {worst:.2e}")


# --- Stage 5 correlation with the EDA ---------------------------------------------------

def check_corr(feature_root: Path) -> Row:
    import pandas as pd

    # The EDA writes one file per split (src/eda.py); a single audio_stats.csv is also accepted.
    proc = ROOT / "data/processed"
    stat_files = [proc / "audio_stats.csv"] if (proc / "audio_stats.csv").exists() else sorted(proc.glob("audio_stats_*.csv"))
    if not stat_files:
        return row("S5-eda-corr", "SKIP", "eGeMAPS loudness/F0 vs EDA rms_db/f0_median_hz, Pearson r > 0.8",
                   "no data/processed/audio_stats*.csv")
    eg = [(d, s, p) for d, s, p in cache_files(feature_root) if extractor_of(d) == "egemaps"]
    if not eg:
        return row("S5-eda-corr", "SKIP", "eGeMAPS vs EDA", "no egemaps cache")
    stats = pd.concat([pd.read_csv(f) for f in stat_files]).drop_duplicates("utt_id").set_index("utt_id")
    frames = []
    for d, s, p in eg:
        c = load_cache(p, split=s)
        names = [str(x) for x in c["feature_names"]]
        low = {n.lower(): i for i, n in enumerate(names)}
        li, fi = low["loudness_sma3_amean"], low["f0semitonefrom27.5hz_sma3nz_amean"]
        frames.append(pd.DataFrame({"utt_id": c["utt_id"], "loud": c["X"][:, li], "f0st": c["X"][:, fi]}))
    f = pd.concat(frames).set_index("utt_id").join(stats[["rms_db", "f0_median_hz"]], how="inner")
    if len(f) < 10:
        return row("S5-eda-corr", "SKIP", "eGeMAPS vs EDA", f"only {len(f)} utt_ids in common with {[x.name for x in stat_files]}")
    # Compare on matching scales: rms_db is a log measure, so pair it with log loudness;
    # the eGeMAPS F0 functional is in semitones (log Hz), so convert the EDA median Hz to
    # semitones from 27.5 Hz instead of exponentiating a mean of semitones into Hz.
    f = f[f["loud"] > 0]
    f["log_loud"] = np.log(f["loud"])
    v = f[(f["f0st"] != 0) & (f["f0_median_hz"] > 0) & f["f0_median_hz"].notna()].copy()
    v["f0_median_st"] = 12.0 * np.log2(v["f0_median_hz"] / 27.5)
    r_l = float(np.corrcoef(f["log_loud"], f["rms_db"])[0, 1])
    r_f = float(np.corrcoef(v["f0st"], v["f0_median_st"])[0, 1]) if len(v) >= 10 else float("nan")
    s_l = float(f["log_loud"].corr(f["rms_db"], method="spearman"))
    s_f = float(v["f0st"].corr(v["f0_median_st"], method="spearman")) if len(v) >= 10 else float("nan")
    ok = r_l > 0.8 and r_f > 0.8
    return row("S5-eda-corr", "PASS" if ok else "FAIL",
               "Pearson r > 0.8 between log eGeMAPS Loudness_sma3_amean and rms_db, and F0semitone_amean and the EDA F0 median in semitones",
               f"files {[x.name for x in stat_files]}; log loudness vs rms_db: r={r_l:.3f} (Spearman {s_l:.3f}, n={len(f)}); "
               f"F0 semitone vs semitone(f0_median_hz): r={r_f:.3f} (Spearman {s_f:.3f}, n={len(v)} voiced)")


# --- crop count -------------------------------------------------------------------------

def check_crop(feature_root: Path) -> Row:
    wl = [(d, s, p) for d, s, p in cache_files(feature_root) if extractor_of(d) == "wavlm-base-plus"]
    if not wl:
        return row("S5-crop", "SKIP", "cropped-clip count per split", "no wavlm cache")
    ev = []
    for d, s, p in wl:
        c = load_cache(p, split=s)
        cr = c["cropped"]
        ev.append(f"{s}: {int(cr.sum())}/{len(cr)} cropped ({100 * cr.mean():.1f}%) [{c['meta']['version']}]")
    return row("S5-crop", "PASS", "clips longer than the max audio length are centre-cropped; count per split logged", "; ".join(ev))


# --- output -----------------------------------------------------------------------------

def render(rows: list[Row], header: str) -> str:
    esc = lambda s: s.replace("|", "\\|").replace("\n", "<br>")  # noqa: E731
    out = [header, "", "| ID | Status | Criterion | Evidence |", "|---|---|---|---|"]
    out += [f"| {i} | {st} | {esc(c)} | {esc(e)} |" for i, st, c, e, _ in rows]
    details = [(i, d) for i, _, _, _, d in rows if d]
    if details:
        out += ["", "## Details", ""]
        for i, d in details:
            out += [f"### {i}", "", "```", d, "```", ""]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--feature-root", default=str(FEATURE_ROOT))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--out", default="reports/gate3_features.md")
    a = ap.parse_args()
    fr = Path(a.feature_root)
    fr = fr if fr.is_absolute() else ROOT / fr
    rng = np.random.default_rng(0)

    rows: list[Row] = []
    rows += check_f3(fr, a.manifest_dir)
    rows.append(check_f4())
    rows += check_f5(fr, a.manifest_dir)
    rows.append(check_f7())
    rows += check_f8(fr)
    rows.append(check_recompute_egemaps(fr, a.manifest_dir, a.n, rng))
    rows.append(check_recompute_wavlm(fr, a.manifest_dir, a.n, rng, a.device or device_name()))
    rows.append(check_corr(fr))
    rows.append(check_crop(fr))

    try:
        feature_root_disp = fr.relative_to(ROOT)
    except ValueError:
        feature_root_disp = fr
    header = (f"# Gate 3, feature checks\n\nCommand: `python scripts/check_leakage.py --manifest-dir {a.manifest_dir} "
              f"--feature-root {feature_root_disp} --n {a.n}`")
    text = render(rows, header)
    print(text)
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    n_fail = sum(r[1] == "FAIL" for r in rows)
    print(f"{n_fail} FAIL, {sum(r[1] == 'WARN' for r in rows)} WARN, {sum(r[1] == 'SKIP' for r in rows)} SKIP -> {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
