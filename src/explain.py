"""Phase 6 explainable AI (PLAN.md 3.6, Gate 6 X1-X6).

Usage: python -m src.explain --run runs/fusion_gated/<seed> [--text-run runs/text_roberta/<seed>]
           [--audio-run runs/audio_wavlm_ft/<seed>] [--egemaps-run runs/audio_egemaps_mlp/<seed>]
           [--manifest-dir DIR] [--feature-root DIR] [--device cuda|mps|cpu] [--n 200] [--k 1,3,5]
           [--out-dir reports/xai] [--limit N] [--ig-steps 50] [--batch-size 4] [--shap-nsamples 1000]
           [--n-random-layers 4] [--seed 0]

Global: SHAP on the eGeMAPS baseline (X5, compared with the E2 class statistics in
reports/eda_summary.md), WavLM layer weights of the audio run, aggregated Integrated
Gradients token attributions (top-20 tokens per predicted class). Local: token heat-maps
(Integrated Gradients on the text run's input embeddings, pad baseline), 250 ms occlusion
saliency under the spectrogram (finetune-mode audio models and fusion models with a
finetune audio branch), modality-ablation deltas of the fusion run (primary contribution
score) next to the model-internal score: the gate value of GatedFusion or the
cross-attention peakedness (``attn_t2a_peak`` / ``attn_a2t_peak``, 1 - H/Hmax over the
valid frames / tokens) of CrossAttentionFusion.
Faithfulness: deletion curves top-k vs random for tokens (X1) and segments (X2), parameter
randomisation (X3, mean |Spearman rho| < 0.3 between trained and re-initialised
attributions, signed value reported next to it), modality contribution per class (X4), case
studies (X6). Everything goes under ``--out-dir``
(``gate6_xai.md``, figures, ``cases/``). The test rows come from ``src.evaluate.test_frame``;
the unimodal runs' ``test_predictions.csv`` (from ``src.evaluate run``) select the cases.
"""
from __future__ import annotations

import argparse
import copy
import gc
import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

from src.common import EMOTIONS, LABEL2ID, NUM_CLASSES, ROOT, device_name, git_commit, load_yaml, set_seed, write_json
from src.evaluate import resolve_path, test_frame

SAMPLE_RATE = 16000
WINDOW_S = 0.25
WINDOW = int(SAMPLE_RATE * WINDOW_S)
POSITIVE_WORDS = {"great", "love", "loved", "nice", "wonderful", "perfect", "fantastic", "awesome", "good", "happy",
                  "glad", "amazing", "fun", "best", "beautiful", "lovely", "excellent", "terrific", "sweet", "cool"}
NEGATIVE_CLASSES = {"anger", "disgust", "sadness", "fear"}


# ----------------------------------------------------------------------------- helpers

def to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def free() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def randomize_(module: nn.Module, seed: int) -> None:
    """Re-initialise every parameter of ``module`` (Adebayo et al. parameter randomisation):
    matrices and tensors N(0, 0.02), normalisation scales 1, biases and other vectors 0."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for sub in module.modules():
            if isinstance(sub, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
                if sub.weight is not None:
                    sub.weight.fill_(1.0)
                if sub.bias is not None:
                    sub.bias.zero_()
                continue
            for _, p in sub.named_parameters(recurse=False):
                if p.ndim >= 2:
                    p.copy_(torch.randn(p.shape, generator=g).to(p.dtype) * 0.02)
                else:
                    p.zero_()


def spearman_rows(a: list[np.ndarray], b: list[np.ndarray]) -> dict:
    rhos = []
    for x, y in zip(a, b):
        if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            rhos.append(float(stats.spearmanr(x, y).statistic))
    pooled = (float(stats.spearmanr(np.concatenate(a), np.concatenate(b)).statistic)
              if sum(len(x) for x in a) >= 3 else float("nan"))
    return {"mean_rho": float(np.nanmean(rhos)) if rhos else float("nan"), "mean_abs_rho": float(np.nanmean(np.abs(rhos))) if rhos else float("nan"),
            "pooled_rho": pooled, "n": len(rhos)}


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df is None or len(df) == 0:
        return "_none_\n"
    return df.to_markdown(index=False, floatfmt=floatfmt) + "\n"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ----------------------------------------------------------------------------- data

def load_wav(rel: str, max_s: float) -> torch.Tensor:
    import soundfile as sf

    wav, sr = sf.read(str(ROOT / rel), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == SAMPLE_RATE, (rel, sr)
    wav = torch.from_numpy(np.ascontiguousarray(wav))
    T, M = wav.shape[-1], int(round(max_s * SAMPLE_RATE))
    if T > M:  # eval centre crop, as in MELDDataset
        s = (T - M) // 2
        wav = wav[s:s + M]
    return wav


def wav_batch(wavs: list[torch.Tensor]) -> dict:
    lens = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.long)
    T = int(lens.max())
    out = torch.zeros(len(wavs), T)
    for i, w in enumerate(wavs):
        out[i, : w.shape[-1]] = w
    return {"wav": out, "wav_len": lens, "wav_mask": torch.arange(T)[None, :] < lens[:, None]}


def read_preds(run: Path | None, fname: str = "test_predictions.csv") -> pd.DataFrame | None:
    if run is None or not (run / fname).exists():
        return None
    return pd.read_csv(run / fname)


# ----------------------------------------------------------------------------- text: integrated gradients

class TextExplainer:
    """Integrated Gradients on the input embeddings of a ``TextClassifier`` (captum), pad
    baseline at every non-special position, attribution summed over the embedding dim."""

    def __init__(self, model, tokenizer, device: str, n_steps: int = 50, max_len: int = 64, internal_batch: int = 10):
        from captum.attr import IntegratedGradients

        self.model = model.eval()
        self.tok = tokenizer
        self.device = device
        self.n_steps = n_steps
        self.max_len = max_len
        self.internal_batch = internal_batch
        self.emb = model.input_embeddings()
        self.special = set(tokenizer.all_special_ids)
        self.ig = IntegratedGradients(self._forward)

    def _forward(self, embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model.forward_from_embeds(embeds, mask)

    def encode(self, text: str) -> dict:
        enc = self.tok([text], padding=False, truncation=True, max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].to(self.device), "attention_mask": enc["attention_mask"].to(self.device)}

    @torch.no_grad()
    def probs(self, input_ids: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        return torch.softmax(self.model({"input_ids": input_ids, "attention_mask": mask}).float(), -1).cpu().numpy()

    def explain(self, text: str, target: int | None = None) -> dict:
        enc = self.encode(text)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        p = self.probs(ids, mask)[0]
        target = int(p.argmax()) if target is None else int(target)
        embeds = self.emb(ids).detach()
        base_ids = ids.clone()
        is_special = torch.tensor([[int(t) in self.special for t in ids[0].tolist()]], device=self.device)
        base_ids[~is_special] = self.tok.pad_token_id
        baseline = self.emb(base_ids).detach()
        attr = self.ig.attribute(embeds, baselines=baseline, target=target, additional_forward_args=(mask,),
                                 n_steps=self.n_steps, internal_batch_size=self.internal_batch)
        a = attr.sum(-1)[0].detach().float().cpu().numpy()
        tokens = self.tok.convert_ids_to_tokens(ids[0].tolist())
        return {"tokens": tokens, "ids": ids[0].cpu().numpy(), "attr": a, "special": is_special[0].cpu().numpy(),
                "target": target, "probs": p, "p_target": float(p[target])}

    @torch.no_grad()
    def deletion(self, ex: dict, ks: list[int], rng: np.random.Generator, n_random: int = 3) -> dict:
        """p(target) after masking the top-k attributed tokens vs k random tokens."""
        ids = torch.as_tensor(ex["ids"]).to(self.device)[None]
        mask = torch.ones_like(ids)
        cand = np.where(~ex["special"])[0]
        out = {}
        for k in ks:
            if len(cand) <= k:
                continue
            order = cand[np.argsort(-ex["attr"][cand])]
            top = order[:k]
            variants = [top] + [rng.choice(cand, size=k, replace=False) for _ in range(n_random)]
            batch = ids.repeat(len(variants), 1)
            for i, pos in enumerate(variants):
                batch[i, torch.as_tensor(pos, device=self.device)] = self.tok.mask_token_id
            p = self.probs(batch, mask.repeat(len(variants), 1))[:, ex["target"]]
            out[k] = {"top": float(p[0]), "random": float(p[1:].mean()), "full": ex["p_target"], "n_tokens": int(len(cand))}
        return out


def randomized_text_model(model, n_layers: int | None, seed: int):
    """Head and the top ``n_layers`` encoder layers re-initialised (``None`` = every encoder
    layer; the input embedding table is kept so the attributions live in the same space)."""
    m = copy.deepcopy(model)
    randomize_(m.head, seed)
    layers = m.encoder.encoder.layer
    n = len(layers) if n_layers is None else n_layers
    for layer in layers[len(layers) - n:]:
        randomize_(layer, seed)
    return m.eval()


def clean_token(t: str) -> str:
    return t.replace("Ġ", "").replace("▁", "").strip()


def global_tokens(examples: list[dict], top: int = 20, min_count: int = 1) -> pd.DataFrame:
    """Aggregated token attribution per predicted class: total and mean positive attribution."""
    rows = []
    agg: dict[tuple[int, str], list[float]] = {}
    for ex in examples:
        for t, a, s in zip(ex["tokens"], ex["attr"], ex["special"]):
            if s:
                continue
            w = clean_token(t).lower()
            if not w or not re.search(r"[a-z0-9']", w):
                continue
            agg.setdefault((ex["target"], w), []).append(float(a))
    for (c, w), vals in agg.items():
        if len(vals) >= min_count:
            rows.append({"class": EMOTIONS[c], "token": w, "total_attr": float(np.sum(vals)), "mean_attr": float(np.mean(vals)), "count": len(vals)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["class", "total_attr"], ascending=[True, False]).groupby("class").head(top).reset_index(drop=True)


def heatmap_html(tokens: list[str], attr: np.ndarray, title: str) -> str:
    m = float(np.abs(attr).max()) or 1.0
    spans = []
    for t, a in zip(tokens, attr):
        v = a / m
        color = f"rgba(220,50,50,{abs(v):.2f})" if v > 0 else f"rgba(50,80,220,{abs(v):.2f})"
        spans.append(f'<span style="background:{color};padding:2px 3px;margin:1px;border-radius:3px" title="{a:+.3f}">{html.escape(clean_token(t) or t)}</span>')
    return (f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title></head><body style='font-family:sans-serif'>"
            f"<h3>{html.escape(title)}</h3><p>red = pushes towards the predicted class, blue = against</p><p>{' '.join(spans)}</p></body></html>")


# ----------------------------------------------------------------------------- audio: occlusion

class AudioOccluder:
    """250 ms occlusion saliency for any model that reads ``batch["wav"]``: ``forward(batch)``
    returns probabilities for a batch dict; ``extra`` holds the fixed non-audio inputs
    (text ids for a fusion model) that are repeated over the occluded variants."""

    def __init__(self, model, device: str, batch_size: int = 4, extra: dict | None = None, window: int = WINDOW):
        self.model = model.eval()
        self.device = device
        self.batch_size = batch_size
        self.extra = extra or {}
        self.window = window

    @torch.no_grad()
    def probs(self, wavs: list[torch.Tensor]) -> np.ndarray:
        out = []
        for s in range(0, len(wavs), self.batch_size):
            chunk = wavs[s:s + self.batch_size]
            batch = wav_batch(chunk)
            for k, v in self.extra.items():
                batch[k] = v.repeat(len(chunk), *([1] * (v.ndim - 1))) if torch.is_tensor(v) else [v] * len(chunk)
            batch = to_device(batch, self.device)
            out.append(torch.softmax(self.model(batch).float(), -1).cpu().numpy())
        return np.concatenate(out)

    def segments(self, wav: torch.Tensor) -> list[tuple[int, int]]:
        T = wav.shape[-1]
        return [(s, min(s + self.window, T)) for s in range(0, T, self.window)]

    def occlude(self, wav: torch.Tensor, segs: list[tuple[int, int]]) -> torch.Tensor:
        w = wav.clone()
        for s, e in segs:
            w[s:e] = 0.0
        return w

    def explain(self, wav: torch.Tensor, target: int | None = None) -> dict:
        p = self.probs([wav])[0]
        target = int(p.argmax()) if target is None else int(target)
        segs = self.segments(wav)
        occ = self.probs([self.occlude(wav, [sg]) for sg in segs])[:, target]
        sal = float(p[target]) - occ
        return {"segments": segs, "saliency": sal, "target": target, "probs": p, "p_target": float(p[target]), "n_segments": len(segs)}

    def deletion(self, wav: torch.Tensor, ex: dict, ks: list[int], rng: np.random.Generator, n_random: int = 3) -> dict:
        out = {}
        n = ex["n_segments"]
        for k in ks:
            if n <= k:
                continue
            order = np.argsort(-ex["saliency"])
            variants = [order[:k]] + [rng.choice(n, size=k, replace=False) for _ in range(n_random)]
            p = self.probs([self.occlude(wav, [ex["segments"][i] for i in v]) for v in variants])[:, ex["target"]]
            out[k] = {"top": float(p[0]), "random": float(p[1:].mean()), "full": ex["p_target"], "n_segments": n}
        return out


def randomized_audio_model(model, n_layers: int | None, seed: int):
    """Head, layer logits and the top ``n_layers`` transformer layers re-initialised
    (``None`` = every transformer layer; the CNN feature encoder is kept)."""
    m = copy.deepcopy(model)
    randomize_(m.head, seed)
    with torch.no_grad():
        m.layer_logits.zero_()
    layers = m.encoder.encoder.layers
    n = len(layers) if n_layers is None else n_layers
    for layer in layers[len(layers) - n:]:
        randomize_(layer, seed)
    return m.eval()


def aopc_table(records: list[dict], ks: list[int], unit: str) -> pd.DataFrame:
    rows = []
    for k in ks:
        top = [r[k]["full"] - r[k]["top"] for r in records if k in r]
        rnd = [r[k]["full"] - r[k]["random"] for r in records if k in r]
        if not top:
            rows.append({"k": k, f"n_{unit}": 0, "AOPC_top": np.nan, "AOPC_random": np.nan, "diff": np.nan, "top > random": "n/a"})
            continue
        rows.append({"k": k, f"n_{unit}": len(top), "AOPC_top": float(np.mean(top)), "AOPC_random": float(np.mean(rnd)),
                     "diff": float(np.mean(top) - np.mean(rnd)), "frac_top_more": float(np.mean(np.array(top) > np.array(rnd))),
                     "top > random": "yes" if np.mean(top) > np.mean(rnd) else "no"})
    return pd.DataFrame(rows)


def plot_aopc(tab: pd.DataFrame, title: str, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if len(tab) and tab["AOPC_top"].notna().any():
        ax.plot(tab["k"], tab["AOPC_top"], "o-", label="top-k attributed")
        ax.plot(tab["k"], tab["AOPC_random"], "s--", label="k random")
    ax.set_xlabel("k removed")
    ax.set_ylabel("mean drop in p(predicted class)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- fusion: modality contribution

def attention_peakedness(w: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor) -> np.ndarray:
    """Per-utterance peakedness of a head-averaged attention map ``w`` [B, Q, K]:
    ``1 - H / Hmax`` with ``H`` the entropy of each valid query's distribution over the valid
    keys and ``Hmax = log(n valid keys)``, averaged over the valid queries. 0 = uniform
    attention over the other modality, 1 = every query attends to a single position. NaN
    when fewer than two valid keys (the entropy is then undefined)."""
    w = w.detach().float().cpu()
    qm = query_mask.detach().bool().cpu()
    km = key_mask.detach().bool().cpu()
    out = np.full(w.shape[0], np.nan)
    for b in range(w.shape[0]):
        k = km[b]
        q = qm[b]
        n_k = int(k.sum())
        if n_k < 2 or int(q.sum()) == 0:
            continue
        p = w[b][q][:, k].clamp_min(0)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)  # renormalise over the valid keys
        h = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
        out[b] = float((1.0 - h / np.log(n_k)).mean())
    return out


def cross_attention_scores(attn: dict[str, torch.Tensor] | None, text_mask: torch.Tensor) -> dict[str, np.ndarray] | None:
    """``attn_t2a_peak`` / ``attn_a2t_peak`` from ``CrossAttentionFusion.last_attn`` (model 8):
    text-to-audio peakedness over the valid audio frames and audio-to-text peakedness over the
    valid text tokens. Valid audio frames are the keys that received non-zero weight from any
    valid text query (padded frames are masked to exactly zero by the key_padding_mask)."""
    if not attn or "text_to_audio" not in attn or "audio_to_text" not in attn:
        return None
    w_t2a = attn["text_to_audio"].detach().float().cpu()  # [B, L, F]
    w_a2t = attn["audio_to_text"].detach().float().cpu()  # [B, F, L]
    tm = text_mask.detach().bool().cpu()  # [B, L]
    fm = (w_t2a * tm.unsqueeze(-1)).sum(1) > 0  # [B, F]
    return {"attn_t2a_peak": attention_peakedness(w_t2a, tm, fm), "attn_a2t_peak": attention_peakedness(w_a2t, fm, tm)}


@torch.no_grad()
def fusion_contribution(model, batch: dict) -> dict:
    """Full / drop-audio / drop-text probabilities, gate (GatedFusion), cross-attention
    peakedness (CrossAttentionFusion) and the ablation share."""
    p = torch.softmax(model(batch).float(), -1).cpu().numpy()
    gate = model.last_gate.flatten().cpu().numpy() if getattr(model, "last_gate", None) is not None else None
    attn = {k: v.cpu() for k, v in model.last_attn.items()} if getattr(model, "last_attn", None) else None
    attn_scores = cross_attention_scores(attn, batch["attention_mask"]) if attn else None
    p_na = torch.softmax(model(batch, drop_audio=True).float(), -1).cpu().numpy()
    p_nt = torch.softmax(model(batch, drop_text=True).float(), -1).cpu().numpy()
    pred = p.argmax(1)
    idx = np.arange(len(pred))
    d_audio = p[idx, pred] - p_na[idx, pred]
    d_text = p[idx, pred] - p_nt[idx, pred]
    share = np.clip(d_audio, 0, None) / (np.clip(d_audio, 0, None) + np.clip(d_text, 0, None) + 1e-8)
    return {"probs": p, "pred": pred, "p_drop_audio": p_na, "p_drop_text": p_nt, "delta_drop_audio": d_audio,
            "delta_drop_text": d_text, "audio_share": share, "gate": gate, "attn": attn, "attn_scores": attn_scores}


CONTRIB_LABELS = {"audio_share": "ablation audio share  d_audio/(d_audio+d_text)", "gate": "gate g (weight on audio)",
                  "attn_t2a_peak": "text->audio attention peakedness 1-H/Hmax", "attn_a2t_peak": "audio->text attention peakedness 1-H/Hmax"}


def contribution_figure(df: pd.DataFrame, path: Path, has_gate: bool, has_attn: bool = False) -> None:
    plt = _plt()
    cols = ["audio_share"] + (["gate"] if has_gate else []) + (["attn_t2a_peak", "attn_a2t_peak"] if has_attn else [])
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 3.8), squeeze=False)
    for ax, col in zip(axes[0], cols):
        data = [df.loc[df["label"] == e, col].dropna().to_numpy() for e in EMOTIONS]
        ax.boxplot(data, tick_labels=EMOTIONS, showmeans=True)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_ylabel(CONTRIB_LABELS[col])
        ax.set_title(f"{col} per true class (n={len(df)})")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- SHAP on eGeMAPS

def parse_e2(path: Path) -> pd.DataFrame | None:
    """The E2 class-statistics table of reports/eda_summary.md as a DataFrame."""
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("| emotion")), None)
    if start is None:
        return None
    rows = []
    header = [c.strip() for c in lines[start].strip("|").split("|")]
    for l in lines[start + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip() for c in l.strip("|").split("|")])
    df = pd.DataFrame(rows, columns=header)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("emotion")


FAMILIES = {"loudness": ("loudness", "rms_db_mean"), "F0": ("F0semitone", "f0_median_hz_mean"),
            "rate/duration": ("VoicedSegments", "duration_s_mean")}


def shap_egemaps(run: Path, cfg_manifest_dir: str | None, feature_root: Path | None, test_df: pd.DataFrame, n_background: int,
                 nsamples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    import shap

    from src.evaluate import load_baseline
    from src.features.egemaps import load_scaler

    model = load_baseline(run)
    cfg = model.cfg
    mdir = cfg_manifest_dir if cfg_manifest_dir is not None else cfg["data"].get("manifest_dir")
    from src.data.dataset import load_manifest

    train_df = load_manifest("train", require_audio=True, apply_exclusions=True, manifest_dir_override=mdir)
    train_df = train_df.sample(n=min(n_background, len(train_df)), random_state=seed)
    root = feature_root or resolve_path(cfg["data"].get("feature_root"))
    Xb = np.asarray(model.features(train_df, "train", root))
    Xt = np.asarray(model.features(test_df, "test", root))
    spec = model.egemaps_spec
    scaler = load_scaler(root / f"{spec['extractor']}_{spec['version']}" / "scaler.pkl")
    names = list(scaler["feature_names"])
    explainer = shap.KernelExplainer(model.estimator.predict_proba, Xb)
    sv = explainer.shap_values(Xt, nsamples=nsamples, silent=True)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)  # [N, F, C]
    sv = np.asarray(sv)
    assert sv.ndim == 3 and sv.shape[1] == len(names), sv.shape
    classes = list(model.estimator.classes_)
    rows = []
    for ci, c in enumerate(classes):
        mean_abs = np.abs(sv[:, :, ci]).mean(0)
        mean_signed = sv[:, :, ci].mean(0)
        for j in np.argsort(-mean_abs)[:5]:
            rows.append({"class": EMOTIONS[int(c)], "rank": int(np.where(np.argsort(-mean_abs) == j)[0][0]) + 1, "feature": names[j],
                         "mean_abs_shap": float(mean_abs[j]), "mean_shap": float(mean_signed[j])})
    top = pd.DataFrame(rows)
    full = pd.DataFrame({"feature": names, **{EMOTIONS[int(c)]: np.abs(sv[:, :, ci]).mean(0) for ci, c in enumerate(classes)}})
    return top, full, sv, names


def shap_vs_eda(top: pd.DataFrame, e2: pd.DataFrame | None) -> pd.DataFrame:
    """Top-5 SHAP features per class next to the class deviation of the E2 statistics from the
    n-weighted mean (rms dB, F0 Hz, duration s). ``consistent`` marks a loudness / F0 feature in
    the top 5 whose mean SHAP sign agrees with the sign of the class deviation (heuristic)."""
    rows = []
    dev = None
    if e2 is not None:
        w = e2["n"] / e2["n"].sum()
        dev = {col: e2[col] - (e2[col] * w).sum() for col in ("rms_db_mean", "f0_median_hz_mean", "duration_s_mean") if col in e2}
    for c, g in top.groupby("class", sort=False):
        feats = g.sort_values("rank")
        fam_hits = {fam: [(f, s) for f, s in zip(feats["feature"], feats["mean_shap"]) if key in f] for fam, (key, _) in FAMILIES.items()}
        row = {"class": c, "top5 (mean SHAP sign)": ", ".join(f"{f} ({'+' if s > 0 else '-'})" for f, s in zip(feats["feature"], feats["mean_shap"])),
               "families in top5": ", ".join(f for f, h in fam_hits.items() if h) or "-"}
        if dev is not None and c in e2.index:
            row["EDA d_rms_dB"] = float(dev["rms_db_mean"][c]) if "rms_db_mean" in dev else np.nan
            row["EDA d_F0_Hz"] = float(dev["f0_median_hz_mean"][c]) if "f0_median_hz_mean" in dev else np.nan
            row["EDA d_dur_s"] = float(dev["duration_s_mean"][c]) if "duration_s_mean" in dev else np.nan
            cons = []
            for fam, (key, col) in FAMILIES.items():
                if fam_hits[fam] and col in dev and col != "duration_s_mean":
                    agree = any(np.sign(s) == np.sign(dev[col][c]) for _, s in fam_hits[fam])
                    cons.append(f"{fam}:{'yes' if agree else 'no'}")
            row["consistent"] = ", ".join(cons) or "n/a"
        else:
            row["consistent"] = "no E2 table"
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures

def layer_weight_figure(w: np.ndarray, title: str, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(w)), w)
    ax.set_xticks(range(len(w)))
    ax.set_xlabel("WavLM layer (0 = CNN output)")
    ax.set_ylabel("softmax weight")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def token_strip(ax, tokens: list[str], attr: np.ndarray, title: str) -> None:
    plt = _plt()
    m = float(np.abs(attr).max()) or 1.0
    ax.bar(range(len(tokens)), attr / m, color=["#d9534f" if a > 0 else "#4a6fd6" for a in attr])
    ax.set_xticks(range(len(tokens)), [clean_token(t) or t for t in tokens], rotation=60, ha="right", fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("IG (norm.)")
    ax.set_title(title, fontsize=9)


def spectrogram_saliency(ax_spec, ax_sal, wav: torch.Tensor, ex: dict, title: str) -> None:
    x = wav.numpy()
    ax_spec.specgram(x, Fs=SAMPLE_RATE, NFFT=512, noverlap=256, cmap="magma")
    ax_spec.set_ylabel("Hz")
    ax_spec.set_title(title, fontsize=9)
    centers = [(s + e) / 2 / SAMPLE_RATE for s, e in ex["segments"]]
    widths = [(e - s) / SAMPLE_RATE for s, e in ex["segments"]]
    ax_sal.bar(centers, ex["saliency"], width=widths, align="center", color=["#d9534f" if v > 0 else "#4a6fd6" for v in ex["saliency"]])
    ax_sal.axhline(0, color="k", lw=0.5)
    ax_sal.set_xlim(0, len(x) / SAMPLE_RATE)
    ax_sal.set_xlabel("time (s)")
    ax_sal.set_ylabel("drop in p(pred)")


def case_figure(path: Path, title: str, text_ex: dict | None, wav: torch.Tensor | None, audio_ex: dict | None, text_title: str, audio_title: str) -> None:
    plt = _plt()
    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.6, 0.8])
    fig.suptitle(title, fontsize=9)
    ax_t = fig.add_subplot(gs[0])
    if text_ex is not None:
        token_strip(ax_t, text_ex["tokens"], text_ex["attr"], text_title)
    else:
        ax_t.text(0.5, 0.5, "no text explanation", ha="center")
        ax_t.axis("off")
    ax_s = fig.add_subplot(gs[1])
    ax_b = fig.add_subplot(gs[2], sharex=ax_s)
    if wav is not None and audio_ex is not None:
        spectrogram_saliency(ax_s, ax_b, wav, audio_ex, audio_title)
    else:
        ax_s.text(0.5, 0.5, "no audio explanation (cached-mode audio branch)", ha="center")
        ax_s.axis("off")
        ax_b.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- cases

def select_cases(df: pd.DataFrame, fusion_preds: pd.DataFrame | None, text_preds: pd.DataFrame | None,
                 audio_preds: pd.DataFrame | None, seed: int, n_each: int = 5) -> pd.DataFrame:
    """X6: 5 utterances where audio-only and text-only disagree and fusion is right, 5 where
    fusion is wrong, 1 sarcasm candidate."""
    if fusion_preds is None:
        return pd.DataFrame()
    m = df[["utt_id", "label", "gold_text", "text"]].merge(fusion_preds[["utt_id", "pred"] + [f"p_{c}" for c in range(NUM_CLASSES)]], on="utt_id")
    m = m.rename(columns={"pred": "fusion_pred"})
    for name, p in (("text_pred", text_preds), ("audio_pred", audio_preds)):
        m[name] = m["utt_id"].map(p.set_index("utt_id")["pred"]) if p is not None else np.nan
    rng = np.random.default_rng(seed)
    have_uni = m["text_pred"].notna().all() and m["audio_pred"].notna().all()
    right = m[m["fusion_pred"] == m["label"]]
    if have_uni:
        right = right[right["text_pred"] != right["audio_pred"]]
    wrong = m[m["fusion_pred"] != m["label"]]
    pos = m["gold_text"].fillna("").str.lower().str.findall(r"[a-z']+").map(lambda ws: any(w in POSITIVE_WORDS for w in ws))
    neg_label = m["label"].map(lambda i: EMOTIONS[int(i)] in NEGATIVE_CLASSES)
    sarc = m[pos & neg_label]
    if have_uni:
        joy_vs_neg = (m["text_pred"] == LABEL2ID["joy"]) & m["fusion_pred"].isin([LABEL2ID["anger"], LABEL2ID["disgust"]])
        sarc = pd.concat([sarc, m[joy_vs_neg]]).drop_duplicates("utt_id")
    parts = []
    for cat, pool, k in (("disagree_fusion_right" if have_uni else "fusion_right", right, n_each), ("fusion_wrong", wrong, n_each), ("sarcasm", sarc, 1)):
        pool = pool[~pool["utt_id"].isin(pd.concat(parts)["utt_id"])] if parts else pool
        if len(pool):
            take = pool.sample(n=min(k, len(pool)), random_state=int(rng.integers(0, 2**31 - 1))).copy()
            take["category"] = cat
            parts.append(take)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).reset_index(drop=True)
    out["unimodal_preds_available"] = have_uni
    return out


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="fusion run directory runs/fusion_gated/<seed>")
    ap.add_argument("--text-run")
    ap.add_argument("--audio-run")
    ap.add_argument("--egemaps-run")
    ap.add_argument("--manifest-dir")
    ap.add_argument("--feature-root")
    ap.add_argument("--device")
    ap.add_argument("--n", type=int, default=200, help="test utterances for the faithfulness tests")
    ap.add_argument("--k", default="1,3,5")
    ap.add_argument("--out-dir", default="reports/xai")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ig-steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--shap-nsamples", type=int, default=1000)
    ap.add_argument("--shap-background", type=int, default=100)
    ap.add_argument("--n-random-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    set_seed(args.seed)
    device = args.device or device_name()
    ks = [int(k) for k in args.k.split(",")]
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases").mkdir(exist_ok=True)
    fusion_run = resolve_path(args.run)
    text_run = resolve_path(args.text_run)
    audio_run = resolve_path(args.audio_run)
    egemaps_run = resolve_path(args.egemaps_run)
    feature_root = resolve_path(args.feature_root)
    fcfg = load_yaml(fusion_run / "config.yaml")
    if args.manifest_dir:
        fcfg["data"]["manifest_dir"] = args.manifest_dir
    if args.feature_root:
        fcfg["data"]["feature_root"] = args.feature_root
    max_audio_s = float(fcfg.get("train", {}).get("max_audio_s", 10.0))
    max_len = int(fcfg.get("train", {}).get("max_text_tokens", 64))
    df = test_frame(fcfg, args.manifest_dir, limit=args.limit)
    sub = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    cases = select_cases(df, read_preds(fusion_run), read_preds(text_run), read_preds(audio_run), args.seed)
    case_ids = cases["utt_id"].tolist() if len(cases) else []
    work = pd.concat([sub, df[df["utt_id"].isin(case_ids) & ~df["utt_id"].isin(sub["utt_id"])]]).reset_index(drop=True)
    print(f"test rows {len(df)}, faithfulness subset {len(sub)}, cases {len(case_ids)} -> {len(work)} utterances to explain; device {device}")
    gate: list[tuple[str, str, str]] = []
    report: list[str] = [f"# Phase 6 XAI ({fusion_run.relative_to(ROOT) if fusion_run.is_relative_to(ROOT) else fusion_run})\n",
                         f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}, git {git_commit()[:8]}, device {device}, "
                         f"n={len(sub)} faithfulness utterances, k={ks}, IG steps {args.ig_steps}.\n"]
    timings: dict[str, float] = {}
    numbers: dict[str, Any] = {}
    text_cases: dict[str, dict] = {}
    audio_cases: dict[str, dict] = {}
    wavs: dict[str, torch.Tensor] = {}

    # ---- text stage: IG, X1, X3, global tokens
    if text_run is not None and (text_run / "best.pt").exists():
        t0 = time.time()
        from transformers import AutoTokenizer

        from src.models import load_run, tokenizer_name

        tmodel, tcfg = load_run(text_run, device)
        tok = AutoTokenizer.from_pretrained(tokenizer_name(tcfg))
        texp = TextExplainer(tmodel, tok, device, n_steps=args.ig_steps, max_len=max_len)
        rng = np.random.default_rng(args.seed)
        examples, dels = [], []
        for i, r in work.iterrows():
            ex = texp.explain(str(r["text"]))
            ex["utt_id"] = r["utt_id"]
            ex["label"] = int(r["label"])
            examples.append(ex)
            if r["utt_id"] in set(sub["utt_id"]):
                dels.append(texp.deletion(ex, ks, rng))
            if r["utt_id"] in case_ids:
                text_cases[r["utt_id"]] = ex
        x1 = aopc_table(dels, ks, "utterances")
        plot_aopc(x1, "X1 text deletion (IG top-k vs random tokens)", out_dir / "x1_text_deletion.png")
        x1_pass = len(x1) and x1["AOPC_top"].notna().all() and (x1["top > random"] == "yes").all()
        gate.append(("X1", "PASS" if x1_pass else ("FAIL" if len(x1) and x1["AOPC_top"].notna().any() else "SKIP"),
                     "; ".join(f"k={int(r['k'])}: top {r['AOPC_top']:.4f} vs random {r['AOPC_random']:.4f} (n={int(r['n_utterances'])})" for _, r in x1.iterrows())))
        numbers["x1"] = x1.to_dict("records")
        report += ["## X1 text faithfulness (deletion, AOPC over the subset)\n", _md(x1, ".4f"), "![x1](x1_text_deletion.png)\n"]
        # X3 text: top-N layers (cascading) and every encoder layer re-initialised
        sub_ids = set(sub["utt_id"])
        trained_attr = [ex["attr"][~ex["special"]] for ex in examples if ex["utt_id"] in sub_ids]
        x3t = {"n_top_layers": args.n_random_layers}
        for level, n_layers in (("top", args.n_random_layers), ("all", None)):
            rmodel = randomized_text_model(tmodel, n_layers, args.seed)
            rexp = TextExplainer(rmodel, tok, device, n_steps=args.ig_steps, max_len=max_len)
            random_attr = []
            for ex in examples:
                if ex["utt_id"] in sub_ids:
                    random_attr.append(rexp.explain(str(work.loc[work["utt_id"] == ex["utt_id"], "text"].iloc[0]), target=ex["target"])["attr"][~ex["special"]])
            x3t[level] = spearman_rows(trained_attr, random_attr)
            del rmodel, rexp
            free()
        numbers["x3_text"] = x3t
        # global tokens
        gt = global_tokens(examples)
        gt.to_csv(out_dir / "global_tokens_ig.csv", index=False)
        report += ["## Global: top-20 tokens per predicted class (aggregated IG, positive = towards the class)\n", _md(gt, ".4f")]
        # local heat-maps (cases)
        for uid, ex in text_cases.items():
            (out_dir / "cases" / f"{uid}_text.html").write_text(heatmap_html(ex["tokens"], ex["attr"], f"{uid}: IG towards {EMOTIONS[ex['target']]} (p={ex['p_target']:.2f})"))
        timings["text_stage_s"] = time.time() - t0
        print(f"text stage: {len(examples)} IG explanations, X1 {x1.to_dict('records')}, X3 text rho {x3t} ({timings['text_stage_s']:.1f}s)")
        del tmodel, texp, examples
        free()
    else:
        gate.append(("X1", "SKIP", "no --text-run with best.pt"))
        numbers["x3_text"] = None
        report += ["## X1: skipped (no text run)\n"]

    # ---- audio stage: layer weights, occlusion, X2, X3 audio
    from src.models import load_run

    for _, r in work.iterrows():
        wavs[r["utt_id"]] = load_wav(r["wav_path"], max_audio_s)
    audio_occ_model_desc = None
    if audio_run is not None and (audio_run / "best.pt").exists():
        t0 = time.time()
        amodel, acfg = load_run(audio_run, device)
        lw = amodel.layer_weights().cpu().numpy() if hasattr(amodel, "layer_weights") else None
        if lw is not None:
            layer_weight_figure(lw, f"{audio_run.parent.name}: learned layer weights", out_dir / "layer_weights.png")
            report += ["## Global: WavLM layer weights\n", "| layer | " + " | ".join(str(i) for i in range(len(lw))) + " |",
                       "|---|" + "---|" * len(lw), "| weight | " + " | ".join(f"{v:.3f}" for v in lw) + " |", "", "![layers](layer_weights.png)\n"]
            numbers["layer_weights"] = lw.tolist()
        if getattr(amodel, "mode", "cached") == "finetune":
            occ = AudioOccluder(amodel, device, args.batch_size)
            rng = np.random.default_rng(args.seed)
            sub_ids = set(sub["utt_id"])
            a_examples, a_dels = {}, []
            for uid in work["utt_id"]:
                ex = occ.explain(wavs[uid])
                a_examples[uid] = ex
                if uid in sub_ids:
                    a_dels.append(occ.deletion(wavs[uid], ex, ks, rng))
            x2 = aopc_table(a_dels, ks, "utterances")
            plot_aopc(x2, "X2 audio deletion (occlusion top-k vs random 250 ms segments)", out_dir / "x2_audio_deletion.png")
            x2_pass = len(x2) and x2["AOPC_top"].notna().all() and (x2["top > random"] == "yes").all()
            gate.append(("X2", "PASS" if x2_pass else ("FAIL" if len(x2) and x2["AOPC_top"].notna().any() else "SKIP"),
                         "; ".join(f"k={int(r['k'])}: top {r['AOPC_top']:.4f} vs random {r['AOPC_random']:.4f} (n={int(r['n_utterances'])})" for _, r in x2.iterrows())))
            numbers["x2"] = x2.to_dict("records")
            report += ["## X2 audio faithfulness (occlusion deletion, AOPC over the subset)\n", _md(x2, ".4f"), "![x2](x2_audio_deletion.png)\n"]
            x3a = {"n_top_layers": args.n_random_layers}
            tr_s = [a_examples[uid]["saliency"] for uid in sub["utt_id"]]
            for level, n_layers in (("top", args.n_random_layers), ("all", None)):
                rmodel = randomized_audio_model(amodel, n_layers, args.seed)
                rocc = AudioOccluder(rmodel, device, args.batch_size)
                rd_s = [rocc.explain(wavs[uid], target=a_examples[uid]["target"])["saliency"] for uid in sub["utt_id"]]
                x3a[level] = spearman_rows(tr_s, rd_s)
                del rmodel, rocc
                free()
            numbers["x3_audio"] = x3a
            audio_occ_model_desc = f"audio run {audio_run.parent.name}"
            for uid in case_ids:
                if uid in a_examples:
                    audio_cases[uid] = a_examples[uid]
            timings["audio_stage_s"] = time.time() - t0
            print(f"audio stage: {len(a_examples)} occlusion maps, X2 {x2.to_dict('records')}, X3 audio rho {x3a} ({timings['audio_stage_s']:.1f}s)")
        else:
            gate.append(("X2", "SKIP", f"audio run {audio_run.parent.name} is cached-mode (no waveform path); occlusion needs audio.mode: finetune"))
            numbers["x3_audio"] = None
            report += ["## X2: skipped (cached-mode audio run has no waveform input)\n"]
        del amodel
        free()
    else:
        gate.append(("X2", "SKIP", "no --audio-run with best.pt"))
        numbers["x3_audio"] = None
        report += ["## X2: skipped (no audio run)\n"]

    # X3 verdict
    x3t, x3a = numbers.get("x3_text"), numbers.get("x3_audio")

    def _x3(name: str, d: dict) -> str:
        return (f"{name}: mean |Spearman rho| {d['all']['mean_abs_rho']:.3f} (signed mean rho {d['all']['mean_rho']:+.3f}, pooled "
                f"{d['all']['pooled_rho']:+.3f}, n={d['all']['n']}) with every encoder layer + head re-initialised; "
                f"top-{d['n_top_layers']} layers only: |rho| {d['top']['mean_abs_rho']:.3f} (signed {d['top']['mean_rho']:+.3f})")

    if x3t is not None:
        ok = x3t["all"]["mean_abs_rho"] < 0.3
        ev = _x3("text IG", x3t)
        if x3a is not None:
            ok = ok and x3a["all"]["mean_abs_rho"] < 0.3
            ev += "; " + _x3("audio occlusion", x3a)
        gate.append(("X3", "PASS" if ok else "FAIL", ev + "; criterion: mean |rho| < 0.3 for the fully re-initialised encoder "
                     "(absolute value, so an anti-correlated random model cannot pass by sign)"))
    else:
        gate.append(("X3", "SKIP", "needs the text run"))
    report += ["## X3 parameter randomisation\n", f"{gate[-1][2]}\n"]

    # ---- fusion stage: contribution scores (X4), fusion occlusion for the cases
    t0 = time.time()
    fmodel, _ = load_run(fusion_run, device)
    tok = None
    from src.models import tokenizer_name
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name(fcfg))
    from src.data.torch_dataset import MELDDataset, collate

    ds = MELDDataset(work, "test", fcfg, train=False, feature_root=feature_root)
    rows = []
    for s in range(0, len(ds), args.batch_size):
        batch = to_device(collate([ds[i] for i in range(s, min(s + args.batch_size, len(ds)))], tok, max_len), device)
        c = fusion_contribution(fmodel, batch)
        for j, uid in enumerate(batch["utt_id"]):
            rows.append({"utt_id": uid, "label": EMOTIONS[int(batch["label"][j])], "pred": EMOTIONS[int(c["pred"][j])],
                         "p_pred": float(c["probs"][j, c["pred"][j]]), "delta_drop_audio": float(c["delta_drop_audio"][j]),
                         "delta_drop_text": float(c["delta_drop_text"][j]), "audio_share": float(c["audio_share"][j]),
                         "gate": float(c["gate"][j]) if c["gate"] is not None else np.nan,
                         "attn_t2a_peak": float(c["attn_scores"]["attn_t2a_peak"][j]) if c["attn_scores"] else np.nan,
                         "attn_a2t_peak": float(c["attn_scores"]["attn_a2t_peak"][j]) if c["attn_scores"] else np.nan})
    contrib = pd.DataFrame(rows)
    contrib.to_csv(out_dir / "modality_contribution.csv", index=False)
    has_gate = contrib["gate"].notna().any()
    has_attn = contrib["attn_t2a_peak"].notna().any()
    csub = contrib[contrib["utt_id"].isin(set(sub["utt_id"]))]
    contribution_figure(csub, out_dir / "x4_modality_contribution.png", has_gate, has_attn)
    aggs = {"n": ("utt_id", "count"), "audio_share": ("audio_share", "mean"), "d_audio": ("delta_drop_audio", "mean"), "d_text": ("delta_drop_text", "mean")}
    if has_gate:
        aggs["gate"] = ("gate", "mean")
    if has_attn:
        aggs["attn_t2a_peak"] = ("attn_t2a_peak", "mean")
        aggs["attn_a2t_peak"] = ("attn_a2t_peak", "mean")
    per_class = csub.groupby("label").agg(**aggs).reindex(EMOTIONS).reset_index()
    per_class["driven_by"] = np.where(per_class["audio_share"] > 0.5, "audio", "text")
    per_class.loc[per_class["n"].isna() | (per_class["n"] == 0), "driven_by"] = "-"
    audio_driven = per_class.loc[per_class["driven_by"] == "audio", "label"].tolist()
    text_driven = per_class.loc[per_class["driven_by"] == "text", "label"].tolist()
    score_desc = ("gate g of GatedFusion (model 7)" if has_gate else
                  "cross-attention peakedness of CrossAttentionFusion (model 8)" if has_attn else "no model-internal score")
    gate.append(("X4", "PASS", f"{len(csub)} utterances; primary score = modality-ablation delta (audio share; driven_by uses it); "
                 f"audio-driven: {audio_driven or 'none'}; text-driven: {text_driven or 'none'}; secondary score = {score_desc}"
                 + (f", mean gate {csub['gate'].mean():.3f}" if has_gate else "")
                 + (f", mean attn_t2a_peak {csub['attn_t2a_peak'].mean():.3f}, mean attn_a2t_peak {csub['attn_a2t_peak'].mean():.3f}" if has_attn else "")))
    report += ["## X4 modality contribution per true class\n",
               "Primary contribution score: the modality-ablation delta (`d_audio` / `d_text` = drop in p(pred) when the modality is zeroed; "
               "`audio_share = d_audio / (d_audio + d_text)`, `driven_by` thresholds it at 0.5). Secondary, model-internal score: `gate` = GatedFusion g "
               "(weight on audio, model 7) or, for CrossAttentionFusion (model 8), `attn_t2a_peak` / `attn_a2t_peak` = entropy-normalised peakedness "
               "`1 - H/Hmax` of the text->audio (audio->text) attention over the valid audio frames (text tokens), averaged over the valid queries: "
               "0 = uniform attention over the other modality, 1 = each query attends to one position. Peakedness says how selectively the other "
               "modality is read, not how much it changes the prediction, which is why the ablation delta stays primary.\n",
               _md(per_class), "![x4](x4_modality_contribution.png)\n"]
    numbers["x4_per_class"] = per_class.to_dict("records")
    fusion_audio_ft = fcfg["model"].get("audio", {}).get("mode") == "finetune"
    fusion_cases: dict[str, dict] = {}
    if fusion_audio_ft and case_ids:
        for i, r in work[work["utt_id"].isin(case_ids)].iterrows():
            item = ds[int(i)]
            b = collate([item], tok, max_len)
            occ = AudioOccluder(fmodel, device, args.batch_size, extra={"input_ids": b["input_ids"], "attention_mask": b["attention_mask"], "utt_id": r["utt_id"], "label": b["label"]})
            fusion_cases[r["utt_id"]] = occ.explain(wavs[r["utt_id"]])
    timings["fusion_stage_s"] = time.time() - t0
    print(f"fusion stage: {len(contrib)} contribution rows, {len(fusion_cases)} fusion occlusion maps ({timings['fusion_stage_s']:.1f}s)")
    del fmodel
    free()

    # ---- SHAP on the eGeMAPS baseline (X5)
    if egemaps_run is not None and (egemaps_run / "model.pkl").exists():
        t0 = time.time()
        try:
            top, full, sv, names = shap_egemaps(egemaps_run, args.manifest_dir, feature_root, sub, args.shap_background, args.shap_nsamples, args.seed)
            full.to_csv(out_dir / "shap_egemaps_mean_abs.csv", index=False)
            e2 = parse_e2(ROOT / "reports/eda_summary.md")
            cmp = shap_vs_eda(top, e2)
            cmp.to_csv(out_dir / "x5_shap_vs_eda.csv", index=False)
            plt = _plt()
            fig, ax = plt.subplots(figsize=(8, 4))
            mean_abs = full.set_index("feature").mean(axis=1).sort_values(ascending=False).head(15)
            ax.barh(mean_abs.index[::-1], mean_abs.values[::-1])
            ax.set_xlabel("mean |SHAP| over classes")
            ax.set_title(f"eGeMAPS SHAP ({egemaps_run.parent.name}, {len(sub)} test rows, {args.shap_background} background)")
            fig.tight_layout()
            fig.savefig(out_dir / "x5_shap_egemaps.png", dpi=120)
            plt.close(fig)
            gate.append(("X5", "PASS" if e2 is not None else "PARTIAL", f"top-5 per class for {top['class'].nunique()} classes; "
                         + ("compared with E2 stats: " + "; ".join(f"{r['class']}: {r['consistent']}" for _, r in cmp.iterrows()) if e2 is not None else "reports/eda_summary.md E2 table not found")))
            report += ["## X5 SHAP on eGeMAPS: top-5 features per class vs EDA class statistics (E2)\n", _md(cmp), "Top-5 detail:\n", _md(top, ".4f"), "![x5](x5_shap_egemaps.png)\n"]
            numbers["x5"] = cmp.to_dict("records")
        except Exception as e:  # noqa: BLE001
            gate.append(("X5", "FAIL", f"SHAP failed: {type(e).__name__}: {e}"))
            report += [f"## X5: failed ({e})\n"]
        timings["shap_s"] = time.time() - t0
        print(f"shap stage: {gate[-1]} ({timings['shap_s']:.1f}s)")
    else:
        gate.append(("X5", "SKIP", "no --egemaps-run with model.pkl"))
        report += ["## X5: skipped (no eGeMAPS run)\n"]

    # ---- case studies (X6)
    t0 = time.time()
    case_rows = []
    for _, r in cases.iterrows():
        uid = r["utt_id"]
        tex = text_cases.get(uid)
        aex = fusion_cases.get(uid) or audio_cases.get(uid)
        asrc = "fusion" if uid in fusion_cases else ("audio run" if uid in audio_cases else "none")
        crow = contrib[contrib["utt_id"] == uid].iloc[0] if (contrib["utt_id"] == uid).any() else None
        title = (f"{uid} [{r['category']}] true={EMOTIONS[int(r['label'])]} fusion={EMOTIONS[int(r['fusion_pred'])]} "
                 f"text={EMOTIONS[int(r['text_pred'])] if pd.notna(r['text_pred']) else '-'} audio={EMOTIONS[int(r['audio_pred'])] if pd.notna(r['audio_pred']) else '-'}"
                 + (f" | gate={crow['gate']:.2f} d_audio={crow['delta_drop_audio']:+.2f} d_text={crow['delta_drop_text']:+.2f}" if crow is not None else "")
                 + (f" attn t2a/a2t={crow['attn_t2a_peak']:.2f}/{crow['attn_a2t_peak']:.2f}" if crow is not None and pd.notna(crow["attn_t2a_peak"]) else "")
                 + f"\ngold: {str(r['gold_text'])[:110]}\nasr : {str(r['text'])[:110] if str(r['text']) != str(r['gold_text']) else '(same as gold)'}")
        path = out_dir / "cases" / f"{uid}.png"
        case_figure(path, title, tex, wavs.get(uid), aex,
                    f"text IG ({text_run.parent.name if text_run else '-'}) towards {EMOTIONS[tex['target']] if tex else '-'}",
                    f"audio occlusion ({asrc}) towards {EMOTIONS[aex['target']] if aex else '-'}")
        case_rows.append({"category": r["category"], "utt_id": uid, "true": EMOTIONS[int(r["label"])], "fusion": EMOTIONS[int(r["fusion_pred"])],
                          "text_only": EMOTIONS[int(r["text_pred"])] if pd.notna(r["text_pred"]) else "-",
                          "audio_only": EMOTIONS[int(r["audio_pred"])] if pd.notna(r["audio_pred"]) else "-",
                          "gate": f"{crow['gate']:.2f}" if crow is not None and pd.notna(crow["gate"]) else "-",
                          "attn_peak t2a/a2t": f"{crow['attn_t2a_peak']:.2f}/{crow['attn_a2t_peak']:.2f}" if crow is not None and pd.notna(crow["attn_t2a_peak"]) else "-",
                          "d_audio": f"{crow['delta_drop_audio']:+.2f}" if crow is not None else "-", "d_text": f"{crow['delta_drop_text']:+.2f}" if crow is not None else "-",
                          "text_expl": tex is not None, "audio_expl": aex is not None, "gold_text": str(r["gold_text"]), "figure": f"cases/{uid}.png"})
    cdf = pd.DataFrame(case_rows)
    both = int((cdf["text_expl"] & cdf["audio_expl"]).sum()) if len(cdf) else 0
    (out_dir / "cases" / "cases.md").write_text("# Case studies (X6)\n\n" + _md(cdf) + "\n" + "\n".join(f"## {r['utt_id']} ({r['category']})\n\n![{r['utt_id']}]({r['utt_id']}.png)\n" for r in case_rows))
    cats = cdf["category"].value_counts().to_dict() if len(cdf) else {}
    gate.append(("X6", "PASS" if both >= 11 else ("PARTIAL" if both else "FAIL"),
                 f"{len(cdf)} cases ({cats}), {both} with both modality explanations; need 11 (5 disagree-and-fusion-right, 5 fusion-wrong, 1 sarcasm)"))
    report += ["## X6 case studies\n", _md(cdf.drop(columns=["gold_text"]) if len(cdf) else cdf), "See `cases/cases.md`.\n"]
    timings["cases_s"] = time.time() - t0

    # ---- gate table
    names = {"X1": "Faithfulness, text", "X2": "Faithfulness, audio", "X3": "Sanity, parameter randomisation", "X4": "Modality contribution",
             "X5": "Global audio features match phonetics", "X6": "Case studies"}
    order = {k: i for i, k in enumerate(names)}
    gate.sort(key=lambda g: order[g[0]])
    gate_md = ["# Gate 6 (XAI)\n", f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from `{fusion_run}` (n={len(sub)}, k={ks}).\n",
               "| ID | Check | Result | Evidence |", "|---|---|---|---|"] + [f"| {g} | {names[g]} | {s} | {e} |" for g, s, e in gate]
    (out_dir / "gate6_xai.md").write_text("\n".join(gate_md) + "\n")
    timings["total_s"] = time.time() - t_start
    report += ["## Gate 6\n", "\n".join(gate_md[3:]) + "\n", f"Timings (s): {json.dumps({k: round(v, 1) for k, v in timings.items()})}\n"]
    (out_dir / "xai_report.md").write_text("\n".join(report))
    write_json(out_dir / "xai_numbers.json", {"numbers": numbers, "timings": timings, "args": vars(args), "git_commit": git_commit()})
    print("\n".join(gate_md[3:]))
    print(f"wrote {out_dir / 'gate6_xai.md'} ({timings['total_s']:.1f}s)")


if __name__ == "__main__":
    main()
