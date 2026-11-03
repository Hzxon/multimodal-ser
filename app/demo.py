"""Gradio demo: speech in, transcript, emotion probabilities and explanations out (PLAN.md Phase 7).

Usage: python app/demo.py --run runs/fusion_gated/13 [--asr-tag whisper-small] [--device cpu|mps|cuda]
       [--port 7860] [--share]

Pipeline: microphone or file -> 16 kHz mono -> Whisper transcript (configs/asr.yaml decoding, loader and
decoding code reused from src/asr/transcribe.py) -> fusion model from ``--run`` -> emotion
probabilities. If the fusion checkpoint is missing, the demo falls back to a text-only run
(``text_roberta`` of the same seed under the same runs root) and says so in the UI.

Explanations (``explain`` below; the demo has its own minimal helpers until src/explain.py is the shared
implementation):
- token heat-map: Integrated Gradients (captum) over the RoBERTa input embeddings, audio branch fixed,
  target = predicted class probability, baseline = pad embedding with the special tokens kept. Supported
  for ``fusion_gated`` and ``text_roberta`` runs.
- audio saliency: drop in the predicted-class probability when a 250 ms window is replaced by silence
  (finetune-mode audio only; cached mode shows a note).
- modality contribution: the gate value ``g`` of ``fusion_gated`` (weight of the audio branch in
  ``[h_a; h_t; g*h_a; (1-g)*h_t]``) and the probability change when each modality is zeroed.
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asr.transcribe import Transcriber, resolve_model  # noqa: E402
from src.common import EMOTIONS, load_yaml, normalize_text  # noqa: E402
from src.models import load_run, tokenizer_name  # noqa: E402

SR = 16000
WINDOW_S = 0.25
IG_STEPS = 24


# ----------------------------------------------------------------------------- loading

def resolve_run(run: str | Path) -> tuple[Path, str]:
    """(run dir to load, note for the UI). Falls back to the text-only run of the same seed."""
    p = Path(run)
    p = p if p.is_absolute() else ROOT / p
    if (p / "best.pt").exists():
        return p, ""
    fallback = p.parent.parent / "text_roberta" / p.name
    if (fallback / "best.pt").exists():
        return fallback, f"Fusion checkpoint {p} not found; using the text-only model {fallback}. The audio is not used."
    raise FileNotFoundError(f"neither {p / 'best.pt'} nor {fallback / 'best.pt'} exists")


def load_state(run: str, asr_tag: str | None, device: str) -> dict:
    asr_cfg = load_yaml("configs/asr.yaml")
    tag, model_id = resolve_model(asr_cfg, asr_tag)
    run_dir, note = resolve_run(run)
    model, cfg = load_run(run_dir, device)
    from transformers import AutoTokenizer

    state = {
        "device": device, "model": model, "cfg": cfg, "run_dir": run_dir, "note": note,
        "kind": cfg["model"]["type"], "asr_tag": tag,
        "tokenizer": AutoTokenizer.from_pretrained(tokenizer_name(cfg)),
        "max_text_tokens": int(cfg["train"].get("max_text_tokens", 64)),
        "max_audio_s": float(cfg["train"].get("max_audio_s", 10.0)),
        "asr": Transcriber(asr_cfg, model_id, device),
        "embedder": None,
    }
    audio_cfg = cfg["model"].get("audio") if state["kind"].startswith("fusion") else None
    state["audio_mode"] = audio_cfg.get("mode", "cached") if audio_cfg else None
    if state["audio_mode"] == "cached":
        from src.features.ssl_embed import WavLMEmbedder

        state["embedder"] = WavLMEmbedder(device)
    return state


def load_audio(path: str) -> np.ndarray:
    """16 kHz mono float32 from any file soundfile or librosa can read."""
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


# ----------------------------------------------------------------------------- model inputs

def make_batch(state: dict, wav: np.ndarray, text: str, wav_override: torch.Tensor | None = None) -> dict:
    """Batch of one for ``model(batch)``; ``wav`` is cropped to the training length (centre crop)."""
    dev = state["device"]
    enc = state["tokenizer"]([text], padding=True, truncation=True, max_length=state["max_text_tokens"], return_tensors="pt")
    batch = {"input_ids": enc["input_ids"].to(dev), "attention_mask": enc["attention_mask"].to(dev)}
    max_n = int(round(state["max_audio_s"] * SR))
    if len(wav) > max_n:
        s = (len(wav) - max_n) // 2
        wav = wav[s:s + max_n]
    state["wav_used"] = wav
    if state["audio_mode"] == "finetune":
        t = torch.from_numpy(wav)[None].to(dev)
        batch["wav"] = t
        batch["wav_mask"] = torch.ones_like(t, dtype=torch.bool)
    elif state["audio_mode"] == "cached":
        x, _ = state["embedder"].embed([wav])
        batch["wavlm"] = torch.from_numpy(x.astype(np.float32)).to(dev)
    return batch


@torch.no_grad()
def probs_of(model, batch: dict, **kw) -> np.ndarray:
    return torch.softmax(model(batch, **kw).float(), dim=-1)[0].cpu().numpy()


# ----------------------------------------------------------------------------- explanations

def ig_tokens(state: dict, batch: dict, target: int) -> tuple[list[str], np.ndarray]:
    """Integrated Gradients per token of the transcript, normalised to [-1, 1] by the largest magnitude."""
    from captum.attr import IntegratedGradients

    model, tok = state["model"], state["tokenizer"]
    text_model = model if state["kind"] == "text_roberta" else model.text
    ids, mask = batch["input_ids"], batch["attention_mask"]
    emb_layer = text_model.input_embeddings()
    with torch.no_grad():
        emb = emb_layer(ids)
        base = emb_layer(torch.full_like(ids, tok.pad_token_id))
        special = torch.tensor(tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True), device=ids.device).bool()
        base[0, special] = emb[0, special]
        if state["kind"] == "fusion_gated":
            h_a = model.proj_a(model.encode_audio(batch))

    def f(inputs_embeds: torch.Tensor) -> torch.Tensor:
        n = inputs_embeds.shape[0]
        m = mask.expand(n, -1)
        if state["kind"] == "text_roberta":
            logits = model.forward_from_embeds(inputs_embeds, m)
        else:
            seq = model.text.encoder(inputs_embeds=inputs_embeds, attention_mask=m).last_hidden_state
            h_t = model.proj_t(model.text._pool(seq, m))
            logits = model.fuse(h_a.expand(n, -1), h_t)
        return torch.softmax(logits, dim=-1)

    model.eval()
    attr = IntegratedGradients(f).attribute(emb, baselines=base, target=target, n_steps=IG_STEPS, internal_batch_size=8)
    score = attr.sum(-1)[0].detach().cpu().numpy()
    score = np.where(special.cpu().numpy(), 0.0, score)
    score = score / (np.abs(score).max() + 1e-12)
    tokens = [t.replace("Ġ", " ") for t in tok.convert_ids_to_tokens(ids[0].tolist())]
    return tokens, score


def tokens_to_html(tokens: list[str], score: np.ndarray) -> str:
    """Coloured spans: red supports the predicted class, blue works against it."""
    spans = []
    for t, s in zip(tokens, score):
        if t in ("<s>", "</s>", "<pad>"):
            continue
        rgb, a = ("214,64,69", s) if s >= 0 else ("52,120,200", -s)
        spans.append(f'<span title="{s:+.2f}" style="background:rgba({rgb},{min(1.0, abs(a)):.2f});padding:2px 1px;'
                     f'border-radius:3px;margin:0 1px;color:#111">{html.escape(t)}</span>')
    legend = ('<div style="font-size:12px;margin-top:6px;color:#555">Integrated Gradients on the text branch: '
              '<span style="background:rgba(214,64,69,.6);padding:0 4px">supports</span> / '
              '<span style="background:rgba(52,120,200,.6);padding:0 4px">opposes</span> the predicted class.</div>')
    return '<div style="font-size:18px;line-height:2.1;font-family:sans-serif">' + "".join(spans) + "</div>" + legend


def occlusion_saliency(state: dict, batch: dict, target: int, base_p: float, batch_size: int = 8) -> np.ndarray:
    """Drop in the predicted-class probability per 250 ms window replaced by silence (finetune audio only)."""
    model = state["model"]
    wav = batch["wav"]
    step = int(WINDOW_S * SR)
    n_win = int(np.ceil(wav.shape[1] / step))
    out = np.zeros(n_win, dtype=np.float32)
    for s in range(0, n_win, batch_size):
        idx = list(range(s, min(n_win, s + batch_size)))
        w = wav.repeat(len(idx), 1)
        for r, k in enumerate(idx):
            w[r, k * step:(k + 1) * step] = 0.0
        b = {"input_ids": batch["input_ids"].expand(len(idx), -1), "attention_mask": batch["attention_mask"].expand(len(idx), -1),
             "wav": w, "wav_mask": torch.ones_like(w, dtype=torch.bool)}
        with torch.no_grad():
            p = torch.softmax(model(b).float(), dim=-1)[:, target].cpu().numpy()
        out[idx] = base_p - p
    return out


def explain(state: dict, wav: np.ndarray, text: str) -> dict:
    """Probabilities and every explanation for one clip; the dict is what the UI renders."""
    model, kind = state["model"], state["kind"]
    batch = make_batch(state, wav, text)
    probs = probs_of(model, batch)
    target = int(probs.argmax())
    res = {"probs": {e: float(p) for e, p in zip(EMOTIONS, probs)}, "pred": EMOTIONS[target], "notes": [],
           "heatmap_html": "", "saliency": None, "gate": None, "ablation": None}
    if state["note"]:
        res["notes"].append(state["note"])
    if kind in ("fusion_gated", "text_roberta"):
        tokens, score = ig_tokens(state, batch, target)
        res["heatmap_html"] = tokens_to_html(tokens, score)
    else:
        res["notes"].append("Token heat-map is implemented for fusion_gated and text_roberta runs only.")
    if kind == "fusion_gated":
        probs_of(model, batch)  # refresh last_gate for this exact input
        res["gate"] = float(model.last_gate[0, 0])
    if kind.startswith("fusion"):
        res["ablation"] = {"audio_zeroed": float(probs_of(model, batch, drop_audio=True)[target]),
                           "text_zeroed": float(probs_of(model, batch, drop_text=True)[target]),
                           "full": float(probs[target])}
        if state["audio_mode"] == "finetune":
            res["saliency"] = occlusion_saliency(state, batch, target, float(probs[target]))
        else:
            res["notes"].append("Audio saliency needs a fusion run with audio.mode: finetune; skipped for cached audio features.")
    else:
        res["notes"].append("Text-only model: no audio saliency and no modality contribution.")
    return res


# ----------------------------------------------------------------------------- UI

def saliency_figure(wav: np.ndarray, sal: np.ndarray):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=400, hop_length=160, n_mels=64)(torch.from_numpy(wav))
    mel = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(mel).numpy()
    dur = len(wav) / SR
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8, 4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    a0.imshow(mel, origin="lower", aspect="auto", extent=[0, dur, 0, 64], cmap="magma")
    a0.set_ylabel("mel bin")
    edges = np.arange(len(sal)) * WINDOW_S
    a1.bar(edges, sal, width=WINDOW_S, align="edge", color=np.where(sal >= 0, "#d64045", "#3478c8"))
    a1.axhline(0, color="k", lw=0.5)
    a1.set_ylabel("p drop")
    a1.set_xlabel("time (s), 250 ms windows")
    a1.set_xlim(0, dur)
    return fig


def predict_ui(state: dict, audio_path: str | None):
    """Gradio callback: (transcript, label dict, heat-map html, saliency figure, modality text, notes)."""
    if not audio_path:
        return "", {}, "", None, "", "Record or upload a clip first."
    wav = load_audio(audio_path)
    if len(wav) < SR // 10:
        return "", {}, "", None, "", "The clip is shorter than 0.1 s."
    text = normalize_text(state["asr"].transcribe([wav])[0])
    if not text:
        return "", {}, "", None, "", "Whisper returned an empty transcript."
    res = explain(state, wav, text)
    fig = saliency_figure(state["wav_used"], res["saliency"]) if res["saliency"] is not None else None
    mod = ""
    if res["gate"] is not None:
        mod = f"Gate g = {res['gate']:.2f}: audio weight {res['gate']:.2f}, text weight {1 - res['gate']:.2f}. "
    if res["ablation"]:
        a = res["ablation"]
        mod += (f"Probability of '{res['pred']}': {a['full']:.2f} with both modalities, {a['audio_zeroed']:.2f} without audio, "
                f"{a['text_zeroed']:.2f} without text.")
    return text, res["probs"], res["heatmap_html"], fig, mod, "\n".join(res["notes"])


def build_app(state: dict):
    import gradio as gr

    title = "Speech emotion recognition: audio + Whisper transcript"
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"## {title}\nModel: `{state['run_dir'].relative_to(ROOT) if state['run_dir'].is_relative_to(ROOT) else state['run_dir']}` "
                    f"({state['kind']}), ASR: `{state['asr_tag']}`, device: `{state['device']}`. "
                    "Trained on MELD (TV dialogue, 7 emotion classes); do not read the output as a clinical or general-purpose estimate.")
        notes = gr.Textbox(label="Notes", value=state["note"], interactive=False, lines=2)
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speech")
        btn = gr.Button("Analyse", variant="primary")
        transcript = gr.Textbox(label="Whisper transcript", interactive=False)
        label = gr.Label(label="Emotion probabilities", num_top_classes=len(EMOTIONS))
        heat = gr.HTML(label="Token heat-map")
        sal = gr.Plot(label="Audio saliency (250 ms occlusion)")
        mod = gr.Textbox(label="Modality contribution", interactive=False, lines=2)
        btn.click(lambda p: predict_ui(state, p), inputs=audio, outputs=[transcript, label, heat, sal, mod, notes])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="runs/fusion_gated/13")
    ap.add_argument("--asr-tag", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    a = ap.parse_args()
    state = load_state(a.run, a.asr_tag, a.device)
    print(f"model {state['run_dir']} ({state['kind']}), asr {state['asr_tag']}, device {a.device}" + (f"\n{state['note']}" if state["note"] else ""))
    build_app(state).launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
