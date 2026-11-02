"""Model registry: ``build_model(cfg)`` and ``load_run(run_dir, device)``.

``cfg["model"]["type"]`` selects the class: ``text_roberta`` (model 4), ``audio_wavlm``
(model 5, ``mode: cached|finetune``), ``egemaps_mlp`` (torch model 3), ``fusion_gated``
(model 7), ``fusion_xattn`` (model 8). Every model follows the interface described in
``src/models/text_model.py``.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.common import ROOT, load_yaml

MODEL_TYPES = ("text_roberta", "audio_wavlm", "egemaps_mlp", "fusion_gated", "fusion_xattn")


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    t = m["type"]
    if t == "text_roberta":
        from src.models.text_model import TextClassifier

        return TextClassifier(m)
    if t == "audio_wavlm":
        from src.models.audio_model import AudioClassifier

        return AudioClassifier(m)
    if t == "egemaps_mlp":
        from src.models.audio_model import EgemapsMLP

        return EgemapsMLP(m)
    if t == "fusion_gated":
        from src.models.fusion import GatedFusion

        return GatedFusion(m)
    if t == "fusion_xattn":
        from src.models.fusion import CrossAttentionFusion

        return CrossAttentionFusion(m)
    raise ValueError(f"unknown model type {t!r}; one of {MODEL_TYPES}")


def split_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """(pretrained-encoder params, everything else), trainable only."""
    enc_ids = {id(p) for mod in model.pretrained_modules() for p in mod.parameters()}
    enc = [p for p in model.parameters() if p.requires_grad and id(p) in enc_ids]
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    return enc, head


def tokenizer_name(cfg: dict) -> str | None:
    m = cfg["model"]
    if m["type"] == "text_roberta":
        return m.get("name", "roberta-base")
    if m["type"] in ("fusion_gated", "fusion_xattn"):
        return m["text"].get("name", "roberta-base")
    return None


def load_run(run_dir: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model of a run from ``config.yaml`` and load ``best.pt`` (eval mode)."""
    p = Path(run_dir)
    p = p if p.is_absolute() else ROOT / p
    cfg = load_yaml(p / "config.yaml")
    m = cfg["model"]
    # unimodal checkpoints used for initialisation are not needed once best.pt exists
    m.pop("init_from", None)
    if m["type"] == "audio_wavlm":
        m.pop("init_from", None)
    model = build_model(cfg)
    ckpt = torch.load(p / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg
