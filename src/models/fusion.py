"""Fusion models 6-8 (PLAN.md 3.4).

- ``GatedFusion`` (model 7): ``g = sigmoid(W[h_a; h_t])``, ``z = [h_a; h_t; g*h_a; (1-g)*h_t]``
  -> MLP -> logits. ``self.last_gate`` [B, 1] is kept after every forward.
- ``CrossAttentionFusion`` (model 8): one multi-head attention block each way over the
  encoders' sequence outputs (text tokens query audio frames, audio frames query text
  tokens), masked mean pool, concat with the pooled unimodal embeddings -> MLP. Needs
  ``audio.mode: finetune`` because the cached features have no frames. ``self.last_attn``
  holds ``{"text_to_audio": [B, L, F], "audio_to_text": [B, F, L]}`` (head-averaged,
  detached).
- ``late_fusion_probs`` (model 6): weighted average of two probability tables; the weight
  search on dev lives in the evaluation code.

Both trained fusion models own a ``TextClassifier`` (``self.text``) and an
``AudioClassifier`` (``self.audio``) built from ``cfg["text"]`` / ``cfg["audio"]``,
optionally initialised from unimodal runs (``init_from: {audio: runs/..., text: runs/...}``).
Modality dropout ``p_mod_drop`` zeroes one modality's embedding per sample in training,
never both. ``drop_audio`` / ``drop_text`` do the same at inference for the XAI
modality-ablation score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.common import NUM_CLASSES, ROOT
from src.models.audio_model import AudioClassifier, mlp_head
from src.models.text_model import TextClassifier, masked_mean


def _load_submodule_from_run(module: nn.Module, run_dir: str | Path, prefix: str = "") -> None:
    p = Path(run_dir)
    p = p if p.is_absolute() else ROOT / p
    ckpt = torch.load(p / "best.pt", map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    if prefix:
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    assert not unexpected, f"{run_dir}: unexpected keys {unexpected[:5]}"
    real_missing = [m for m in missing if not m.startswith("encoder.")]
    assert not real_missing, f"{run_dir}: missing keys {real_missing[:5]}"


class _FusionBase(nn.Module):
    modalities = frozenset({"audio", "text"})

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.text = TextClassifier(cfg["text"])
        self.audio = AudioClassifier(cfg["audio"])
        init = cfg.get("init_from") or {}
        if init.get("text"):
            _load_submodule_from_run(self.text, init["text"])
        if init.get("audio"):
            _load_submodule_from_run(self.audio, init["audio"])
        self.p_mod_drop = float(cfg.get("p_mod_drop", 0.15))
        self.d_model = int(cfg.get("d_model", 256))
        self.proj_a = nn.Sequential(nn.LayerNorm(self.audio.embedding_dim), nn.Linear(self.audio.embedding_dim, self.d_model))
        self.proj_t = nn.Sequential(nn.LayerNorm(self.text.embedding_dim), nn.Linear(self.text.embedding_dim, self.d_model))
        self.embedding_dim = self.d_model
        self.num_classes = int(cfg.get("num_classes", NUM_CLASSES))
        self.last_drop: torch.Tensor | None = None

    def pretrained_modules(self) -> list[nn.Module]:
        return self.text.pretrained_modules() + self.audio.pretrained_modules()

    def encode_text(self, batch: dict) -> torch.Tensor:
        return self.text.encode_text(batch["input_ids"], batch["attention_mask"])

    def encode_audio(self, batch: dict) -> torch.Tensor:
        return self.audio.encode_audio(batch)

    def _modality_masks(self, B: int, device, drop_audio: bool, drop_text: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample keep masks (keep_a, keep_t) as float [B, 1]; never both zero."""
        keep_a = torch.ones(B, 1, device=device)
        keep_t = torch.ones(B, 1, device=device)
        if self.training and self.p_mod_drop > 0:
            drop = torch.rand(B, 1, device=device) < self.p_mod_drop
            which_audio = torch.rand(B, 1, device=device) < 0.5
            keep_a = keep_a * (~(drop & which_audio)).float()
            keep_t = keep_t * (~(drop & ~which_audio)).float()
        if drop_audio:
            keep_a = torch.zeros_like(keep_a)
        if drop_text:
            keep_t = torch.zeros_like(keep_t)
        self.last_drop = torch.stack([1 - keep_a, 1 - keep_t], dim=1).squeeze(-1).detach()
        return keep_a, keep_t


class GatedFusion(_FusionBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        d = self.d_model
        self.gate = nn.Linear(2 * d, 1)
        self.head = mlp_head(4 * d, int(cfg.get("head_hidden", 256)), self.num_classes, float(cfg.get("dropout", 0.1)))
        self.last_gate: torch.Tensor | None = None

    def fuse(self, h_a: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([h_a, h_t], dim=-1)))  # [B, 1]
        self.last_gate = g.detach()
        z = torch.cat([h_a, h_t, g * h_a, (1 - g) * h_t], dim=-1)
        return self.head(z)

    def forward(self, batch: dict, drop_audio: bool = False, drop_text: bool = False) -> torch.Tensor:
        h_a = self.proj_a(self.encode_audio(batch))
        h_t = self.proj_t(self.encode_text(batch))
        keep_a, keep_t = self._modality_masks(h_a.shape[0], h_a.device, drop_audio, drop_text)
        return self.fuse(h_a * keep_a, h_t * keep_t)


class CrossAttentionFusion(_FusionBase):
    def __init__(self, cfg: dict):
        assert cfg["audio"].get("mode") == "finetune", "cross-attention fusion needs audio.mode: finetune (frame outputs)"
        super().__init__(cfg)
        d, heads = self.d_model, int(cfg.get("n_heads", 4))
        drop = float(cfg.get("dropout", 0.1))
        self.attn_t2a = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.attn_a2t = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm_t = nn.LayerNorm(d)
        self.norm_a = nn.LayerNorm(d)
        self.head = mlp_head(4 * d, int(cfg.get("head_hidden", 256)), self.num_classes, drop)
        self.last_attn: dict[str, torch.Tensor] = {}

    def forward(self, batch: dict, drop_audio: bool = False, drop_text: bool = False) -> torch.Tensor:
        frames, fmask = self.audio.encode_audio_seq(batch)  # [B, F, Ha], [B, F]
        tokens = self.text.encode_text_seq(batch["input_ids"], batch["attention_mask"])  # [B, L, Ht]
        tmask = batch["attention_mask"].bool()
        a = self.proj_a(frames)
        t = self.proj_t(tokens)
        keep_a, keep_t = self._modality_masks(a.shape[0], a.device, drop_audio, drop_text)
        a = a * keep_a.unsqueeze(-1)
        t = t * keep_t.unsqueeze(-1)
        t2a, w_t2a = self.attn_t2a(t, a, a, key_padding_mask=~fmask, need_weights=True, average_attn_weights=True)
        a2t, w_a2t = self.attn_a2t(a, t, t, key_padding_mask=~tmask, need_weights=True, average_attn_weights=True)
        self.last_attn = {"text_to_audio": w_t2a.detach(), "audio_to_text": w_a2t.detach()}
        h_ta = masked_mean(self.norm_t(t + t2a), tmask)
        h_at = masked_mean(self.norm_a(a + a2t), fmask)
        h_a = masked_mean(a, fmask)
        h_t = masked_mean(t, tmask)
        return self.head(torch.cat([h_a, h_t, h_ta, h_at], dim=-1))


def late_fusion_probs(p_a: np.ndarray, p_t: np.ndarray, w: float) -> np.ndarray:
    """``w * p_a + (1 - w) * p_t`` row-normalised (model 6)."""
    p_a, p_t = np.asarray(p_a, dtype=float), np.asarray(p_t, dtype=float)
    assert p_a.shape == p_t.shape, (p_a.shape, p_t.shape)
    p = w * p_a + (1.0 - w) * p_t
    return p / p.sum(axis=1, keepdims=True)


class LateFusion:
    """Model 6: holds the weight chosen on dev and applies ``late_fusion_probs``."""

    def __init__(self, w: float = 0.5):
        self.w = float(w)

    def __call__(self, p_a: np.ndarray, p_t: np.ndarray) -> np.ndarray:
        return late_fusion_probs(p_a, p_t, self.w)
