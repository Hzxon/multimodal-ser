"""Model 5 (WavLM-base-plus, layer-weighted mean pool, MLP head) and the eGeMAPS MLP.

``AudioClassifier`` modes (``cfg["mode"]``):

- ``cached``: input ``batch["wavlm"]`` [B, 13, 768] = per-layer mean-pooled hidden states
  from the feature cache. Softmax over 13 learned layer logits, weighted sum, MLP head.
  Weighting the per-layer means equals mean-pooling the weighted frames, so this is the
  frozen version of ``finetune`` exactly.
- ``finetune``: runs ``microsoft/wavlm-base-plus`` on ``batch["wav"]``/``batch["wav_mask"]``
  with ``output_hidden_states=True``, masked mean pool per layer, same head. Waveform
  normalisation and whether the attention mask is passed follow the checkpoint's
  preprocessor config (``do_normalize``, ``return_attention_mask``) unless
  ``normalize_wav`` / ``use_attention_mask`` are set in the model config. The CNN
  feature encoder is always frozen; the top ``unfreeze_top_n`` transformer layers train
  (0 = fully frozen). SpecAugment is the encoder's own ``mask_time_prob`` /
  ``mask_time_length`` masking, active only in ``train()`` mode. ``init_from`` loads the
  layer logits and head from a ``cached`` run's ``best.pt``.

Both modes expose ``layer_weights()`` (softmax, detached), ``encode_audio(batch) -> h_a``
and, in finetune mode, ``encode_audio_seq(batch) -> (frames [B, F, 768], frame_mask)``.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.common import NUM_CLASSES, ROOT
from src.models.text_model import masked_mean


def mlp_head(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Dropout(dropout), nn.Linear(hidden, out_dim),
    )


class AudioClassifier(nn.Module):
    modalities = frozenset({"audio"})

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.mode = cfg.get("mode", "cached")
        assert self.mode in ("cached", "finetune"), self.mode
        self.name = cfg.get("name", "microsoft/wavlm-base-plus")
        self.n_layers = int(cfg.get("n_layers", 13))
        self.embedding_dim = int(cfg.get("hidden_size", 768))
        self.normalize_wav = bool(cfg.get("normalize_wav", True))
        self.use_attention_mask = bool(cfg.get("use_attention_mask", True))
        self.layer_logits = nn.Parameter(torch.zeros(self.n_layers))
        self.head = mlp_head(self.embedding_dim, int(cfg.get("head_hidden", 256)),
                             int(cfg.get("num_classes", NUM_CLASSES)), float(cfg.get("dropout", 0.1)))
        self.encoder = None
        if self.mode == "finetune":
            from transformers import AutoFeatureExtractor, WavLMModel

            # follow the checkpoint's preprocessing config unless the experiment config overrides it
            fe = AutoFeatureExtractor.from_pretrained(self.name)
            v = cfg.get("normalize_wav")
            self.normalize_wav = bool(fe.do_normalize) if v is None else bool(v)
            v = cfg.get("use_attention_mask")
            self.use_attention_mask = bool(fe.return_attention_mask) if v is None else bool(v)
            sa = cfg.get("spec_augment") or {}
            self.encoder = WavLMModel.from_pretrained(
                self.name, output_hidden_states=True,
                apply_spec_augment=bool(sa.get("enabled", True)),
                mask_time_prob=float(sa.get("mask_time_prob", 0.05)),
                mask_time_length=int(sa.get("mask_time_length", 10)),
                mask_feature_prob=float(sa.get("mask_feature_prob", 0.0)),
                mask_feature_length=int(sa.get("mask_feature_length", 10)),
                layerdrop=0.0,
            )
            assert self.encoder.config.num_hidden_layers + 1 == self.n_layers
            assert self.encoder.config.hidden_size == self.embedding_dim
            self._set_trainable(int(cfg.get("unfreeze_top_n", 4)))
        init_from = cfg.get("init_from")
        if init_from:
            self.load_head_from_run(init_from)

    def _set_trainable(self, unfreeze_top_n: int) -> None:
        enc = self.encoder
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.freeze_feature_encoder()
        layers = enc.encoder.layers
        n = max(0, min(unfreeze_top_n, len(layers)))
        for layer in layers[len(layers) - n:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        self.unfreeze_top_n = n

    def load_head_from_run(self, run_dir: str | Path) -> None:
        """Copy ``layer_logits`` and ``head`` from another AudioClassifier run's ``best.pt``."""
        p = Path(run_dir)
        p = p if p.is_absolute() else ROOT / p
        ckpt = torch.load(p / "best.pt", map_location="cpu", weights_only=False)
        sd = {k: v for k, v in ckpt["state_dict"].items() if k.startswith("head.") or k == "layer_logits"}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert "layer_logits" not in missing and not any(m.startswith("head.") for m in missing), missing

    def pretrained_modules(self) -> list[nn.Module]:
        return [self.encoder] if self.encoder is not None else []

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits.detach(), dim=0)

    def _weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    @staticmethod
    def _normalize(wav: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(wav.dtype)
        n = m.sum(1, keepdim=True).clamp_min(1.0)
        mean = (wav * m).sum(1, keepdim=True) / n
        var = (((wav - mean) * m) ** 2).sum(1, keepdim=True) / n
        return (wav - mean) / torch.sqrt(var + 1e-7) * m

    def _frame_mask(self, wav_mask: torch.Tensor, n_frames: int) -> torch.Tensor:
        lens = self.encoder._get_feat_extract_output_lengths(wav_mask.sum(1)).to(torch.long)
        return torch.arange(n_frames, device=wav_mask.device)[None, :] < lens[:, None]

    def encode_audio_seq(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer-weighted frame sequence [B, F, H] and its mask [B, F] (finetune mode only)."""
        assert self.mode == "finetune", "frame-level outputs need audio.mode: finetune"
        wav, mask = batch["wav"], batch["wav_mask"]
        if self.normalize_wav:
            wav = self._normalize(wav, mask)
        out = self.encoder(input_values=wav, attention_mask=mask.long() if self.use_attention_mask else None,
                           output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)  # [13, B, F, H]: input of layer 0, then each layer's output
        w = self._weights().to(hs.dtype)
        frames = torch.einsum("l,lbfh->bfh", w, hs)
        return frames, self._frame_mask(mask, frames.shape[1])

    def encode_audio(self, batch: dict) -> torch.Tensor:
        if self.mode == "cached":
            x = batch["wavlm"]  # [B, 13, H]
            assert x.shape[1] == self.n_layers, x.shape
            return torch.einsum("l,blh->bh", self._weights().to(x.dtype), x)
        frames, fmask = self.encode_audio_seq(batch)
        return masked_mean(frames, fmask)

    def classify(self, h_a: torch.Tensor) -> torch.Tensor:
        return self.head(h_a)

    def forward(self, batch: dict, drop_audio: bool = False, drop_text: bool = False) -> torch.Tensor:
        h_a = self.encode_audio(batch)
        if drop_audio:
            h_a = torch.zeros_like(h_a)
        return self.classify(h_a)


class EgemapsMLP(nn.Module):
    """Model 3 (torch variant): scaled eGeMAPS (88) -> hidden layer(s) -> 7.

    ``hidden`` is an int (one hidden layer) or a list of ints (one layer per entry), the same
    values ``src.baselines`` passes to scikit-learn's ``hidden_layer_sizes``; the last entry is
    ``embedding_dim``.
    """

    modalities = frozenset({"audio"})

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.in_dim = int(cfg.get("in_dim", 88))
        hidden = cfg.get("hidden", 128)
        sizes = [int(h) for h in hidden] if isinstance(hidden, (list, tuple)) else [int(hidden)]
        assert sizes and all(h > 0 for h in sizes), f"hidden must be a positive int or non-empty list, got {hidden!r}"
        self.embedding_dim = sizes[-1]
        d = float(cfg.get("dropout", 0.2))
        layers: list[nn.Module] = []
        prev = self.in_dim
        for h in sizes:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(d)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(self.embedding_dim, int(cfg.get("num_classes", NUM_CLASSES)))

    def pretrained_modules(self) -> list[nn.Module]:
        return []

    def encode_audio(self, batch: dict) -> torch.Tensor:
        return self.body(batch["egemaps"])

    def forward(self, batch: dict, drop_audio: bool = False, drop_text: bool = False) -> torch.Tensor:
        h = self.encode_audio(batch)
        if drop_audio:
            h = torch.zeros_like(h)
        return self.head(h)
