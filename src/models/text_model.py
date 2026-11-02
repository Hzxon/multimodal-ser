"""Model 4: RoBERTa-base with a pooled linear head (PLAN.md 3.4).

Interface shared by every model in ``src/models``: ``forward(batch, drop_audio=False,
drop_text=False) -> logits [B, 7]``, ``encode_text(input_ids, attention_mask) -> h_t``,
``modalities``, ``embedding_dim``, ``pretrained_modules()`` (parameters that get the
encoder learning rate). ``forward_from_embeds`` and ``input_embeddings`` exist for
Integrated Gradients (Phase 6).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from src.common import NUM_CLASSES


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` [B, L, H] over the positions where ``mask`` [B, L] is non-zero."""
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


class TextClassifier(nn.Module):
    modalities = frozenset({"text"})

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)
        self.name = cfg.get("name", "roberta-base")
        self.pooling = cfg.get("pooling", "cls")
        assert self.pooling in ("cls", "mean"), self.pooling
        self.encoder = AutoModel.from_pretrained(self.name, add_pooling_layer=False)
        self.embedding_dim = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.1)))
        self.head = nn.Linear(self.embedding_dim, int(cfg.get("num_classes", NUM_CLASSES)))
        n_freeze = int(cfg.get("freeze_bottom_n", 0))
        if n_freeze:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad_(False)
            for layer in self.encoder.encoder.layer[:n_freeze]:
                for p in layer.parameters():
                    p.requires_grad_(False)

    def pretrained_modules(self) -> list[nn.Module]:
        return [self.encoder]

    def _pool(self, seq: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return seq[:, 0]
        return masked_mean(seq, attention_mask)

    def encode_text_seq(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last hidden states [B, L, H] (for cross-attention fusion)."""
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self._pool(self.encode_text_seq(input_ids, attention_mask), attention_mask)

    def classify(self, h_t: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(h_t))

    def forward(self, batch: dict, drop_audio: bool = False, drop_text: bool = False) -> torch.Tensor:
        h_t = self.encode_text(batch["input_ids"], batch["attention_mask"])
        if drop_text:
            h_t = torch.zeros_like(h_t)
        return self.classify(h_t)

    def forward_from_embeds(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Logits from input embeddings [B, L, H] (Integrated Gradients path)."""
        seq = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
        return self.classify(self._pool(seq, attention_mask))

    def input_embeddings(self) -> nn.Embedding:
        return self.encoder.get_input_embeddings()
