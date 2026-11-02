"""Train-only waveform and text augmentation, applied inside ``MELDDataset.__getitem__``
(PLAN.md Phase 3 step 5, Gate F6).

``build_transforms(cfg, seed)`` returns the list the train dataset applies; dev and test
datasets get ``[]``. Every transform draws its randomness from one ``numpy`` generator
seeded from the run seed, so a run is reproducible with ``num_workers=0``.

SpecAugment is not implemented here. It is the WavLM encoder's own feature masking
(``mask_time_prob`` / ``mask_time_length`` / ``apply_spec_augment`` in the Hugging Face
config), which the encoder applies to its CNN output only in ``model.train()`` mode. It is
configured under ``model.audio.spec_augment`` in the experiment YAML and only matters for
``audio.mode: finetune``; see ``src/models/audio_model.py``.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import torch

SAMPLE_RATE = 16000


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.float() ** 2) + 1e-12)


class AddNoise:
    """Additive Gaussian noise at an SNR drawn uniformly from ``snr_db`` (dB)."""

    def __init__(self, snr_db: Sequence[float] = (10.0, 30.0), rng: np.random.Generator | None = None):
        self.snr_db = (float(snr_db[0]), float(snr_db[1]))
        self.rng = rng or np.random.default_rng(0)
        self.last_snr: float | None = None

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        snr = float(self.rng.uniform(*self.snr_db))
        self.last_snr = snr
        noise = torch.from_numpy(self.rng.standard_normal(wav.shape[-1]).astype(np.float32))
        sigma = _rms(wav) / (10.0 ** (snr / 20.0))
        return wav + noise * sigma

    @staticmethod
    def measured_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
        p_signal = torch.mean(clean.float() ** 2)
        p_noise = torch.mean((noisy.float() - clean.float()) ** 2)
        return float(10.0 * torch.log10(p_signal / (p_noise + 1e-12)))


def _resample(wav: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    try:
        import torchaudio.functional as F

        return F.resample(wav, orig_freq=orig_freq, new_freq=new_freq)
    except (ImportError, AttributeError):
        import librosa

        y = librosa.resample(wav.numpy(), orig_sr=orig_freq, target_sr=new_freq)
        return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))


class SpeedPerturb:
    """Speed change by a factor drawn uniformly from ``factors``: the clip is treated as if
    sampled at ``rate * factor`` and resampled back to ``rate``, so its length becomes
    ``T / factor`` and the pitch shifts with it (Kaldi-style speed perturbation)."""

    def __init__(self, factors: Sequence[float] = (0.9, 1.1), rate: int = SAMPLE_RATE,
                 rng: np.random.Generator | None = None):
        self.factors = (float(factors[0]), float(factors[1]))
        self.rate = rate
        self.rng = rng or np.random.default_rng(0)
        self.last_factor: float | None = None

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        f = float(self.rng.uniform(*self.factors))
        self.last_factor = f
        if abs(f - 1.0) < 1e-3:
            return wav
        orig = int(round(self.rate * f))
        g = math.gcd(orig, self.rate)
        return _resample(wav, orig // g, self.rate // g)


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __len__(self) -> int:
        return len(self.transforms)

    def __iter__(self):
        return iter(self.transforms)


class TokenDropout:
    """Word-level dropout on the transcript string (ablation only, off by default). Keeps
    at least one word so the tokenizer never sees an empty augmented string."""

    def __init__(self, p: float = 0.1, rng: np.random.Generator | None = None):
        self.p = float(p)
        self.rng = rng or np.random.default_rng(0)

    def __call__(self, text: str) -> str:
        words = text.split()
        if len(words) <= 1:
            return text
        keep = self.rng.uniform(size=len(words)) >= self.p
        if not keep.any():
            keep[self.rng.integers(len(words))] = True
        return " ".join(w for w, k in zip(words, keep) if k)


def build_transforms(aug_cfg: dict | None, seed: int) -> tuple[list, list]:
    """``(wave_transforms, text_transforms)`` from ``cfg["train"]["augment"]``.

    Schema (all optional): ``enabled: true``, ``noise: {snr_db: [10, 30]}``,
    ``speed: {factors: [0.9, 1.1]}``, ``token_dropout: {p: 0.1}``. A section set to
    ``false``/``null`` is skipped. Returns two empty lists when disabled.
    """
    if not aug_cfg or not aug_cfg.get("enabled", True):
        return [], []
    rng = np.random.default_rng(seed)
    wave: list = []
    text: list = []
    speed = aug_cfg.get("speed")
    if speed:
        wave.append(SpeedPerturb(tuple(speed.get("factors", (0.9, 1.1))), rng=rng))
    noise = aug_cfg.get("noise")
    if noise:
        wave.append(AddNoise(tuple(noise.get("snr_db", (10.0, 30.0))), rng=rng))
    td = aug_cfg.get("token_dropout")
    if td:
        text.append(TokenDropout(float(td.get("p", 0.1)), rng=rng))
    return wave, text
