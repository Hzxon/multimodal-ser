"""Stage 6 of reports/preprocessing_validation.md: augmentation validity (finite, length
policy, RMS within 10 dB) and measured SNR within 2 dB of the drawn SNR."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.augment import AddNoise, Compose, SpeedPerturb, TokenDropout, build_transforms  # noqa: E402

SMOKE = ROOT / "data/processed/smoke_manifests"


def _clips(n: int) -> list[torch.Tensor]:
    import pandas as pd
    import soundfile as sf

    if not (SMOKE / "train.csv").exists():
        rng = np.random.default_rng(0)
        return [torch.from_numpy(rng.standard_normal(16000 * 2).astype(np.float32)) for _ in range(n)]
    df = pd.read_csv(SMOKE / "train.csv")
    out = []
    for p in df["wav_path"].head(n):
        wav, sr = sf.read(str(ROOT / p), dtype="float32")
        assert sr == 16000
        out.append(torch.from_numpy(wav))
    return out


def _db(x: torch.Tensor) -> float:
    return float(20 * torch.log10(torch.sqrt(torch.mean(x ** 2)) + 1e-9))


def test_measured_snr_within_2db():
    aug = AddNoise((10, 30), rng=np.random.default_rng(1))
    for clean in _clips(20):
        noisy = aug(clean)
        assert 10 <= aug.last_snr <= 30
        assert abs(AddNoise.measured_snr(clean, noisy) - aug.last_snr) < 2.0


def test_augmented_audio_valid():
    wave, text = build_transforms({"enabled": True, "speed": {"factors": [0.9, 1.1]}, "noise": {"snr_db": [10, 30]}}, seed=3)
    assert [type(t).__name__ for t in wave] == ["SpeedPerturb", "AddNoise"] and text == []
    pipe = Compose(wave)
    max_samples = 16000 * 10
    for clean in _clips(20):
        for _ in range(5):
            out = pipe(clean)
            assert torch.isfinite(out).all()
            assert out.dtype == torch.float32 and out.ndim == 1
            f = wave[0].last_factor
            assert abs(out.shape[0] - clean.shape[0] / f) <= 0.01 * clean.shape[0] + 2
            assert out.shape[0] <= max(max_samples, int(clean.shape[0] / 0.9) + 2)
            assert abs(_db(out) - _db(clean)) < 10.0


def test_speed_perturb_length():
    sp = SpeedPerturb((1.1, 1.1), rng=np.random.default_rng(0))
    x = torch.randn(16000)
    y = sp(x)
    assert abs(y.shape[0] - 16000 / 1.1) <= 2


def test_reproducible_with_seed():
    x = torch.randn(8000)
    a = Compose(build_transforms({"noise": {"snr_db": [10, 30]}, "speed": {}}, seed=7)[0])(x)
    b = Compose(build_transforms({"noise": {"snr_db": [10, 30]}, "speed": {}}, seed=7)[0])(x)
    assert torch.equal(a, b)


def test_token_dropout():
    td = TokenDropout(0.5, rng=np.random.default_rng(0))
    s = "one two three four five six seven eight"
    outs = {td(s) for _ in range(20)}
    assert all(len(o.split()) >= 1 for o in outs) and any(len(o.split()) < 8 for o in outs)
    assert td("hello") == "hello"
    assert build_transforms({"enabled": False, "noise": {}}, 0) == ([], [])
