"""Before/after log-mel spectrograms for the train-time augmentation (preprocessing_validation.md Stage 6,
"visible sanity").

Usage: python scripts/plot_augmentation.py [--manifest-dir DIR] [--out reports/figures/augmentation_examples.png]
       [--config configs/audio_wavlm_ft.yaml] [--seed 13] [--n 3]

Picks ``--n`` train clips (different emotions, 2-6 s) from the manifest, applies ``AddNoise`` and
``SpeedPerturb`` from ``src.data.augment`` with the ranges of ``--config`` (``train.augment``) and a fixed
seed, and saves one row per clip: original, noise only, speed only, speed then noise (the training order).
Panel titles carry the drawn SNR, the measured SNR and the drawn speed factor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_yaml  # noqa: E402
from src.data.augment import AddNoise, SpeedPerturb  # noqa: E402
from src.data.dataset import load_manifest  # noqa: E402

SR = 16000


def log_mel(wav: torch.Tensor) -> np.ndarray:
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=400, hop_length=160, n_mels=64, f_min=20, f_max=8000)(wav)
    return torchaudio.transforms.AmplitudeToDB(top_db=80.0)(mel).numpy()


def pick_clips(manifest_dir: str | None, n: int, seed: int):
    df = load_manifest("train", manifest_dir_override=manifest_dir)
    df = df[(df["duration_s"] >= 2.0) & (df["duration_s"] <= 6.0)]
    rng = np.random.default_rng(seed)
    chosen = []
    for emotion in rng.permutation(sorted(df["emotion"].unique())):
        rows = df[df["emotion"] == emotion]
        chosen.append(rows.iloc[int(rng.integers(len(rows)))])
        if len(chosen) == n:
            break
    assert len(chosen) == n, f"only {len(chosen)} emotions with 2-6 s train clips"
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--out", default="reports/figures/augmentation_examples.png")
    ap.add_argument("--config", default="configs/audio_wavlm_ft.yaml", help="experiment config whose train.augment ranges are used")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    aug = load_yaml(a.config)["train"]["augment"]
    snr_range = tuple(aug["noise"]["snr_db"])
    speed_range = tuple(aug["speed"]["factors"])
    clips = pick_clips(a.manifest_dir, a.n, a.seed)

    fig, axes = plt.subplots(a.n, 4, figsize=(16, 3.1 * a.n), squeeze=False, constrained_layout=True)
    for r, row in enumerate(clips):
        wav_np, sr = sf.read(ROOT / row["wav_path"], dtype="float32", always_2d=False)
        assert sr == SR
        wav = torch.from_numpy(np.ascontiguousarray(wav_np if wav_np.ndim == 1 else wav_np.mean(axis=1)))
        noise = AddNoise(snr_range, rng=np.random.default_rng(a.seed + r))
        speed = SpeedPerturb(speed_range, rng=np.random.default_rng(100 + a.seed + r))
        noisy = noise(wav)
        snr_noise = noise.last_snr
        fast = speed(wav)
        both_speed = SpeedPerturb(speed_range, rng=np.random.default_rng(100 + a.seed + r))
        both_noise = AddNoise(snr_range, rng=np.random.default_rng(a.seed + r))
        both = both_noise(both_speed(wav))
        variants = [
            (wav, f"original ({len(wav) / SR:.2f} s)"),
            (noisy, f"AddNoise: drawn SNR {snr_noise:.1f} dB, measured {AddNoise.measured_snr(wav, noisy):.1f} dB"),
            (fast, f"SpeedPerturb: factor {speed.last_factor:.3f} ({len(fast) / SR:.2f} s)"),
            (both, f"speed {both_speed.last_factor:.3f} then noise {both_noise.last_snr:.1f} dB"),
        ]
        mels = [log_mel(w) for w, _ in variants]
        vmax = float(mels[0].max())
        vmin = vmax - 80.0
        for c, ((w, title), mel) in enumerate(zip(variants, mels)):
            ax = axes[r][c]
            ax.imshow(mel, origin="lower", aspect="auto", extent=[0, len(w) / SR, 0, mel.shape[0]], vmin=vmin, vmax=vmax, cmap="magma")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("time (s)", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{row['emotion']}\n{row['utt_id']}\nmel bin", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"Train-time augmentation (seed {a.seed}, noise SNR {snr_range[0]:g}-{snr_range[1]:g} dB, "
                 f"speed {speed_range[0]:g}-{speed_range[1]:g}); common colour scale per row", fontsize=11)
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    for r, row in enumerate(clips):
        print(f"  clip {r}: {row['utt_id']} ({row['emotion']}, {row['duration_s']:.2f} s)")


if __name__ == "__main__":
    main()
