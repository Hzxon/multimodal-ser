"""Frozen WavLM-base-plus embeddings: 13 hidden states, mean-pooled over valid frames.

Usage: python -m src.features.ssl_embed --split {train,dev,test,all} [--manifest-dir DIR]
           [--feature-root DIR] [--limit N] [--device {cuda,mps,cpu}] [--batch-size 4]
           [--max-audio-s 10]

Cache: ``<feature-root>/wavlm-base-plus_v1-L13-mean-<max_audio_s>s/<split>.npz`` with
``X`` float16 of shape (N, 13, 768) (embedding output + 12 transformer layers), plus the
extra arrays ``cropped`` (bool, clip was centre-cropped to --max-audio-s) and ``n_frames``
(int, number of encoder frames that entered the mean). Rows follow the manifest order.
Clips are processed in length-sorted batches to limit zero padding; padded frames are
excluded from the mean.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from src.common import ROOT, SPLITS, device_name
from src.data.dataset import load_manifest
from src.features.cache import FEATURE_ROOT, cache_path, save_cache

EXTRACTOR = "wavlm-base-plus"
MODEL_ID = "microsoft/wavlm-base-plus"
SR = 16000
N_LAYERS = 13
DIM = 768


def feature_version(max_audio_s: float) -> str:
    return f"v1-L{N_LAYERS}-mean-{max_audio_s:g}s"


def read_wav(path: str | Path, max_audio_s: float) -> tuple[np.ndarray, bool]:
    """Mono float32 samples, centre-cropped to ``max_audio_s``; second value tells if cropped."""
    sig, sr = sf.read(str(path), dtype="float32", always_2d=False)
    assert sr == SR, (path, sr)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    max_n = int(round(max_audio_s * SR))
    if len(sig) > max_n:
        start = (len(sig) - max_n) // 2
        return sig[start:start + max_n], True
    return sig, False


class WavLMEmbedder:
    def __init__(self, device: str | None = None, model_id: str = MODEL_ID):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel

        self.torch = torch
        self.device = device or device_name()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = WavLMModel.from_pretrained(model_id).eval().float()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

    def _forward(self, batch: dict):
        torch = self.torch
        with torch.inference_mode():
            return self.model(batch["input_values"].to(self.device), output_hidden_states=True)

    def embed(self, wavs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """(B, 13, 768) float16 mean over valid frames, and the valid frame count per clip."""
        torch = self.torch
        lengths = torch.tensor([len(w) for w in wavs])
        batch = self.fe(wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        try:
            out = self._forward(batch)
        except (RuntimeError, NotImplementedError) as e:
            if self.device != "mps":
                raise
            print(f"mps op failed ({str(e)[:80]!r}); continuing on cpu")
            self.device = "cpu"
            self.model.to("cpu")
            out = self._forward(batch)
        n_valid = self.model._get_feat_extract_output_lengths(lengths).clamp(min=1)
        hs = torch.stack(out.hidden_states, dim=1).float().cpu()  # (B, 13, T, 768)
        t = hs.shape[2]
        mask = (torch.arange(t)[None, :] < n_valid[:, None]).float()  # (B, T)
        pooled = (hs * mask[:, None, :, None]).sum(dim=2) / mask.sum(dim=1)[:, None, None]
        return pooled.numpy().astype(np.float16), n_valid.numpy().astype(np.int64)


def embed_paths(paths: list[str], embedder: WavLMEmbedder, batch_size: int, max_audio_s: float,
                progress: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embeddings for ``paths`` in the given order; batches are formed after sorting by length."""
    wavs, cropped = [], []
    for p in paths:
        w, c = read_wav(p, max_audio_s)
        wavs.append(w)
        cropped.append(c)
    order = np.argsort([len(w) for w in wavs], kind="stable")
    X = np.zeros((len(paths), N_LAYERS, DIM), dtype=np.float16)
    n_frames = np.zeros(len(paths), dtype=np.int64)
    it = range(0, len(order), batch_size)
    for i in tqdm(it, desc="wavlm", disable=not progress):
        idx = order[i:i + batch_size]
        x, n = embedder.embed([wavs[j] for j in idx])
        X[idx] = x
        n_frames[idx] = n
    return X, n_frames, np.array(cropped, dtype=bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=[*SPLITS, "all"])
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--feature-root", default=str(FEATURE_ROOT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-audio-s", type=float, default=10.0)
    a = ap.parse_args()

    feature_root = Path(a.feature_root)
    if not feature_root.is_absolute():
        feature_root = ROOT / feature_root
    version = feature_version(a.max_audio_s)
    embedder = WavLMEmbedder(a.device)
    print(f"device={embedder.device} model={MODEL_ID} version={version}")
    splits = list(SPLITS) if a.split == "all" else [a.split]
    for split in splits:
        df = load_manifest(split, require_audio=True, apply_exclusions=False, manifest_dir_override=a.manifest_dir)
        if a.limit:
            df = df.head(a.limit)
        paths = [str(ROOT / p) for p in df["wav_path"]]
        t0 = time.time()
        X, n_frames, cropped = embed_paths(paths, embedder, a.batch_size, a.max_audio_s)
        dt = time.time() - t0
        assert np.isfinite(X.astype(np.float32)).all(), f"non-finite WavLM values in {split}"
        out = save_cache(cache_path(EXTRACTOR, version, split, feature_root), split=split, extractor=EXTRACTOR,
                         version=version, utt_ids=df["utt_id"].to_numpy(), X=X, cropped=cropped, n_frames=n_frames)
        audio_s = float(df["duration_s"].clip(upper=a.max_audio_s).sum())
        print(f"{split}: {len(df)} clips -> {out} ({dt:.1f}s, {1000 * dt / max(len(df), 1):.0f} ms/clip, "
              f"{audio_s / dt:.1f}x real time, device={embedder.device})")
        print(f"{split}: cropped to {a.max_audio_s:g}s: {int(cropped.sum())} / {len(df)} "
              f"({100 * cropped.mean():.1f}%)")


if __name__ == "__main__":
    main()
