"""Torch ``Dataset``/``DataLoader`` over the manifests for models 4-8.

``MELDDataset`` reads one split through ``src.data.dataset.load_manifest`` (data policy,
ASR join, label index from ``configs/data.yaml``), loads only the inputs the model needs
(``required_inputs``), applies the audio length policy (random crop on train, centre crop
otherwise, ``max_audio_s`` from the config) and, on the train split only, the on-the-fly
augmentation from ``src.data.augment`` (Gate F6). Cached features come from
``src.features.cache`` and are aligned to the manifest rows by ``utt_id``.

Batch keys produced by ``collate``: ``utt_id`` (list[str]), ``label`` (long [B]),
``wav`` (float32 [B, T], zero padded), ``wav_mask`` (bool [B, T], True on real samples),
``wav_len`` (long [B]), ``input_ids``/``attention_mask`` (long [B, L]), ``wavlm``
(float32 [B, 13, 768]), ``egemaps`` (float32 [B, 88]). Labels are never padded, so no
padding value can collide with a class index.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from src.common import NUM_CLASSES, ROOT
from src.data.augment import build_transforms
from src.data.dataset import load_manifest
from src.features.cache import align, cache_path, load_cache

SAMPLE_RATE = 16000
WAVLM_CACHE = {"extractor": "wavlm-base-plus", "version": "v1-L13-mean-10s"}
EGEMAPS_CACHE = {"extractor": "egemaps", "version": "v02-opensmile2.6.0"}
INPUT_KEYS = ("wav", "text", "wavlm", "egemaps")


def required_inputs(cfg: dict) -> set[str]:
    """Which item keys the configured model consumes, from ``cfg["model"]``."""
    m = cfg["model"]
    t = m["type"]
    if t in ("text_roberta",):
        return {"text"}
    if t == "egemaps_mlp":
        return {"egemaps"}
    if t == "audio_wavlm":
        return {"wav"} if m.get("mode", "cached") == "finetune" else {"wavlm"}
    if t in ("fusion_gated", "fusion_xattn"):
        audio_mode = m["audio"].get("mode", "cached")
        return {"text", "wav" if audio_mode == "finetune" else "wavlm"}
    raise ValueError(f"unknown model type {t!r}")


def _feature_root(cfg: dict, override: str | Path | None) -> Path:
    p = Path(override) if override is not None else Path(cfg["data"].get("feature_root", "data/processed/features"))
    return p if p.is_absolute() else ROOT / p


def load_split_df(split: str, cfg: dict, *, limit: int | None = None,
                  manifest_dir: str | Path | None = None) -> pd.DataFrame:
    """Manifest rows for ``split`` with the config's ASR tag / text source applied."""
    d = cfg["data"]
    text_source = d.get("text_source", "gold")
    df = load_manifest(
        split,
        asr_tag=d.get("asr_tag") if text_source == "asr" else None,
        text_source=text_source,
        manifest_dir_override=manifest_dir if manifest_dir is not None else d.get("manifest_dir"),
    )
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)
    return df


class MELDDataset(Dataset):
    """One manifest split as torch items; see the module docstring for the item keys."""

    def __init__(self, df: pd.DataFrame, split: str, cfg: dict, train: bool,
                 feature_root: str | Path | None = None, inputs: Iterable[str] | None = None,
                 seed: int | None = None):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.cfg = cfg
        self.train = bool(train)
        self.inputs = set(inputs) if inputs is not None else required_inputs(cfg)
        assert self.inputs <= set(INPUT_KEYS), self.inputs
        tcfg = cfg.get("train", {})
        self.max_audio_s = float(tcfg.get("max_audio_s", 10.0))
        self.max_samples = int(round(self.max_audio_s * SAMPLE_RATE))
        seed = int(seed if seed is not None else tcfg.get("seed", 0))
        self.rng = np.random.default_rng(seed + (0 if train else 1))
        self.transforms, self.text_transforms = ([], []) if not train else build_transforms(tcfg.get("augment"), seed)
        self.n_cropped = 0
        self.n_served_audio = 0
        self.utt_ids: list[str] = self.df["utt_id"].tolist()
        self.labels: np.ndarray = self.df["label"].to_numpy().astype(int)
        self.texts: list[str] = ["" if pd.isna(t) else str(t) for t in self.df["text"]]
        root = _feature_root(cfg, feature_root)
        caches = cfg["data"].get("caches", {})
        self.wavlm = self.egemaps = None
        if "wavlm" in self.inputs:
            spec = {**WAVLM_CACHE, **caches.get("wavlm", {})}
            c = load_cache(cache_path(spec["extractor"], spec["version"], split, root=root), split=split,
                           extractor=spec["extractor"], version=spec["version"])
            self.wavlm = align(c, self.utt_ids)
            assert self.wavlm.ndim == 3, self.wavlm.shape
        if "egemaps" in self.inputs:
            spec = {**EGEMAPS_CACHE, **caches.get("egemaps", {})}
            c = load_cache(cache_path(spec["extractor"], spec["version"], split, root=root), split=split,
                           extractor=spec["extractor"], version=spec["version"])
            X = align(c, self.utt_ids).astype(np.float32)
            scaler_path = root / f"{spec['extractor']}_{spec['version']}" / "scaler.pkl"
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            assert scaler.get("fit_split") == "train", f"{scaler_path} fit on {scaler.get('fit_split')!r}"
            self.egemaps = (X - np.asarray(scaler["mean"], dtype=np.float32)) / (np.asarray(scaler["std"], dtype=np.float32) + 1e-8)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def fraction_cropped(self) -> float:
        return self.n_cropped / self.n_served_audio if self.n_served_audio else 0.0

    def _load_wav(self, i: int) -> torch.Tensor:
        wav, sr = sf.read(str(ROOT / self.df.at[i, "wav_path"]), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        assert sr == SAMPLE_RATE, (sr, self.df.at[i, "utt_id"])
        return torch.from_numpy(np.ascontiguousarray(wav))

    def _crop(self, wav: torch.Tensor) -> torch.Tensor:
        self.n_served_audio += 1
        T = wav.shape[-1]
        if T <= self.max_samples:
            return wav
        self.n_cropped += 1
        if self.train:
            start = int(self.rng.integers(0, T - self.max_samples + 1))
        else:
            start = (T - self.max_samples) // 2
        return wav[start:start + self.max_samples]

    def __getitem__(self, i: int) -> dict[str, Any]:
        item: dict[str, Any] = {"utt_id": self.utt_ids[i], "label": int(self.labels[i])}
        if "text" in self.inputs:
            text = self.texts[i]
            for t in self.text_transforms:
                text = t(text)
            item["text"] = text
        if "wav" in self.inputs:
            wav = self._load_wav(i)
            for t in self.transforms:
                wav = t(wav)
            item["wav"] = self._crop(wav).float()
        if self.wavlm is not None:
            item["wavlm"] = torch.from_numpy(self.wavlm[i].astype(np.float32))
        if self.egemaps is not None:
            item["egemaps"] = torch.from_numpy(self.egemaps[i])
        return item


def collate(batch: list[dict], tokenizer=None, max_len: int = 64) -> dict[str, Any]:
    out: dict[str, Any] = {
        "utt_id": [b["utt_id"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }
    assert int(out["label"].min()) >= 0 and int(out["label"].max()) < NUM_CLASSES
    if "wav" in batch[0]:
        lens = torch.tensor([b["wav"].shape[-1] for b in batch], dtype=torch.long)
        T = int(lens.max())
        wav = torch.zeros(len(batch), T, dtype=torch.float32)
        for k, b in enumerate(batch):
            wav[k, : b["wav"].shape[-1]] = b["wav"]
        out["wav"] = wav
        out["wav_len"] = lens
        out["wav_mask"] = torch.arange(T)[None, :] < lens[:, None]
    if "text" in batch[0]:
        out["text"] = [b["text"] for b in batch]
        if tokenizer is not None:
            enc = tokenizer(out["text"], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            out["input_ids"] = enc["input_ids"]
            out["attention_mask"] = enc["attention_mask"]
    for key in ("wavlm", "egemaps"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch]).float()
    return out


def make_loader(split: str, cfg: dict, train: bool, *, seed: int = 0, batch_size: int | None = None,
                feature_root: str | Path | None = None, tokenizer=None, limit: int | None = None,
                manifest_dir: str | Path | None = None, inputs: Iterable[str] | None = None,
                df: pd.DataFrame | None = None, num_workers: int = 0, shuffle: bool | None = None,
                drop_last: bool = False) -> tuple[DataLoader, MELDDataset]:
    """Loader for one of the splits listed in ``cfg["data"]["splits"]`` (never ``test``)."""
    splits = cfg["data"]["splits"]
    assert split in splits.values(), f"split {split!r} not in cfg data.splits {splits}"
    assert "test" not in split, split
    if df is None:
        df = load_split_df(split, cfg, limit=limit, manifest_dir=manifest_dir)
    ds = MELDDataset(df, split, cfg, train=train, feature_root=feature_root, inputs=inputs, seed=seed)
    bs = int(batch_size or cfg.get("train", {}).get("batch_size", 16))
    max_len = int(cfg.get("train", {}).get("max_text_tokens", 64))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(
        ds, batch_size=bs, shuffle=train if shuffle is None else shuffle, num_workers=num_workers,
        collate_fn=lambda b: collate(b, tokenizer=tokenizer, max_len=max_len), generator=gen, drop_last=drop_last,
    )
    return loader, ds
