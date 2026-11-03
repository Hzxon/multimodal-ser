"""Whisper transcription of every clip, cached next to the manifests (PLAN.md Phase 3, step 1).

Usage: python -m src.asr.transcribe --split {train,dev,test,all} [--tag whisper-small|whisper-large-v3-turbo]
       [--manifest-dir DIR] [--limit N] [--device cuda|mps|cpu] [--batch-size N]

Writes ``<manifest_dir>/asr/<tag>/<split>.csv`` with ``utt_id, asr_text, asr_model, asr_flag``.
Decoding is pinned in ``configs/asr.yaml`` (forced English, greedy, temperature 0, fp32), so
the output is deterministic (Gate F1). Rows excluded by the data policy are transcribed too.
Resumable: utt_ids already in the file are skipped. ``asr_flag`` is the hallucination screen of
preprocessing_validation.md Stage 3; flagged transcripts are kept.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.asr.quality import wer_normalize
from src.common import ROOT, SPLITS, device_name, load_yaml, set_seed
from src.data.dataset import asr_path, load_manifest, manifest_dir

ASR_COLUMNS = ["utt_id", "asr_text", "asr_model", "asr_flag"]


def resolve_model(cfg: dict, tag: str | None) -> tuple[str, str]:
    """(tag, model_id) for ``tag``; defaults to the iteration model."""
    tag = tag or cfg["tag"]
    if tag == cfg["tag"]:
        return tag, cfg["model_id"]
    if tag == cfg["final"]["tag"]:
        return tag, cfg["final"]["model_id"]
    raise ValueError(f"tag {tag!r} is neither {cfg['tag']!r} nor {cfg['final']['tag']!r} in configs/asr.yaml")


def asr_model_string(cfg: dict, model_id: str) -> str:
    """The pinned ``asr_model`` value: model id, library versions and decoding options."""
    import torch
    import transformers

    d = cfg["decoding"]
    greedy = "greedy" if d["num_beams"] == 1 and not d["do_sample"] else f"beams={d['num_beams']},sample={d['do_sample']}"
    return (f"{model_id}|transformers={transformers.__version__}|torch={torch.__version__}|lang={cfg['language']}"
            f"|task={cfg['task']}|{cfg['dtype']}|{greedy}|temp={d['temperature']:g}|max_new_tokens={d['max_new_tokens']}"
            f"|long_form={cfg['long_form']['mode']}")


def clip_rms_dbfs(wav: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(wav, dtype=np.float64))) + 1e-9)) if len(wav) else -180.0


def hallucination_flags(asr_text: str, gold_text: str, rms_db: float, cfg: dict) -> str:
    """Comma-joined subset of {empty, too_long, repeat_3gram, filler_on_silence}; "" when clean."""
    h = cfg["hallucination"]
    a, g = wer_normalize(asr_text), wer_normalize(gold_text)
    a_words, g_words = a.split(), g.split()
    flags = []
    if not a_words:
        flags.append("empty")
    if len(g_words) >= h["too_long_min_gold_words"] and len(a_words) > h["too_long_ratio"] * len(g_words):
        flags.append("too_long")
    trigrams = Counter(zip(a_words, a_words[1:], a_words[2:]))
    if trigrams and max(trigrams.values()) > h["repeat_3gram_max"]:
        flags.append("repeat_3gram")
    if rms_db < h["silence_rms_dbfs"] and a_words and (
            a in {wer_normalize(f) for f in h["fillers"]} or any(wer_normalize(p) in a for p in h["filler_phrases"])):
        flags.append("filler_on_silence")
    return ",".join(flags)


class Transcriber:
    """Whisper wrapper: fp32, forced language and task, greedy, no sampling. Short clips are
    batched; clips above ``long_form.max_short_seconds`` go one at a time through HF's
    sequential long-form decoding. On mps, a failing call is retried on cpu and the model
    stays there."""

    def __init__(self, cfg: dict, model_id: str, device: str | None = None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch, self.cfg = torch, cfg
        self.device = device or device_name()
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=getattr(torch, cfg["dtype"]))
        self.model.eval().to(self.device)
        self.max_short = int(cfg["long_form"]["max_short_seconds"] * cfg["sample_rate"])

    def _generate(self, feats: dict, long_form: bool) -> list[str]:
        torch, c = self.torch, self.cfg
        d = c["decoding"]
        kw = dict(language=c["language"], task=c["task"], num_beams=d["num_beams"], do_sample=d["do_sample"],
                  max_new_tokens=d["max_new_tokens"])
        if long_form:
            kw.update(return_timestamps=True, temperature=d["temperature"],
                      condition_on_prev_tokens=c["long_form"]["condition_on_prev_tokens"])
        inputs = {k: v.to(self.device, dtype=getattr(torch, c["dtype"])) if k == "input_features" else v.to(self.device)
                  for k, v in feats.items()}
        with torch.inference_mode():
            ids = self.model.generate(**inputs, **kw)
        if isinstance(ids, dict) or hasattr(ids, "sequences"):
            ids = ids["sequences"]
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]

    def _run(self, feats: dict, long_form: bool) -> list[str]:
        try:
            return self._generate(feats, long_form)
        except (RuntimeError, NotImplementedError) as e:
            if self.device == "cpu":
                raise
            print(f"[asr] {self.device} failed ({type(e).__name__}: {str(e)[:120]}); falling back to cpu")
            self.model.to("cpu")
            self.device = "cpu"
            return self._generate(feats, long_form)

    def transcribe(self, wavs: list[np.ndarray]) -> list[str]:
        """Transcripts in input order; a batch may mix short and long clips."""
        sr = self.cfg["sample_rate"]
        out: list[str | None] = [None] * len(wavs)
        short = [i for i, w in enumerate(wavs) if len(w) <= self.max_short]
        if short:
            feats = self.processor.feature_extractor([wavs[i] for i in short], sampling_rate=sr, return_tensors="pt")
            for i, t in zip(short, self._run({"input_features": feats.input_features}, long_form=False)):
                out[i] = t
        for i, w in enumerate(wavs):
            if out[i] is None:
                f = self.processor.feature_extractor(w, sampling_rate=sr, truncation=False, padding="longest",
                                                     return_attention_mask=True, return_tensors="pt")
                out[i] = self._run({"input_features": f.input_features, "attention_mask": f.attention_mask},
                                   long_form=True)[0]
        return out  # type: ignore[return-value]


def read_wav(wav_path: str, sample_rate: int) -> np.ndarray:
    wav, sr = sf.read(ROOT / wav_path, dtype="float32", always_2d=False)
    assert sr == sample_rate, f"{wav_path}: {sr} Hz"
    return wav if wav.ndim == 1 else wav.mean(axis=1)


def transcribe_rows(tr: Transcriber, rows: pd.DataFrame, batch_size: int, cfg: dict, model_str: str,
                    on_batch=None) -> pd.DataFrame:
    """Transcribe ``rows`` (needs utt_id, wav_path, duration_s, gold_text); clips are batched
    by duration so decoder lengths in a batch are similar. Returns the ASR_COLUMNS frame."""
    order = rows.sort_values(["duration_s", "utt_id"]).reset_index(drop=True)
    out = []
    for s in range(0, len(order), batch_size):
        chunk = order.iloc[s:s + batch_size]
        wavs = [read_wav(p, cfg["sample_rate"]) for p in chunk["wav_path"]]
        texts = tr.transcribe(wavs)
        part = pd.DataFrame({
            "utt_id": chunk["utt_id"].to_numpy(),
            "asr_text": texts,
            "asr_model": model_str,
            "asr_flag": [hallucination_flags(t, g, clip_rms_dbfs(w), cfg)
                         for t, g, w in zip(texts, chunk["gold_text"], wavs)],
        })
        out.append(part)
        if on_batch:
            on_batch(part)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=ASR_COLUMNS)


def read_asr_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"asr_text": "string", "asr_flag": "string", "asr_model": "string"})
    df["asr_text"] = df["asr_text"].fillna("")
    df["asr_flag"] = df["asr_flag"].fillna("")
    return df


def run_split(tr: Transcriber, cfg: dict, tag: str, model_str: str, split: str, mdir: str | None,
              limit: int | None, batch_size: int) -> None:
    rows = load_manifest(split, require_audio=True, apply_exclusions=False, manifest_dir_override=mdir)
    if limit:
        rows = rows.head(limit)
    path = asr_path(tag, split, mdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = pd.DataFrame(columns=ASR_COLUMNS)
    if path.exists():
        done = read_asr_file(path)
        other = set(done["asr_model"].unique()) - {model_str}
        if other:
            raise RuntimeError(f"{path} was written with a different asr_model {sorted(other)}; "
                               f"remove the file to re-transcribe with {model_str!r}")
    todo = rows[~rows["utt_id"].isin(set(done["utt_id"]))]
    print(f"[{split}] {len(rows)} clips, {len(done)} already done, {len(todo)} to transcribe on {tr.device}")
    if todo.empty:
        return
    t0, n = time.time(), 0
    if not path.exists():
        pd.DataFrame(columns=ASR_COLUMNS).to_csv(path, index=False)

    def append(part: pd.DataFrame) -> None:
        nonlocal n
        part.to_csv(path, mode="a", header=False, index=False)
        n += len(part)
        if (n // batch_size) % 25 == 0:
            el = time.time() - t0
            print(f"[{split}] {n}/{len(todo)}  {n / el:.2f} clips/s  elapsed {el:.0f}s", flush=True)

    transcribe_rows(tr, todo, batch_size, cfg, model_str, on_batch=append)
    full = read_asr_file(path).drop_duplicates("utt_id", keep="last").sort_values("utt_id")
    full.to_csv(path, index=False)
    el = time.time() - t0
    print(f"[{split}] done: {n} clips in {el:.1f}s = {n / el:.2f} clips/s = {el / n:.2f} s/clip; "
          f"flagged {int((full['asr_flag'] != '').sum())}/{len(full)}; wrote {path}")


def main() -> None:
    cfg = load_yaml("configs/asr.yaml")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=[*SPLITS, "all"])
    ap.add_argument("--tag", default=cfg["tag"], choices=[cfg["tag"], cfg["final"]["tag"]])
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N rows of each split")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    a = ap.parse_args()

    tag, model_id = resolve_model(cfg, a.tag)
    set_seed(13)
    model_str = asr_model_string(cfg, model_id)
    print(f"asr_model: {model_str}")
    t0 = time.time()
    tr = Transcriber(cfg, model_id, a.device)
    print(f"model loaded on {tr.device} in {time.time() - t0:.1f}s")
    t1 = time.time()
    for split in (SPLITS if a.split == "all" else [a.split]):
        run_split(tr, cfg, tag, model_str, split, a.manifest_dir, a.limit, a.batch_size)
    print(f"total transcription time {time.time() - t1:.1f}s")


if __name__ == "__main__":
    main()
