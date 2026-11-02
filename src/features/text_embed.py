"""Text features: train-only TF-IDF and RoBERTa tokenisation helpers.

Usage: python -m src.features.text_embed --fit-tfidf --text-source {gold,asr} [--asr-tag whisper-small]
           [--manifest-dir DIR] [--feature-root DIR]

Fits the TF-IDF vectoriser on the non-excluded train rows (``text`` column of
``load_manifest``, already passed through ``src.common.normalize_text``), saves
``<feature-root>/tfidf_<text_source>_<asr_tag or gold>/vectorizer.pkl`` and prints the
RoBERTa token-length report for train and dev (Stage 4: 64 tokens must cover 99 % of rows).
RoBERTa embeddings are computed inside the model at training time and are not cached.
"""
from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common import ROOT
from src.data.dataset import load_manifest
from src.features.cache import FEATURE_ROOT

TOKENIZER_NAME = "roberta-base"
MAX_LEN = 64
DEFAULT_TFIDF_CFG: dict = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "lowercase": True,
    "strip_accents": None,
    "token_pattern": r"(?u)\b\w+(?:'\w+)*\b",
}


def tfidf_dir(feature_root: Path, text_source: str, asr_tag: str | None) -> Path:
    return feature_root / f"tfidf_{text_source}_{asr_tag if text_source == 'asr' else 'gold'}"


def fit_tfidf(train_texts: list[str], cfg: dict | None = None) -> TfidfVectorizer:
    params = {**DEFAULT_TFIDF_CFG, **(cfg or {})}
    vec = TfidfVectorizer(**params)
    vec.fit(list(train_texts))
    return vec


def save_tfidf(path: Path, vec: TfidfVectorizer, *, text_source: str, asr_tag: str | None,
               n_fit: int, params: dict | None = None) -> Path:
    obj = {
        "vectorizer": vec,
        "fit_split": "train",
        "text_source": text_source,
        "asr_tag": asr_tag,
        "n_fit": int(n_fit),
        "params": {**DEFAULT_TFIDF_CFG, **(params or {})},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_tfidf(path: str | Path) -> dict:
    """Load vectorizer.pkl (dict with ``vectorizer`` and metadata); asserts fit_split == 'train'."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    assert obj.get("fit_split") == "train", f"{path}: fit_split={obj.get('fit_split')!r}, expected 'train'"
    return obj


def train_texts(text_source: str, asr_tag: str | None, manifest_dir: str | None) -> list[str]:
    """Texts of the non-excluded train rows (audio availability is not required for text)."""
    df = load_manifest("train", require_audio=False, apply_exclusions=True, asr_tag=asr_tag,
                       text_source=text_source, manifest_dir_override=manifest_dir)
    return df["text"].tolist()


def get_tokenizer(name: str = TOKENIZER_NAME):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def tokenize(texts: list[str], tokenizer, max_len: int = MAX_LEN) -> dict:
    """Padded and truncated batch as torch tensors (``input_ids``, ``attention_mask``)."""
    return tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")


def token_length_report(texts: list[str], tokenizer, max_len: int = MAX_LEN,
                        utt_ids: list[str] | None = None) -> dict:
    """Token counts including special tokens. ``frac_fit`` is the share of rows with count <= max_len."""
    enc = tokenizer(list(texts), truncation=False, add_special_tokens=True)
    lens = np.array([len(x) for x in enc["input_ids"]])
    over = np.flatnonzero(lens > max_len)
    ids = utt_ids if utt_ids is not None else [str(i) for i in range(len(texts))]
    return {
        "n": int(len(lens)),
        "max_len": max_len,
        "frac_fit": float((lens <= max_len).mean()) if len(lens) else float("nan"),
        "n_truncated": int(len(over)),
        "truncated_utt_ids": [ids[i] for i in over],
        "mean_tokens": float(lens.mean()) if len(lens) else float("nan"),
        "p99_tokens": float(np.percentile(lens, 99)) if len(lens) else float("nan"),
        "max_tokens": int(lens.max()) if len(lens) else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit-tfidf", action="store_true")
    ap.add_argument("--text-source", required=True, choices=["gold", "asr"])
    ap.add_argument("--asr-tag", default=None)
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--feature-root", default=str(FEATURE_ROOT))
    a = ap.parse_args()
    if a.text_source == "asr" and not a.asr_tag:
        ap.error("--text-source asr needs --asr-tag")
    asr_tag = a.asr_tag if a.text_source == "asr" else None
    feature_root = Path(a.feature_root)
    if not feature_root.is_absolute():
        feature_root = ROOT / feature_root

    if a.fit_tfidf:
        texts = train_texts(a.text_source, asr_tag, a.manifest_dir)
        vec = fit_tfidf(texts)
        path = save_tfidf(tfidf_dir(feature_root, a.text_source, asr_tag) / "vectorizer.pkl", vec,
                          text_source=a.text_source, asr_tag=asr_tag, n_fit=len(texts))
        obj = load_tfidf(path)
        print(f"tfidf: {path} (fit_split={obj['fit_split']}, n_fit={obj['n_fit']}, "
              f"vocab={len(vec.vocabulary_)})")

    tok = get_tokenizer()
    for split in ("train", "dev"):
        df = load_manifest(split, require_audio=False, apply_exclusions=True, asr_tag=asr_tag,
                           text_source=a.text_source, manifest_dir_override=a.manifest_dir)
        rep = token_length_report(df["text"].tolist(), tok, MAX_LEN, df["utt_id"].tolist())
        status = "PASS" if rep["frac_fit"] >= 0.99 else "FAIL"
        print(f"{split}: {rep['frac_fit']:.4f} of {rep['n']} rows fit in {MAX_LEN} tokens "
              f"(mean {rep['mean_tokens']:.1f}, p99 {rep['p99_tokens']:.0f}, max {rep['max_tokens']}); "
              f"{rep['n_truncated']} truncated {status}")
        if rep["truncated_utt_ids"]:
            print(f"{split}: truncated utt_ids: {rep['truncated_utt_ids']}")


if __name__ == "__main__":
    main()
