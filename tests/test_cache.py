"""Cache isolation (Gate F5) and train-only fit metadata (Gate F3) tests.

Run: .venv/bin/python -m pytest tests/test_cache.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.cache import align, cache_path, load_cache, save_cache  # noqa: E402
from src.features.egemaps import apply_scaler, fit_scaler_arrays, load_scaler  # noqa: E402
from src.features.text_embed import fit_tfidf, load_tfidf, save_tfidf  # noqa: E402


def _save(tmp_path: Path, split: str = "dev", ids: list[str] | None = None, **kw) -> Path:
    ids = ids or [f"{split}_a", f"{split}_b", f"{split}_c"]
    X = np.arange(len(ids) * 2, dtype=np.float32).reshape(len(ids), 2)
    path = cache_path("toy", "v1", split, tmp_path)
    return save_cache(path, split=split, extractor="toy", version="v1", utt_ids=ids, X=X, **kw)


def test_roundtrip_with_extra_arrays(tmp_path):
    p = _save(tmp_path, flag=np.array([True, False, True]))
    c = load_cache(p, split="dev", extractor="toy", version="v1")
    assert c["utt_id"].tolist() == ["dev_a", "dev_b", "dev_c"]
    assert c["X"].shape == (3, 2)
    assert c["flag"].tolist() == [True, False, True]
    assert c["meta"] == {"split": "dev", "extractor": "toy", "version": "v1"}


@pytest.mark.parametrize("wrong", ["train", "test"])
def test_wrong_split_raises(tmp_path, wrong):
    p = _save(tmp_path, "dev")
    with pytest.raises(AssertionError):
        load_cache(p, split=wrong)


def test_wrong_extractor_or_version_raises(tmp_path):
    p = _save(tmp_path)
    with pytest.raises(AssertionError):
        load_cache(p, split="dev", extractor="other")
    with pytest.raises(AssertionError):
        load_cache(p, split="dev", version="v2")


def test_duplicate_ids_raise(tmp_path):
    with pytest.raises(AssertionError):
        _save(tmp_path, ids=["dev_a", "dev_a"])


def test_foreign_split_id_raises(tmp_path):
    with pytest.raises(AssertionError):
        _save(tmp_path, "dev", ids=["dev_a", "train_b"])


def test_length_mismatch_raises(tmp_path):
    with pytest.raises(AssertionError):
        save_cache(cache_path("toy", "v1", "dev", tmp_path), split="dev", extractor="toy", version="v1",
                   utt_ids=["dev_a", "dev_b"], X=np.zeros((3, 2), dtype=np.float32))


def test_align_order_and_missing(tmp_path):
    c = load_cache(_save(tmp_path), split="dev")
    out = align(c, ["dev_c", "dev_a"])
    np.testing.assert_array_equal(out, c["X"][[2, 0]])
    with pytest.raises(KeyError):
        align(c, ["dev_a", "dev_zzz"])


def test_load_scaler_requires_train(tmp_path):
    X = np.random.default_rng(0).normal(3.0, 2.0, size=(20, 4)).astype(np.float32)
    mean, std = fit_scaler_arrays(X)
    for split, ok in (("train", True), ("dev", False), ("test", False)):
        p = tmp_path / f"scaler_{split}.pkl"
        p.write_bytes(pickle.dumps({"mean": mean, "std": std, "fit_split": split, "n_fit": 20}))
        if ok:
            sc = load_scaler(p)
            Z = apply_scaler(X, sc)
            assert np.allclose(Z.mean(axis=0), 0, atol=1e-5) and np.allclose(Z.std(axis=0), 1, atol=1e-4)
        else:
            with pytest.raises(AssertionError):
                load_scaler(p)


def test_scaler_constant_column_does_not_divide_by_zero():
    X = np.ones((5, 2), dtype=np.float32)
    mean, std = fit_scaler_arrays(X)
    assert (std == 1.0).all()


def test_load_tfidf_requires_train(tmp_path):
    texts = ["I don't know", "I don't care", "oh my god", "oh my god what"]
    vec = fit_tfidf(texts)
    assert "don't" in vec.vocabulary_
    p = save_tfidf(tmp_path / "v.pkl", vec, text_source="gold", asr_tag=None, n_fit=len(texts))
    assert load_tfidf(p)["fit_split"] == "train"
    obj = pickle.loads(p.read_bytes())
    for split in ("dev", "test"):
        obj["fit_split"] = split
        bad = tmp_path / f"bad_{split}.pkl"
        bad.write_bytes(pickle.dumps(obj))
        with pytest.raises(AssertionError):
            load_tfidf(bad)
