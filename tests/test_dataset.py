"""Stage 7 of reports/preprocessing_validation.md: label mapping, item integrity, collation,
train-only transforms (Gate F6), class weights. Runs on the smoke manifests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import EMOTIONS, NUM_CLASSES  # noqa: E402
from src.data.dataset import class_weights, load_manifest  # noqa: E402
from src.data.torch_dataset import MELDDataset, collate, load_split_df, make_loader  # noqa: E402

SMOKE = ROOT / "data/processed/smoke_manifests"
pytestmark = pytest.mark.skipif(not (SMOKE / "train.csv").exists(), reason="smoke manifests missing")


def cfg(augment: bool = True) -> dict:
    return {
        "name": "t", "data": {"manifest_dir": str(SMOKE), "splits": {"train": "train", "dev": "dev"}, "text_source": "gold"},
        "model": {"type": "fusion_gated", "audio": {"mode": "finetune"}, "text": {}},
        "train": {"batch_size": 4, "max_audio_s": 2.0, "seed": 0,
                  "augment": {"enabled": augment, "speed": {"factors": [0.9, 1.1]}, "noise": {"snr_db": [10, 30]}}},
    }


@pytest.fixture(scope="module")
def train_df():
    return load_split_df("train", cfg())


def test_label_mapping_and_item_integrity(train_df):
    ds = MELDDataset(train_df, "train", cfg(augment=False), train=False, inputs={"text", "wav"})
    raw = load_manifest("train", manifest_dir_override=SMOKE)
    emotion_of = dict(zip(raw["utt_id"], raw["emotion"]))
    for i in range(len(ds)):
        item = ds[i]
        assert item["utt_id"] == train_df.utt_id[i]
        assert item["label"] == EMOTIONS.index(train_df.emotion[i])
        assert item["label"] == EMOTIONS.index(emotion_of[item["utt_id"]])
        assert item["text"] == train_df.text[i]
        assert item["wav"].dtype.is_floating_point and item["wav"].ndim == 1
        assert item["wav"].shape[0] <= ds.max_samples


def test_collate_shapes_and_masks(train_df):
    ds = MELDDataset(train_df, "train", cfg(augment=False), train=False, inputs={"text", "wav"})
    items = [ds[i] for i in range(4)]
    true_len = [it["wav"].shape[0] for it in items]
    b = collate(items, tokenizer=None, max_len=64)
    assert b["wav"].shape == (4, max(true_len))
    assert b["wav_mask"].shape == b["wav"].shape and b["wav_mask"].dtype == __import__("torch").bool
    assert b["wav_mask"].sum(1).tolist() == true_len
    assert b["wav_len"].tolist() == true_len
    assert b["label"].shape == (4,) and b["label"].dtype == __import__("torch").long
    assert set(b["label"].tolist()) <= set(range(NUM_CLASSES))
    assert b["utt_id"] == [it["utt_id"] for it in items]
    assert b["text"] == [it["text"] for it in items]
    # padded samples are zero
    for k, n in enumerate(true_len):
        assert float(b["wav"][k, n:].abs().sum()) == 0.0


def test_collate_with_tokenizer(train_df):
    tr = pytest.importorskip("transformers")
    tok = tr.AutoTokenizer.from_pretrained("roberta-base")
    ds = MELDDataset(train_df, "train", cfg(augment=False), train=False, inputs={"text"})
    b = collate([ds[i] for i in range(4)], tokenizer=tok, max_len=8)
    assert b["input_ids"].shape[0] == 4 and b["input_ids"].shape[1] <= 8
    assert b["attention_mask"].shape == b["input_ids"].shape
    assert "wav" not in b


def test_transforms_train_only(train_df):
    c = cfg(augment=True)
    tr = MELDDataset(train_df, "train", c, train=True, inputs={"wav"})
    dv = MELDDataset(load_split_df("dev", c), "dev", c, train=False, inputs={"wav"})
    assert len(tr.transforms) == 2
    assert dv.transforms == [] and dv.text_transforms == []
    _, dl_ds = make_loader("dev", c, train=False, batch_size=4)
    assert dl_ds.transforms == []
    with pytest.raises(AssertionError):
        make_loader("test", c, train=False)
    off = MELDDataset(train_df, "train", cfg(augment=False), train=True, inputs={"wav"})
    assert off.transforms == []


def test_crop_policy_and_counter(train_df):
    c = cfg(augment=False)
    c["train"]["max_audio_s"] = 1.0
    ds = MELDDataset(train_df, "train", c, train=False, inputs={"wav"})
    for i in range(len(ds)):
        assert ds[i]["wav"].shape[0] <= 16000
    expect = int((train_df["duration_s"] > 1.0).sum())
    assert ds.n_cropped == expect and ds.n_served_audio == len(ds)
    assert abs(ds.fraction_cropped - expect / len(ds)) < 1e-9


def test_class_weights_hand_computation(train_df):
    y = train_df["label"].to_numpy()
    counts = np.array([(y == c).sum() for c in range(NUM_CLASSES)], dtype=float)
    n = len(y)
    expect = np.sqrt(n / (NUM_CLASSES * np.maximum(counts, 1)))
    assert np.allclose(class_weights(y, "sqrt_inv"), expect)
    assert np.allclose(class_weights(y, "inv"), n / (NUM_CLASSES * np.maximum(counts, 1)))
    assert np.allclose(class_weights(y, "none"), np.ones(NUM_CLASSES))


def test_init_from_seed_template_substitution():
    from src.train import resolve_init_templates, substitute_init_from

    assert substitute_init_from("runs/audio_wavlm_cached/{seed}", 42) == "runs/audio_wavlm_cached/42"
    # a templated runs/ path follows --runs-root; an explicit path never does
    assert substitute_init_from("runs/a/{seed}", 7, "data/processed/smoke_runs") == "data/processed/smoke_runs/a/7"
    assert substitute_init_from("{runs_root}/a/{seed}", 7, "/abs/runs") == "/abs/runs/a/7"
    assert substitute_init_from("runs/a/13", 42, "data/processed/smoke_runs") == "runs/a/13"
    assert substitute_init_from(None, 1) is None
    cfg = {"model": {"type": "fusion_gated", "init_from": {"text": "runs/t/{seed}", "audio": "runs/a/{seed}"},
                     "audio": {"init_from": "runs/c/{seed}"}, "note": "{seed} untouched outside init_from"}}
    resolve_init_templates(cfg, 2024, "runs")
    assert cfg["model"]["init_from"] == {"text": "runs/t/2024", "audio": "runs/a/2024"}
    assert cfg["model"]["audio"]["init_from"] == "runs/c/2024"
    assert cfg["model"]["note"] == "{seed} untouched outside init_from"
    from src.train import init_from_paths

    assert sorted(init_from_paths(cfg)) == ["runs/a/2024", "runs/c/2024", "runs/t/2024"]  # values, never dict keys
    assert init_from_paths({"model": {"init_from": None}}) == []


def test_undersample_counts():
    import pandas as pd

    from src.train import undersample_df

    labels = [0] * 10 + [1] * 6 + [2] * 3 + [4] * 5      # classes 3, 5, 6 absent: they must not set n_min
    df = pd.DataFrame({"utt_id": [f"u{i}" for i in range(len(labels))], "label": labels})
    out, counts = undersample_df(df, seed=13)
    assert [counts[e] for e in EMOTIONS] == [3, 3, 3, 0, 3, 0, 0]
    assert len(out) == 12 and set(out["utt_id"]) <= set(df["utt_id"])
    out2, _ = undersample_df(df, seed=13)
    assert out.equals(out2)                               # seeded
    assert not out.equals(undersample_df(df, seed=14)[0])  # a different seed picks different rows
    assert list(out["utt_id"]) == sorted(out["utt_id"], key=lambda u: int(u[1:]))  # row order preserved
