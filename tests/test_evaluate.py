"""Pure functions of src/evaluate.py (Gate 5): metrics against scikit-learn, bootstrap CIs,
McNemar against known values, late-fusion weight search, the dev-based class-weight table
(M7) and the fusion auto-selection on fake run directories."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import NUM_CLASSES  # noqa: E402
from src.evaluate import (_cm, _f1_from_cm, bootstrap_ci, class_weight_table, load_dev_results, load_run_results,  # noqa: E402
                          mcnemar, mcnemar_table, metrics_dict, rejected_families, select_fusion, transcript_gate)


def _data(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, NUM_CLASSES, n)
    pred = np.where(rng.random(n) < 0.6, y, rng.integers(0, NUM_CLASSES, n))
    return y, pred


def test_f1_from_cm_matches_sklearn():
    y, pred = _data()
    f1, w, m, acc = _f1_from_cm(_cm(y, pred))
    labels = list(range(NUM_CLASSES))
    assert np.allclose(f1, f1_score(y, pred, labels=labels, average=None, zero_division=0))
    assert w == pytest.approx(f1_score(y, pred, labels=labels, average="weighted", zero_division=0))
    assert m == pytest.approx(f1_score(y, pred, labels=labels, average="macro", zero_division=0))
    assert acc == pytest.approx((y == pred).mean())


def test_metrics_dict_fields():
    y, pred = _data(1)
    d = metrics_dict(y, pred)
    assert set(d["per_class"]) == {"neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"}
    rows = np.asarray(d["confusion_row_norm"])
    assert rows.shape == (NUM_CLASSES, NUM_CLASSES) and np.allclose(rows.sum(1), 1.0, atol=1e-3)
    assert d["fear_recall"] == pytest.approx(d["per_class"]["fear"]["recall"])
    assert d["disgust_recall"] == pytest.approx(d["per_class"]["disgust"]["recall"])


def test_bootstrap_ci_brackets_point_and_is_seeded():
    y, pred = _data(2)
    a = bootstrap_ci(y, pred, "weighted_f1", n=200, seed=3)
    b = bootstrap_ci(y, pred, "weighted_f1", n=200, seed=3)
    assert a == b
    assert a["lo"] <= a["point"] <= a["hi"]
    c = bootstrap_ci(y, pred, lambda yy, pp: float((yy == pp).mean()), n=50, seed=0)
    assert c["lo"] <= c["point"] <= c["hi"]


def test_mcnemar_known_values():
    # 10 discordant pairs split 8/2 -> exact two-sided p = 2 * P(X <= 2 | n=10, 0.5) = 0.109375
    a = np.array([True] * 8 + [False] * 2 + [True] * 20)
    b = np.array([False] * 8 + [True] * 2 + [True] * 20)
    r = mcnemar(a, b)
    assert (r["n01"], r["n10"]) == (8, 2)
    assert r["method"] == "exact" and r["p_value"] == pytest.approx(0.109375)
    # 40 discordant pairs split 30/10 -> chi2 with continuity correction (|30-10|-1)^2/40 = 9.025
    a = np.array([True] * 30 + [False] * 10)
    b = np.array([False] * 30 + [True] * 10)
    r = mcnemar(a, b)
    assert r["method"] == "chi2_cc" and r["statistic"] == pytest.approx(9.025)
    assert 0.002 < r["p_value"] < 0.003
    assert mcnemar(a, a)["p_value"] == 1.0


def test_late_fusion_probs_rows_sum_to_one():
    from src.models.fusion import late_fusion_probs

    rng = np.random.default_rng(0)
    pa = rng.dirichlet(np.ones(NUM_CLASSES), 5)
    pt = rng.dirichlet(np.ones(NUM_CLASSES), 5)
    p = late_fusion_probs(pa, pt, 0.3)
    assert np.allclose(p.sum(1), 1.0)
    assert np.allclose(late_fusion_probs(pa, pt, 1.0), pa)


# ----------------------------------------------------------------------------- fake run dirs

def _fake_run(root: Path, name: str, seed: int, model_type: str, dev_w: float, dev_m: float, scheme: str = "sqrt_inv",
              loss: str = "ce", test: dict | None = None, preds: np.ndarray | None = None, labels: np.ndarray | None = None) -> Path:
    """``<root>/<name>/<seed>`` with config.yaml + metrics.json (dev scores only, like a
    training run); optional test_metrics.json / test_predictions.csv (like ``evaluate run``)."""
    run = root / name / str(seed)
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"name": name, "model": {"type": model_type}, "data": {}, "train": {}}))
    (run / "metrics.json").write_text(json.dumps({"name": name, "seed": seed, "class_weight_scheme": scheme, "loss": loss,
                                                   "best": {"dev_weighted_f1": dev_w, "dev_macro_f1": dev_m, "train_weighted_f1": dev_w + 0.1}}))
    if test is not None:
        (run / "test_metrics.json").write_text(json.dumps({"name": name, "seed": seed, "model_type": model_type, "n_test_evaluations": 1,
                                                            "test": test, "clean_subset": {}}))
    if preds is not None:
        pd.DataFrame({"utt_id": [f"u{i}" for i in range(len(preds))], "label": labels, "pred": preds}).to_csv(run / "test_predictions.csv", index=False)
    return run


def test_class_weight_table_from_dev_scores_with_rejection(tmp_path):
    # text_roberta (sqrt_inv) has the higher weighted F1 but a LOWER macro F1 than its unweighted sibling -> REJECTED
    _fake_run(tmp_path, "text_roberta", 13, "text_roberta", dev_w=0.60, dev_m=0.30)
    _fake_run(tmp_path, "text_roberta_cwnone", 13, "text_roberta", dev_w=0.58, dev_m=0.35, scheme="none")
    _fake_run(tmp_path, "text_roberta_cwinv", 13, "text_roberta", dev_w=0.55, dev_m=0.36, scheme="inv")
    # a family without ablation arms must not appear; a _gold twin is a transcript condition, not an arm
    _fake_run(tmp_path, "audio_wavlm_ft", 13, "audio_wavlm", dev_w=0.40, dev_m=0.20)
    _fake_run(tmp_path, "text_roberta_gold", 13, "text_roberta", dev_w=0.62, dev_m=0.31)
    dev = load_dev_results(tmp_path)
    assert len(dev) == 5 and not dev["has_test"].any()
    cw = class_weight_table(dev, pd.DataFrame())
    assert cw["name"].tolist() == ["text_roberta", "text_roberta_cwnone", "text_roberta_cwinv"]
    col = "M7 rule (dev mF1 >= _cwnone)"
    assert dict(zip(cw["name"], cw[col])) == {"text_roberta": "REJECTED", "text_roberta_cwnone": "baseline (unweighted CE)", "text_roberta_cwinv": "ok"}
    assert cw.loc[cw["name"] == "text_roberta", "dev_weighted_f1"].iloc[0] == pytest.approx(0.60)
    assert cw["test_weighted_f1"].isna().all()  # no arm was test-evaluated
    assert rejected_families(cw) == {"text_roberta"}


def test_select_fusion_auto_and_mcnemar_for_every_family(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, NUM_CLASSES, 40)
    tm = {"weighted_f1": 0.5, "macro_f1": 0.4, "fear_recall": 0.1, "disgust_recall": 0.2, "n": 40,
          "per_class": {}, "classes_with_zero_recall": []}
    _fake_run(tmp_path, "text_roberta", 13, "text_roberta", 0.55, 0.30, test=tm, preds=np.where(rng.random(40) < 0.5, y, (y + 1) % NUM_CLASSES), labels=y)
    _fake_run(tmp_path, "fusion_gated", 13, "fusion_gated", 0.58, 0.33, test=tm, preds=np.where(rng.random(40) < 0.6, y, (y + 2) % NUM_CLASSES), labels=y)
    _fake_run(tmp_path, "fusion_xattn", 13, "fusion_xattn", 0.61, 0.35, test=tm, preds=np.where(rng.random(40) < 0.6, y, (y + 3) % NUM_CLASSES), labels=y)
    _fake_run(tmp_path, "fusion_xattn_cwnone", 13, "fusion_xattn", 0.59, 0.36, scheme="none")  # trained, never test-evaluated
    _fake_run(tmp_path, "fusion_best_untested", 13, "fusion_gated", 0.90, 0.80)  # no test_metrics.json -> not a candidate
    dev, test = load_dev_results(tmp_path), load_run_results(tmp_path)
    assert len(test) == 3
    name, tab, reason = select_fusion(dev, test, "auto", set())
    assert name == "fusion_xattn" and tab["name"].tolist() == ["fusion_xattn", "fusion_gated"] and "auto-selected" in reason
    # the M7 rule: fusion_xattn's dev macro F1 (0.35) is below its _cwnone sibling's (0.36) -> rejected -> fusion_gated wins
    rejected = rejected_families(class_weight_table(dev, test))
    assert rejected == {"fusion_xattn"}
    name2, tab2, reason2 = select_fusion(dev, test, "auto", rejected)
    assert name2 == "fusion_gated" and "skipped (M7 REJECTED): fusion_xattn" in reason2
    assert dict(zip(tab2["name"], tab2["M7"])) == {"fusion_xattn": "REJECTED", "fusion_gated": "ok"}
    assert select_fusion(dev, test, "fusion_gated", set())[0] == "fusion_gated"  # explicit override
    mc, best = mcnemar_table(test, tmp_path, ["fusion_gated", "fusion_xattn"], ["text_roberta"])
    assert best == "text_roberta" and sorted(mc["fusion"]) == ["fusion_gated", "fusion_xattn"]
    assert ((mc["p_value"] >= 0) & (mc["p_value"] <= 1)).all()


def test_transcript_gate_requires_text_and_fusion_families():
    rows = [("text_roberta", tr, te) for tr in ("gold", "asr") for te in ("gold", "asr")] + [("fusion_gated", "asr", "gold"), ("fusion_gated", "asr", "asr")]
    tt = pd.DataFrame(rows, columns=["family", "train", "test"]).assign(weighted_f1=0.5, macro_f1=0.4, seeds=1, run="x")
    gid, status, ev = transcript_gate(tt, "text_roberta", "fusion_gated")
    assert (gid, status) == ("V5", "PARTIAL") and "fusion_gated missing ['train gold/test asr', 'train gold/test gold']" in ev
    full = pd.concat([tt, pd.DataFrame([("fusion_gated", "gold", "gold"), ("fusion_gated", "gold", "asr")], columns=["family", "train", "test"]).assign(weighted_f1=0.5, macro_f1=0.4, seeds=1, run="x")])
    assert transcript_gate(full, "text_roberta", "fusion_gated")[1] == "PASS"
    assert transcript_gate(pd.DataFrame(), "text_roberta", "fusion_gated")[1] == "SKIP"
