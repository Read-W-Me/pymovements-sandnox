import numpy as np
import pytest

from src.evaluation import (agreement_stats, cohens_kappa, match_events, relative_error,
                            sample_labels, split_subjects)


def test_agreement_stats_known_values():
    s = agreement_stats([10, 20, 30], [12, 18, 33])
    assert s["n"] == 3
    assert s["mae"] == pytest.approx((2 + 2 + 3) / 3)
    assert s["bias"] == pytest.approx((-2 + 2 - 3) / 3)
    assert s["loa_low"] < s["bias"] < s["loa_high"]
    assert s["pearson_r"] > 0.95


def test_agreement_stats_handles_nan_and_empty():
    assert agreement_stats([1, np.nan, 3], [1, 2, 3])["n"] == 2
    s = agreement_stats([], [])
    assert s["n"] == 0 and np.isnan(s["mae"])
    assert np.isnan(agreement_stats([1, 1, 1], [1, 2, 3])["pearson_r"])  # constant input


def test_relative_error_floor():
    assert relative_error(3, 0) == pytest.approx(3.0)
    assert relative_error(11, 10) == pytest.approx(0.1)


def test_match_events_perfect_shifted_and_missing():
    gt = [(0, 100), (200, 300), (400, 500)]
    assert match_events(gt, gt)["f1"] == 1.0
    shifted = [(10, 110), (210, 310), (410, 510)]
    r = match_events(shifted, gt, min_iou=0.5)
    assert r["tp"] == 3 and r["mean_iou"] == pytest.approx(90 / 110)
    r = match_events(shifted, gt, min_iou=0.95)
    assert r["tp"] == 0 and r["fp"] == 3 and r["fn"] == 3 and r["f1"] == 0.0
    r = match_events([(0, 100)], gt)
    assert r["tp"] == 1 and r["fn"] == 2 and r["recall"] == pytest.approx(1 / 3)
    assert np.isnan(match_events([], [])["f1"])


def test_match_events_is_one_to_one():
    r = match_events([(0, 100), (5, 105)], [(0, 100)])
    assert r["tp"] == 1 and r["fp"] == 1


def test_sample_labels_and_kappa():
    a = sample_labels([(0, 9), (20, 29)], 40)
    assert a.sum() == 20
    assert cohens_kappa(a, a) == 1.0
    assert cohens_kappa(a, ~a) < 0
    rng = np.random.default_rng(0)
    x, y = rng.random(5000) < 0.5, rng.random(5000) < 0.5
    assert abs(cohens_kappa(x, y)) < 0.05
    with pytest.raises(ValueError):
        cohens_kappa(np.array([True]), np.array([True, False]))


def test_split_subjects_disjoint_and_deterministic():
    subs = [str(i) for i in range(10)] * 3  # repeated ids collapse
    tr, te = split_subjects(subs, 0.4, seed=1)
    assert set(tr).isdisjoint(te) and len(tr) + len(te) == 10 and len(te) == 4
    assert (tr, te) == split_subjects(subs, 0.4, seed=1)
    assert (tr, te) != split_subjects(subs, 0.4, seed=2)
    with pytest.raises(ValueError):
        split_subjects(["a"], 0.5)
    assert len(split_subjects(["a", "b"], 0.99)[1]) == 1
