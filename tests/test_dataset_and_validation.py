import zipfile

import numpy as np
import pandas as pd
import pytest

from src.dataset import discover_trials, extract_zip_safely, load_ground_truth
from src.evaluation import split_subjects
from src.movement_classifier import DetectionConfig
from src.validation import evaluate_trials, score, summarize
from tests.synthetic import EASY_POINTS, synth_scanpath


def _write_trial(root, subject, trial="T5", task="Pseudo_Text", with_metrics=True, sep=","):
    df = synth_scanpath(EASY_POINTS, fix_ms=400, seed=int(subject))
    stem = f"Subject_{subject}_{trial}_{task}"
    df.to_csv(root / f"{stem}_raw.csv", index=False)
    if with_metrics:
        gt = pd.DataFrame({"n_fix_trial": [5, 5], "mean_fix_dur_trial": [400.0, 400.0],
                           "n_regress_trial": [0, 0], "n_sacc_trial": [4, 4],
                           "mean_sacc_dur_trial": [12.0, 12.0], "aoi": ["a", "b"]})
        gt.columns = [c.upper() for c in gt.columns]  # loader must lower-case
        gt.to_csv(root / f"{stem}_metrics.csv", index=False, sep=sep)


def test_discover_and_ground_truth(tmp_path):
    _write_trial(tmp_path, "1001")
    _write_trial(tmp_path, "1002", task="Other_Task")
    _write_trial(tmp_path, "1003", with_metrics=False)
    (tmp_path / "notes.txt").write_text("ignore me")
    assert len(discover_trials(tmp_path)) == 2                       # 1003 lacks metrics
    assert len(discover_trials(tmp_path, require_metrics=False)) == 3
    only = discover_trials(tmp_path, tasks=["pseudo_text"])
    assert [t.subject for t in only] == ["1001"] and only[0].trial == "T5"
    gt = load_ground_truth(only[0].metrics_path)
    assert gt["fixation_count"] == 5 and gt["fixation_dur_mean_ms"] == 400.0


def test_ground_truth_semicolon_separator(tmp_path):
    _write_trial(tmp_path, "1001", sep=";")
    gt = load_ground_truth(discover_trials(tmp_path)[0].metrics_path)
    assert gt["regression_count"] == 0 and gt["saccade_count"] == 4


def test_extract_zip_safely(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("a/b.txt", "hi")
    assert extract_zip_safely(good, tmp_path / "out") == 1
    assert (tmp_path / "out" / "a" / "b.txt").read_text() == "hi"
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../escape.txt", "no")
    with pytest.raises(ValueError):
        extract_zip_safely(evil, tmp_path / "out2")
    assert not (tmp_path / "escape.txt").exists()


def test_end_to_end_validation_on_synthetic_trials(tmp_path):
    for s in ("1001", "1002", "1003", "1004"):
        _write_trial(tmp_path, s)
    trials = discover_trials(tmp_path)
    train, test = split_subjects([t.subject for t in trials], 0.5, seed=0)
    df = evaluate_trials([t for t in trials if t.subject in train])
    assert (df["error"] == "").all() and len(df) == 2
    assert (df["ours_fixation_count"] == 5).all()
    assert score(df) < 0.05
    summ = summarize(df)
    assert summ["n_failed"] == 0 and summ["fixation_count"]["mae"] == 0

    # a badly wrong threshold must score worse than the good one
    bad = evaluate_trials(trials, det=DetectionConfig(dispersion_threshold_deg=0.1))
    assert score(bad) > score(evaluate_trials(trials))


def test_failures_are_reported_not_raised(tmp_path):
    _write_trial(tmp_path, "1001")
    trial = discover_trials(tmp_path)[0]
    pd.DataFrame({"foo": [1, 2, 3]}).to_csv(trial.raw_path, index=False)  # wrong columns
    df = evaluate_trials([trial])
    assert df.loc[0, "error"].startswith("KeyError")
    assert summarize(df)["n_failed"] == 1
