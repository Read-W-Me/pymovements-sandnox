"""Shared helpers for validating the pipeline against dataset ground truth."""
from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

import numpy as np
import pandas as pd

from .dataset import Trial, load_ground_truth
from .evaluation import agreement_stats, relative_error
from .movement_classifier import (DetectionConfig, GazeColumns, ReadingConfig,
                                  ScreenConfig, process_real_gaze_data)

# (our key, ground-truth key, label, primary)
# "primary" metrics drive tuning. Gap-vs-saccade is informational only: I-DT gaps
# are not saccades, so a mismatch there says little about fixation quality.
METRICS = [
    ("fixation_count", "fixation_count", "Fixation count", True),
    ("fixation_dur_mean_ms", "fixation_dur_mean_ms", "Mean fixation duration (ms)", True),
    ("regression_count", "regression_count", "Regression count", True),
    ("gap_count", "saccade_count", "Inter-fixation gaps vs GT saccade count", False),
    ("gap_dur_mean_ms", "saccade_dur_mean_ms", "Mean gap vs GT saccade duration (ms)", False),
]


def _eval_one(args):
    trial, det, screen, reading, columns = args
    row = {"subject": trial.subject, "trial": trial.trial, "task": trial.task, "error": ""}
    try:
        ours = process_real_gaze_data(str(trial.raw_path), columns=columns, screen=screen,
                                      det=det, reading=reading)
        gt = load_ground_truth(trial.metrics_path) if trial.metrics_path else {}
        for our_key, gt_key, _, _ in METRICS:
            row[f"ours_{our_key}"] = ours[our_key]
            row[f"gt_{our_key}"] = gt.get(gt_key, float("nan"))
        row["tracking_loss_pct"] = ours["tracking_loss_pct"]
        row["sampling_rate_hz"] = ours["sampling_rate_hz"]
    except Exception as exc:  # keep going; failures are reported, not hidden
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=3)
    return row


def evaluate_trials(
    trials: Sequence[Trial],
    det: DetectionConfig = DetectionConfig(),
    screen: ScreenConfig = ScreenConfig(),
    reading: ReadingConfig = ReadingConfig(),
    columns: GazeColumns = GazeColumns(),
    n_jobs: int = 1,
) -> pd.DataFrame:
    jobs = [(t, det, screen, reading, columns) for t in trials]
    if n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            rows = list(pool.map(_eval_one, jobs))
    else:
        rows = [_eval_one(j) for j in jobs]
    return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> float:
    """Tuning objective: mean relative error over primary metrics (lower is better)."""
    errs = []
    for our_key, _, _, primary in METRICS:
        if primary and f"ours_{our_key}" in df:
            errs.append(relative_error(df[f"ours_{our_key}"], df[f"gt_{our_key}"]))
    if not errs:
        return float("nan")
    return float(np.nanmean(np.concatenate([np.asarray(e, float) for e in errs])))


def summarize(df: pd.DataFrame) -> dict:
    ok = df[df["error"] == ""] if "error" in df else df
    out = {"n_trials": int(len(df)), "n_failed": int(len(df) - len(ok)), "objective": score(ok) if len(ok) else float("nan")}
    for our_key, _, label, primary in METRICS:
        if f"ours_{our_key}" in ok:
            stats = agreement_stats(ok[f"ours_{our_key}"], ok[f"gt_{our_key}"])
            stats.update({"label": label, "primary": primary})
            out[our_key] = stats
    return out


def run_grid(
    trials: Sequence[Trial],
    dispersions: Sequence[float],
    durations: Sequence[float],
    base_det: DetectionConfig = DetectionConfig(),
    screen: ScreenConfig = ScreenConfig(),
    reading: ReadingConfig = ReadingConfig(),
    columns: GazeColumns = GazeColumns(),
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Evaluate every (dispersion, min-duration) pair on ``trials``.

    One row per pair: objective, failure count, mean relative error and mean
    value of each primary metric.
    """
    from dataclasses import replace

    rows = []
    for d in dispersions:
        for m in durations:
            det = replace(base_det, dispersion_threshold_deg=float(d), min_duration_ms=float(m))
            df = evaluate_trials(trials, det, screen, reading, columns, n_jobs)
            ok = df[df["error"] == ""]
            row = {"dispersion_deg": float(d), "min_duration_ms": float(m),
                   "n_trials": len(df), "n_failed": int(len(df) - len(ok)),
                   "objective": score(ok) if len(ok) else float("nan")}
            for our_key, _, _, primary in METRICS:
                if primary and len(ok):
                    row[f"mre_{our_key}"] = float(np.nanmean(
                        relative_error(ok[f"ours_{our_key}"], ok[f"gt_{our_key}"])))
                    row[f"mean_{our_key}"] = float(ok[f"ours_{our_key}"].mean())
            rows.append(row)
            if verbose:
                print(f"  dispersion={d:<4} min_dur={m:<5} objective={row['objective']:.4f}", flush=True)
    return pd.DataFrame(rows)
