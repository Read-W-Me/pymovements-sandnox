"""Gaze -> reading-parameter pipeline for Read W/Me (I-DT fixation detection).

What changed relative to the first version
------------------------------------------
* Timestamps are respected: samples are placed on a uniform time grid, so dropped
  frames become NaN gaps instead of silently shortening durations.
* Both eyes are used when available (mean of the valid eyes per sample).
* Only *short* gaps (default <= 75 ms) are interpolated. Longer gaps (blinks,
  tracking loss) stay NaN and split fixations.
* Screen geometry, thresholds and line height are explicit config objects
  instead of hard-coded numbers.
* I-DT only produces fixations. The "saccade" numbers of the old version were
  just the gaps between consecutive fixations (count == fixations - 1 by
  construction), so they are now reported honestly as ``gap_*`` metrics and the
  tautological saccade/fixation ratio was removed.
* Regression logic no longer counts return sweeps as regressions when the
  configured line height is larger than the real line spacing.
* AOI ``revisit_count`` no longer under-counts when the first fixation is
  already inside the AOI.

pymovements 0.28 notes (verified against its source)
----------------------------------------------------
* I-DT dispersion is (x-range + y-range) of the window in degrees, not the
  Euclidean diameter, so ``dispersion_threshold_deg`` is on that scale.
* The window is expanded until dispersion reaches the threshold and the sample
  that broke it is included, so each fixation ends one sample late. At 250 Hz
  that is 4 ms; durations here follow pymovements (offset - onset).
* Timesteps must be integers with a constant interval, which is why the
  uniform-grid step below is needed.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pymovements as pm

EVENT_COLUMNS = ["name", "onset", "offset", "duration"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScreenConfig:
    """Screen geometry. Defaults are carried over from the original script and
    are NOT verified against the real ETDD70 setup; check them before trusting
    degree-based thresholds."""

    width_px: int = 1920
    height_px: int = 1080
    width_cm: float = 53.0
    height_cm: float = 30.0
    distance_cm: float = 60.0


@dataclass(frozen=True)
class DetectionConfig:
    dispersion_threshold_deg: float = 2.2
    min_duration_ms: float = 100.0
    max_interp_gap_ms: float = 75.0  # 0 disables interpolation
    zero_is_invalid: bool = True  # exact 0 px is treated as tracker loss


@dataclass(frozen=True)
class ReadingConfig:
    line_height_px: float = 40.0  # set to the real line spacing of the stimulus
    regression_dx_px: float = 15.0  # min leftward step to count as regression
    same_line_fraction: float = 0.5  # |dy| <= fraction * line height => same line


@dataclass(frozen=True)
class GazeColumns:
    time: str = "time"
    time_unit: str = "us"  # "us", "ms" or "s"
    x: Sequence[str] = ("gaze_x_left", "gaze_x_right")
    y: Sequence[str] = ("gaze_y_left", "gaze_y_right")


_TIME_TO_MS = {"us": 1e-3, "ms": 1.0, "s": 1e3}


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
@dataclass
class GazeSeries:
    """Gaze on a uniform time grid (NaN = missing)."""

    x: np.ndarray
    y: np.ndarray
    sampling_rate_hz: float
    n_dropped_samples: int
    usable_sample_pct: float  # before interpolation, dropped frames count as loss

    @property
    def period_ms(self) -> float:
        return 1000.0 / self.sampling_rate_hz


def min_duration_samples(min_duration_ms: float, sampling_rate_hz: float) -> int:
    """Minimum duration in samples (rounded up, at least 2, float-safe)."""
    period = 1000.0 / sampling_rate_hz
    return max(2, math.ceil(min_duration_ms / period - 1e-9))


def fill_short_gaps(x: np.ndarray, y: np.ndarray, max_gap_samples: int):
    """Linearly interpolate NaN runs of <= max_gap_samples bounded by valid data.

    Longer runs and runs touching the start/end of the recording stay NaN.
    """
    x = np.array(x, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)
    if max_gap_samples <= 0 or len(x) == 0:
        return x, y
    missing = np.isnan(x) | np.isnan(y)
    d = np.diff(np.concatenate(([0], missing.astype(int), [0])))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):  # e exclusive
        if s == 0 or e == len(x) or (e - s) > max_gap_samples:
            continue
        idx = np.arange(s, e)
        x[s:e] = np.interp(idx, [s - 1, e], [x[s - 1], x[e]])
        y[s:e] = np.interp(idx, [s - 1, e], [y[s - 1], y[e]])
    return x, y


def prepare_gaze(
    df: pd.DataFrame,
    columns: GazeColumns = GazeColumns(),
    det: DetectionConfig = DetectionConfig(),
) -> GazeSeries:
    """Turn a raw gaze table into a uniform-grid GazeSeries."""
    if columns.time not in df.columns:
        raise KeyError(f"time column '{columns.time}' not found")
    if columns.time_unit not in _TIME_TO_MS:
        raise ValueError(f"time_unit must be one of {sorted(_TIME_TO_MS)}")

    df = df.copy()
    df["_t_ms"] = pd.to_numeric(df[columns.time], errors="coerce") * _TIME_TO_MS[columns.time_unit]
    df = df.dropna(subset=["_t_ms"]).sort_values("_t_ms").drop_duplicates("_t_ms")
    if len(df) < 2:
        raise ValueError("need at least 2 samples with valid timestamps")

    t_ms = df["_t_ms"].to_numpy(float)
    period = float(np.median(np.diff(t_ms)))
    if period <= 0:
        raise ValueError("non-positive median sample interval")

    # Per-eye validity, then mean of the valid eyes.
    xs, ys = [], []
    for xc, yc in zip(columns.x, columns.y):
        if xc in df.columns and yc in df.columns:
            ex = pd.to_numeric(df[xc], errors="coerce").to_numpy(float)
            ey = pd.to_numeric(df[yc], errors="coerce").to_numpy(float)
            bad = np.isnan(ex) | np.isnan(ey)
            if det.zero_is_invalid:
                bad |= (ex == 0) | (ey == 0)
            xs.append(np.where(bad, np.nan, ex))
            ys.append(np.where(bad, np.nan, ey))
    if not xs:
        raise KeyError(f"none of the gaze columns {list(columns.x)}/{list(columns.y)} found")
    xs, ys = np.vstack(xs), np.vstack(ys)
    n_valid = (~np.isnan(xs)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        x_obs = np.where(n_valid > 0, np.nansum(xs, axis=0) / n_valid, np.nan)
        y_obs = np.where(n_valid > 0, np.nansum(ys, axis=0) / n_valid, np.nan)

    # Place samples on a uniform grid; missing frames stay NaN.
    idx = np.rint((t_ms - t_ms[0]) / period).astype(int)
    idx, first = np.unique(idx, return_index=True)
    x = np.full(idx[-1] + 1, np.nan)
    y = np.full(idx[-1] + 1, np.nan)
    x[idx], y[idx] = x_obs[first], y_obs[first]
    dropped = len(x) - len(idx)

    usable_pct = float((~np.isnan(x)).mean() * 100.0)
    max_gap = int(det.max_interp_gap_ms // period)
    x, y = fill_short_gaps(x, y, max_gap)
    return GazeSeries(x, y, 1000.0 / period, int(dropped), usable_pct)


def load_gaze_csv(filepath: str, columns: GazeColumns = GazeColumns(),
                  det: DetectionConfig = DetectionConfig()) -> GazeSeries:
    return prepare_gaze(pd.read_csv(filepath), columns, det)


# --------------------------------------------------------------------------- #
# Fixation detection
# --------------------------------------------------------------------------- #
def add_centroids(events: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Vectorised mean gaze position (px) over each event's samples."""
    ev = events.copy()
    if ev.empty:
        ev["centroid_x"], ev["centroid_y"] = [], []
        return ev
    valid = ~(np.isnan(x) | np.isnan(y))
    csx = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    csy = np.concatenate(([0.0], np.cumsum(np.where(valid, y, 0.0))))
    csn = np.concatenate(([0], np.cumsum(valid)))
    on = ev["onset"].to_numpy(int)
    off = ev["offset"].to_numpy(int) + 1  # inclusive -> exclusive
    n = csn[off] - csn[on]
    with np.errstate(invalid="ignore", divide="ignore"):
        ev["centroid_x"] = np.where(n > 0, (csx[off] - csx[on]) / n, np.nan)
        ev["centroid_y"] = np.where(n > 0, (csy[off] - csy[on]) / n, np.nan)
    return ev


def detect_fixations(
    series: GazeSeries,
    screen: ScreenConfig = ScreenConfig(),
    det: DetectionConfig = DetectionConfig(),
) -> pd.DataFrame:
    """Run pymovements I-DT. onset/offset are sample indices on the uniform grid."""
    min_samples = min_duration_samples(det.min_duration_ms, series.sampling_rate_hz)
    if len(series.x) < min_samples + 1 or np.isnan(series.x).all():
        return add_centroids(pd.DataFrame(columns=EVENT_COLUMNS), series.x, series.y)

    experiment = pm.Experiment(
        screen.width_px, screen.height_px, screen.width_cm, screen.height_cm,
        screen.distance_cm, sampling_rate=series.sampling_rate_hz,
    )
    gaze = pm.gaze.from_numpy(
        pixel=np.vstack([series.x, series.y]), time=np.arange(len(series.x)),
        experiment=experiment,
    )
    gaze.pix2deg()
    with warnings.catch_warnings():
        # An empty trial is a valid outcome; don't spam one warning per trial.
        warnings.filterwarnings("ignore", message="idt: No events were detected")
        gaze.detect("idt", dispersion_threshold=det.dispersion_threshold_deg,
                    minimum_duration=int(min_samples))
    events = gaze.events.frame.to_pandas()
    if events.empty:
        events = pd.DataFrame(columns=EVENT_COLUMNS)
    events = events[events["name"] == "fixation"].reset_index(drop=True)
    return add_centroids(events, series.x, series.y)


# --------------------------------------------------------------------------- #
# Reading parameters
# --------------------------------------------------------------------------- #
def count_regressions(fix: pd.DataFrame, reading: ReadingConfig = ReadingConfig()) -> int:
    """Backward same-line steps plus upward jumps of more than half a line.

    Steps down by more than half a line (return sweeps) are never regressions.
    """
    if len(fix) < 2:
        return 0
    dx = fix["centroid_x"].diff()
    dy = fix["centroid_y"].diff()
    half = reading.same_line_fraction * reading.line_height_px
    same_line = (dx < -reading.regression_dx_px) & (dy.abs() <= half)
    line_up = dy < -half
    return int((same_line | line_up).sum())


def aoi_metrics(fix: pd.DataFrame, aoi_bounds, period_ms: float, recording_start_idx: int = 0) -> dict:
    """aoi_bounds = (x_min, x_max, y_min, y_max) in px."""
    x_min, x_max, y_min, y_max = aoi_bounds
    in_aoi = ((fix["centroid_x"] >= x_min) & (fix["centroid_x"] <= x_max)
              & (fix["centroid_y"] >= y_min) & (fix["centroid_y"] <= y_max)).to_numpy()
    durations = (fix["offset"] - fix["onset"]).to_numpy(float) * period_ms
    entries = int((in_aoi & ~np.concatenate(([False], in_aoi[:-1]))).sum())
    latency = None
    if in_aoi.any():
        latency = float((fix["onset"].to_numpy()[in_aoi].min() - recording_start_idx) * period_ms)
    return {
        "dwell_time_ms": float(durations[in_aoi].sum()),
        "first_fixation_latency_ms": latency,
        "entry_count": entries,
        "revisit_count": max(0, entries - 1),
    }


def extract_reading_parameters(
    fixations: pd.DataFrame,
    sampling_rate_hz: float,
    reading: ReadingConfig = ReadingConfig(),
    aoi_bounds: Optional[Sequence[float]] = None,
    recording_start_idx: int = 0,
) -> dict:
    """Summarise fixations (with centroid_x/centroid_y) into reading metrics."""
    period = 1000.0 / sampling_rate_hz
    n = len(fixations)
    out = {
        "fixation_count": n,
        "fixation_dur_mean_ms": 0.0,
        "fixation_dur_std_ms": 0.0,
        "gap_count": 0,
        "gap_dur_mean_ms": 0.0,
        "regression_count": 0,
        "regression_rate_pct": 0.0,
        "aoi_metrics": {},
    }
    if n == 0:
        return out

    dur = (fixations["offset"] - fixations["onset"]).to_numpy(float) * period
    out["fixation_dur_mean_ms"] = float(dur.mean())
    out["fixation_dur_std_ms"] = float(dur.std(ddof=1)) if n > 1 else 0.0

    if n > 1:
        # Time between consecutive fixations. NOT saccade duration: it also holds
        # tracking loss and any sub-threshold fixation the algorithm dropped.
        gaps = (fixations["onset"].to_numpy()[1:] - fixations["offset"].to_numpy()[:-1]).astype(float)
        gaps = gaps[gaps > 0]
        out["gap_count"] = int(len(gaps))
        out["gap_dur_mean_ms"] = float(gaps.mean() * period) if len(gaps) else 0.0

    out["regression_count"] = count_regressions(fixations, reading)
    out["regression_rate_pct"] = 100.0 * out["regression_count"] / n
    if aoi_bounds is not None:
        out["aoi_metrics"] = aoi_metrics(fixations, aoi_bounds, period, recording_start_idx)
    return out


def process_gaze_dataframe(
    df: pd.DataFrame,
    columns: GazeColumns = GazeColumns(),
    screen: ScreenConfig = ScreenConfig(),
    det: DetectionConfig = DetectionConfig(),
    reading: ReadingConfig = ReadingConfig(),
    aoi_bounds: Optional[Sequence[float]] = None,
    return_events: bool = False,
):
    series = prepare_gaze(df, columns, det)
    fixations = detect_fixations(series, screen, det)
    params = extract_reading_parameters(fixations, series.sampling_rate_hz, reading, aoi_bounds)
    params.update({
        "sampling_rate_hz": series.sampling_rate_hz,
        "usable_sample_pct": series.usable_sample_pct,
        "tracking_loss_pct": 100.0 - series.usable_sample_pct,
        "n_dropped_samples": series.n_dropped_samples,
        "config": {"screen": asdict(screen), "detection": asdict(det), "reading": asdict(reading)},
    })
    if return_events:
        ev = fixations.copy()
        ev["onset_ms"] = ev["onset"] * series.period_ms
        ev["offset_ms"] = ev["offset"] * series.period_ms
        return params, ev
    return params


def process_real_gaze_data(filepath: str, **kwargs):
    """Backward-compatible entry point: CSV path -> parameter dict."""
    return process_gaze_dataframe(pd.read_csv(filepath), **kwargs)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main() -> None:
    p = argparse.ArgumentParser(description="Extract reading parameters from one gaze CSV.")
    p.add_argument("csv")
    p.add_argument("--dispersion", type=float, default=DetectionConfig.dispersion_threshold_deg)
    p.add_argument("--min-duration", type=float, default=DetectionConfig.min_duration_ms)
    p.add_argument("--max-gap", type=float, default=DetectionConfig.max_interp_gap_ms)
    p.add_argument("--line-height", type=float, default=ReadingConfig.line_height_px)
    a = p.parse_args()
    params = process_real_gaze_data(
        a.csv,
        det=DetectionConfig(a.dispersion, a.min_duration, a.max_gap),
        reading=ReadingConfig(line_height_px=a.line_height),
    )
    print(json.dumps(params, indent=2, default=float))


if __name__ == "__main__":
    _main()
