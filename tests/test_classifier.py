import numpy as np
import pandas as pd
import pytest

from src.movement_classifier import (
    DetectionConfig, GazeColumns, ReadingConfig, aoi_metrics, count_regressions,
    extract_reading_parameters, fill_short_gaps, min_duration_samples,
    prepare_gaze, process_gaze_dataframe, process_real_gaze_data,
)
from tests.synthetic import EASY_POINTS, synth_scanpath


def test_recovers_count_duration_and_rate(easy_df):
    p = process_gaze_dataframe(easy_df)
    assert p["fixation_count"] == 5
    assert p["fixation_dur_mean_ms"] == pytest.approx(400, abs=12)
    assert p["sampling_rate_hz"] == pytest.approx(250, rel=1e-3)
    assert p["tracking_loss_pct"] == pytest.approx(0, abs=1e-6)


def test_gap_metrics_replace_saccade_ratio(easy_df):
    p = process_gaze_dataframe(easy_df)
    assert "s_f_ratio" not in p and "saccade_count" not in p
    assert p["gap_count"] == p["fixation_count"] - 1  # 4 gaps between 5 fixations


def test_dropped_frames_are_counted_and_bridged(easy_df):
    dropped = easy_df.drop(index=[200, 201, 202, 350, 450]).reset_index(drop=True)
    p = process_gaze_dataframe(dropped)
    assert p["n_dropped_samples"] == 5
    assert p["fixation_count"] == 5
    assert p["fixation_dur_mean_ms"] == pytest.approx(400, abs=12)
    assert p["tracking_loss_pct"] > 0


def test_long_blink_splits_fixation_but_short_gap_does_not():
    # 600 ms fixations = 150 samples each, plus 3 transit samples between them.
    df = synth_scanpath(EASY_POINTS, fix_ms=600)
    cols = ["gaze_x_left", "gaze_y_left", "gaze_x_right", "gaze_y_right"]
    third_start = 2 * (150 + 3)  # first sample of the 3rd fixation

    long_blink = df.copy()
    s = third_start + 50
    long_blink.loc[s:s + 29, cols] = 0.0  # 30 samples = 120 ms > 75 ms: not bridged
    # leaves 200 ms and 280 ms of valid data, both above the 100 ms minimum
    assert process_gaze_dataframe(long_blink)["fixation_count"] == 6

    short_gap = df.copy()
    short_gap.loc[s:s + 9, cols] = 0.0  # 10 samples = 40 ms < 75 ms: bridged
    assert process_gaze_dataframe(short_gap)["fixation_count"] == 5

    no_interp = process_gaze_dataframe(short_gap, det=DetectionConfig(max_interp_gap_ms=0))
    assert no_interp["fixation_count"] == 6  # interpolation off -> the 40 ms gap splits it


def test_both_eyes_and_single_eye_fallback(easy_df):
    left_lost = easy_df.copy()
    left_lost[["gaze_x_left", "gaze_y_left"]] = 0.0
    p = process_gaze_dataframe(left_lost)
    assert p["usable_sample_pct"] == pytest.approx(100, abs=1e-6)
    assert p["fixation_count"] == 5

    only_left = easy_df.drop(columns=["gaze_x_right", "gaze_y_right"])
    assert process_gaze_dataframe(only_left)["fixation_count"] == 5

    with pytest.raises(KeyError):
        process_gaze_dataframe(easy_df.drop(columns=["gaze_x_left", "gaze_x_right"]))


def test_eye_average_is_used():
    df = synth_scanpath([(700, 500)], fix_ms=400, noise_px=0.0)
    df["gaze_x_left"] += 10
    df["gaze_x_right"] -= 10
    series = prepare_gaze(df)
    assert np.nanmean(series.x) == pytest.approx(700, abs=1e-6)


def test_regressions_ignore_return_sweeps():
    # forward, forward, BACK (regression), forward, return sweep down a line
    # (not a regression), forward, UP to previous line (regression)
    pts = [(700, 500), (850, 500), (1000, 500), (850, 500), (1150, 500),
           (700, 540), (850, 540), (700, 500)]
    df = synth_scanpath(pts, fix_ms=300)
    p = process_gaze_dataframe(df, reading=ReadingConfig(line_height_px=40))
    assert p["fixation_count"] == len(pts)
    assert p["regression_count"] == 2
    assert p["regression_rate_pct"] == pytest.approx(100 * 2 / len(pts))


def test_count_regressions_unit():
    fix = pd.DataFrame({"centroid_x": [100, 200, 150, 300, 100, 120],
                        "centroid_y": [100, 100, 100, 100, 140, 100]})
    # 200->150 same-line back (yes); 300->100 with dy=+40 is a return sweep (no);
    # 100->120 with dy=-40 is up a line (yes)
    assert count_regressions(fix, ReadingConfig(line_height_px=40)) == 2


def test_aoi_entries_include_first_fixation_and_revisits():
    fix = pd.DataFrame({
        "onset": [0, 100, 200, 300, 400], "offset": [50, 150, 250, 350, 450],
        "centroid_x": [10, 900, 10, 900, 10], "centroid_y": [10, 10, 10, 10, 10],
    })
    m = aoi_metrics(fix, (0, 100, 0, 100), period_ms=4.0)
    assert m["entry_count"] == 3
    assert m["revisit_count"] == 2
    assert m["first_fixation_latency_ms"] == 0.0
    assert m["dwell_time_ms"] == pytest.approx(3 * 50 * 4.0)


def test_fill_short_gaps_rules():
    x = np.array([1, np.nan, np.nan, 4, np.nan, np.nan, np.nan, np.nan, 9, np.nan], float)
    fx, fy = fill_short_gaps(x, x.copy(), max_gap_samples=3)
    assert list(fx[:4]) == [1, 2, 3, 4]           # short interior gap filled linearly
    assert np.isnan(fx[4:8]).all()                # 4-sample gap left alone
    assert np.isnan(fx[9])                        # trailing gap left alone
    assert np.isnan(fill_short_gaps(x, x.copy(), 0)[0][1])  # disabled


def test_min_duration_samples_rounds_up():
    assert min_duration_samples(100, 250) == 25
    assert min_duration_samples(100, 60) == 6      # 100/16.67 = 6.0, not truncated to 5
    assert min_duration_samples(1, 250) == 2       # never below 2 samples


def test_empty_and_too_short_input_do_not_crash():
    df = synth_scanpath([(700, 500)], fix_ms=20)  # 5 samples, below the 100 ms minimum
    p = process_gaze_dataframe(df)
    assert p["fixation_count"] == 0 and p["regression_count"] == 0
    all_lost = synth_scanpath([(700, 500)], fix_ms=400)
    all_lost[["gaze_x_left", "gaze_y_left", "gaze_x_right", "gaze_y_right"]] = 0.0
    assert process_gaze_dataframe(all_lost)["fixation_count"] == 0


def test_extract_reading_parameters_direct():
    fix = pd.DataFrame({"onset": [0, 40, 90], "offset": [25, 80, 140],
                        "centroid_x": [100, 300, 500], "centroid_y": [100, 100, 100]})
    p = extract_reading_parameters(fix, sampling_rate_hz=250)
    assert p["fixation_count"] == 3
    assert p["fixation_dur_mean_ms"] == pytest.approx(np.mean([25, 40, 50]) * 4)
    assert p["gap_count"] == 2
    assert p["gap_dur_mean_ms"] == pytest.approx(np.mean([15, 10]) * 4)
    assert p["regression_count"] == 0


def test_wrapper_reads_csv(tmp_path, easy_df):
    path = tmp_path / "s.csv"
    easy_df.to_csv(path, index=False)
    assert process_real_gaze_data(str(path))["fixation_count"] == 5


def test_time_units_and_config_validation(easy_df):
    ms = easy_df.copy()
    ms["time"] = ms["time"] / 1000.0
    p = process_gaze_dataframe(ms, columns=GazeColumns(time_unit="ms"))
    assert p["sampling_rate_hz"] == pytest.approx(250, rel=1e-3)
    with pytest.raises(ValueError):
        prepare_gaze(easy_df, GazeColumns(time_unit="minutes"))
    with pytest.raises(KeyError):
        prepare_gaze(easy_df.rename(columns={"time": "t"}))


def test_stricter_dispersion_splits_more():
    # two clusters 1.2 deg apart: merged at 2.2, separate at 0.8
    df = synth_scanpath([(900, 500), (948, 500)], fix_ms=300, noise_px=0.5)
    assert process_gaze_dataframe(df, det=DetectionConfig(dispersion_threshold_deg=2.2))["fixation_count"] == 1
    assert process_gaze_dataframe(df, det=DetectionConfig(dispersion_threshold_deg=0.8))["fixation_count"] == 2
