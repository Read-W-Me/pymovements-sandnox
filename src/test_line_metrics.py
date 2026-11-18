"""
Run:  python test_line_metrics.py        (no pytest needed; pytest also works)
Needs line_metrics.py in the same folder.
All data is synthetic, so you know the right answers in advance.
"""
import numpy as np
import pandas as pd
from line_metrics import line_metrics, assign_lines

CENTRES = [200, 300, 400, 500, 600]


def make_fix(points, dur=200, gap=50):
    """points = list of (x, y) -> fixation table."""
    n = len(points)
    onset = np.arange(n) * (dur + gap)
    return pd.DataFrame({
        "onset_ms": onset, "offset_ms": onset + dur,
        "centroid_x": [p[0] for p in points],
        "centroid_y": [p[1] for p in points],
    })


def simulated_reading(rng, x_noise=10, y_noise=12, n_per_line=8, lines=range(5),
                      regress_on=None, skip_to=None):
    """Clean left-to-right reading of every line, with noise.
    regress_on: line index where one 80 px jump back is injected.
    skip_to:   (from_line, to_line) -> jump straight from end of from_line to to_line."""
    pts, ls = [], list(lines)
    for l in ls:
        xs = [100 + 110 * k for k in range(n_per_line)]
        if regress_on == l:
            xs.insert(4, xs[3] - 80)           # one regression
        pts += [(x + rng.normal(0, x_noise), CENTRES[l] + rng.normal(0, y_noise)) for x in xs]
    return make_fix(pts)


# ------------------------------------------------------------------ tests
def test_hand_made():
    pts = [(100, 205), (200, 205), (300, 205), (250, 205), (400, 205),
           (100, 305), (200, 305), (150, 505), (300, 305)]
    per_line, s, _ = line_metrics(make_fix(pts), CENTRES[:4])
    assert per_line.set_index("line").n_fix.to_dict() == {0: 5, 1: 3, 3: 1}
    assert s["within_line_regressions"] == 1
    assert s["return_sweeps"] == 1
    assert s["line_skips"] == 1
    assert s["inter_line_regressions"] == 1
    assert abs(s["mean_fwd_amp_px"] - 112.5) < 1e-9


def test_clean_noisy_reading():
    rng = np.random.default_rng(0)
    _, s, _ = line_metrics(simulated_reading(rng), CENTRES)
    assert s["return_sweeps"] == 4            # 5 lines -> 4 sweeps
    assert s["within_line_regressions"] == 0
    assert s["line_skips"] == 0
    assert s["inter_line_regressions"] == 0
    assert s["off_line_fix"] == 0
    assert 95 < s["mean_fwd_amp_px"] < 125    # true step is 110


def test_injected_regression():
    rng = np.random.default_rng(1)
    _, s, _ = line_metrics(simulated_reading(rng, regress_on=2), CENTRES)
    assert s["within_line_regressions"] == 1
    assert s["return_sweeps"] == 4


def test_skipped_line():
    rng = np.random.default_rng(2)
    fix = simulated_reading(rng, lines=[0, 1, 3, 4])      # line 2 never read
    _, s, _ = line_metrics(fix, CENTRES)
    assert s["line_skips"] == 1
    assert s["return_sweeps"] == 2                         # 0->1 and 3->4


def test_reread_previous_line():
    rng = np.random.default_rng(3)
    fix = simulated_reading(rng, lines=[0, 1, 2, 1, 2, 3])  # goes back to line 1
    per_line, s, _ = line_metrics(fix, CENTRES)
    assert s["inter_line_regressions"] == 1
    assert per_line.set_index("line").revisits[1] == 1


def test_off_screen_fixations():
    pts = [(100, 205), (200, 205), (300, 900), (400, 205)]   # y=900 is off all lines
    _, s, f = line_metrics(make_fix(pts), CENTRES)
    assert s["off_line_fix"] == 1
    assert (f.transition == "off_line").sum() == 2


def test_assign_lines_tolerance():
    fix = make_fix([(0, 210), (0, 255), (0, 290)])
    assert list(assign_lines(fix, CENTRES, tol=30)) == [0, -1, 1]   # 255 is >30 px from both lines


def test_empty_input_does_not_crash():
    empty = make_fix([])
    per_line, s, _ = line_metrics(empty, CENTRES)
    assert s["n_fix"] == 0 and len(per_line) == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")

    # also dump a sample so you can eyeball the labels
    rng = np.random.default_rng(1)
    fix = simulated_reading(rng, regress_on=2)
    _, _, labelled = line_metrics(fix, CENTRES)
    labelled.round(1).to_csv("sample_labelled_fixations.csv", index=False)
    print("wrote sample_labelled_fixations.csv")