"""
Line-level reading metrics on top of pymovements' fixation table.

Input : fix  -> DataFrame with onset_ms, offset_ms, centroid_x, centroid_y (px)
        centres -> list of line centre y-values in px (your reading["centres"])
Output: (per_line DataFrame, summary dict, labelled fixation DataFrame)

Transition types between consecutive fixations (a -> b), with dl = line_b - line_a:
  forward         dl == 0, dx > 0                 (amplitude kept)
  regression      dl == 0, dx < -min_regress_px   (small jump back inside a line)
  return_sweep    dl == +1, dx < -return_px       (normal "next line" move)
  line_advance    dl == +1, no big leftward jump  (moved down without a sweep)
  line_skip       dl >= +2
  line_back       dl <= -1                        (inter-line regression)
  off_line        either fixation is not on any line
"""
import numpy as np
import pandas as pd


def assign_lines(fix, centres, tol=None):
    c = np.asarray(centres, dtype=float)
    if tol is None:
        tol = np.median(np.diff(c)) / 2 if len(c) > 1 else 40.0
    y = fix["centroid_y"].to_numpy()
    d = np.abs(y[:, None] - c[None, :])
    idx = d.argmin(axis=1)
    idx[d.min(axis=1) > tol] = -1          # too far from any line
    return idx


def line_metrics(fix, centres, min_regress_px=20, return_px=150, tol=None):
    f = fix.sort_values("onset_ms").reset_index(drop=True).copy()
    f["line"] = assign_lines(f, centres, tol)
    f["dur_ms"] = f["offset_ms"] - f["onset_ms"]

    kinds, dxs = ([None], [np.nan]) if len(f) else ([], [])
    for i in range(1, len(f)):
        la, lb = f.line[i - 1], f.line[i]
        dx = f.centroid_x[i] - f.centroid_x[i - 1]
        dxs.append(dx)
        if la < 0 or lb < 0:
            k = "off_line"
        else:
            dl = lb - la
            if dl == 0:
                k = "forward" if dx > 0 else ("regression" if dx < -min_regress_px else "other")
            elif dl == 1:
                k = "return_sweep" if dx < -return_px else "line_advance"
            elif dl >= 2:
                k = "line_skip"
            else:
                k = "line_back"
        kinds.append(k)
    f["dx"], f["transition"] = dxs, kinds

    # visits to each line = number of separate entries
    visits = {}
    for i in range(len(f)):
        l = f.line[i]
        if l >= 0 and (i == 0 or f.line[i - 1] != l):
            visits[l] = visits.get(l, 0) + 1

    rows = []
    for l, g in f[f.line >= 0].groupby("line"):
        t = g.transition
        rows.append(dict(
            line=l, n_fix=len(g), dwell_ms=g.dur_ms.sum(),
            regressions=int((t == "regression").sum()),
            visits=visits.get(l, 0), revisits=max(visits.get(l, 0) - 1, 0),
            mean_fwd_amp_px=g.loc[t == "forward", "dx"].mean()))
    per_line = pd.DataFrame(rows)

    fwd = f.loc[f.transition == "forward", "dx"]
    summary = dict(
        n_fix=len(f),
        off_line_fix=int((f.line < 0).sum()),
        within_line_regressions=int((f.transition == "regression").sum()),
        return_sweeps=int((f.transition == "return_sweep").sum()),
        line_advances_no_sweep=int((f.transition == "line_advance").sum()),
        line_skips=int((f.transition == "line_skip").sum()),
        inter_line_regressions=int((f.transition == "line_back").sum()),
        mean_fwd_amp_px=float(fwd.mean()) if len(fwd) else np.nan,
    )
    return per_line, summary, f


# ---------------------------------------------------------------- self-test
def _selftest():
    centres = [200, 300, 400, 500]
    pts = [  # (line_y, x)
        (200, 100), (200, 200), (200, 300), (200, 250), (200, 400),  # L0
        (300, 100), (300, 200),                                        # L1 (sweep)
        (500, 150),                                                    # L3 (skip)
        (300, 300),                                                    # back to L1
    ]
    fix = pd.DataFrame({
        "onset_ms": np.arange(len(pts)) * 250,
        "offset_ms": np.arange(len(pts)) * 250 + 200,
        "centroid_x": [p[1] for p in pts],
        "centroid_y": [p[0] + 5 for p in pts],   # small vertical noise
    })
    per_line, s, _ = line_metrics(fix, centres)
    print(per_line, "\n", s)

    assert per_line.set_index("line").n_fix.to_dict() == {0: 5, 1: 3, 3: 1}
    assert s["within_line_regressions"] == 1
    assert s["return_sweeps"] == 1
    assert s["line_skips"] == 1
    assert s["inter_line_regressions"] == 1
    assert s["line_advances_no_sweep"] == 0
    assert abs(s["mean_fwd_amp_px"] - 112.5) < 1e-9   # 100,100,150,100
    assert per_line.set_index("line").revisits[1] == 1
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _selftest()