"""Synthetic gaze generators used by the tests (no real data needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synth_scanpath(points, fix_ms=400, hz=250, noise_px=2.0, transit_samples=3, seed=0,
                   t0_us=1_000_000, fix_ms_per_point=None):
    """Fixations at ``points`` (x, y) px with short linear transits in between.

    Returns a DataFrame with the ETDD70-style columns (time in microseconds,
    left/right eye positions). ``fix_ms_per_point`` overrides ``fix_ms`` per fixation.
    """
    rng = np.random.default_rng(seed)
    period_ms = 1000.0 / hz
    xs, ys = [], []
    prev = None
    for k, (px, py) in enumerate(points):
        if prev is not None and transit_samples:
            for f in np.linspace(0, 1, transit_samples + 2)[1:-1]:
                xs.append(prev[0] + f * (px - prev[0]))
                ys.append(prev[1] + f * (py - prev[1]))
        dur = fix_ms if fix_ms_per_point is None else fix_ms_per_point[k]
        n = int(round(dur / period_ms))
        xs.extend(px + rng.normal(0, noise_px, n))
        ys.extend(py + rng.normal(0, noise_px, n))
        prev = (px, py)
    x, y = np.asarray(xs), np.asarray(ys)
    t_us = t0_us + np.arange(len(x)) * int(round(period_ms * 1000))
    return pd.DataFrame({
        "time": t_us,
        "gaze_x_left": x, "gaze_y_left": y,
        "gaze_x_right": x, "gaze_y_right": y,
    })


# Five fixations, 150 px apart (~3.5 deg) near screen centre: an easy, unambiguous case.
EASY_POINTS = [(600, 500), (750, 500), (900, 500), (1050, 500), (1200, 500)]
