"""
End-to-end test with a FAKE recording (no webcam, no real data needed).

    python test_pipeline.py

Builds a 30 Hz gaze CSV (t_ns, screen_fx, screen_fy) of someone reading 5 lines,
writes its _meta.json, runs run_assessment_features, and checks the numbers.
Run it from the src/ folder, next to the other scripts.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from session_meta import save_session_meta
from run_assessment_features import run
from session_meta import load_session_meta, meta_path_for

SW, SH = 1920, 1080
CENTRES = [200, 300, 400, 500, 600]
HZ = 30


def fake_recording(path, lines=range(5), step=150, n_per_line=8, fix_ms=250,
                   regress_on=None, seed=0):
    rng = np.random.default_rng(seed)
    pts = []                                    # one (x, y) per fixation
    for l in lines:
        xs = [150 + step * k for k in range(n_per_line)]
        if regress_on == l:
            xs.insert(4, xs[3] - 100)           # one clear jump back
        pts += [(x, CENTRES[l]) for x in xs]
    samples_per_fix = int(round(fix_ms / 1000 * HZ))
    rows = []
    i = 0
    for x, y in pts:
        for _ in range(samples_per_fix):
            rows.append((round(i * 1e9 / HZ),
                         (x + rng.normal(0, 3)) / SW, (y + rng.normal(0, 3)) / SH))
            i += 1
    pd.DataFrame(rows, columns=["t_ns", "screen_fx", "screen_fy"]).to_csv(path, index=False)


def test_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "fake_gaze.csv"
        fake_recording(csv, regress_on=2)
        save_session_meta(csv, width_px=SW, height_px=SH, width_cm=34.5, height_cm=19.4,
                          distance_cm=50, line_centres=CENTRES)
        meta = load_session_meta(meta_path_for(csv))
        params, per_line, s, labelled = run(csv, meta, out_dir=tmp)

        assert 29 < params["sampling_rate_hz"] < 31, params["sampling_rate_hz"]
        assert s["return_sweeps"] == 4, s
        assert s["within_line_regressions"] == 1, s
        assert s["line_skips"] == 0 and s["inter_line_regressions"] == 0, s
        assert s["off_line_fix"] == 0, s
        assert len(per_line) == 5
        assert 130 < s["mean_fwd_amp_px"] < 170, s["mean_fwd_amp_px"]
        for suffix in ("_fixations.csv", "_per_line.csv", "_features.json"):
            assert (Path(tmp) / f"fake_gaze{suffix}").exists(), suffix
        print("PASS  test_pipeline")
        print(per_line.round(1).to_string(index=False))
        print(s)


if __name__ == "__main__":
    test_pipeline()