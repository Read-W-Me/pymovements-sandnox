"""
One command from a recorded gaze CSV to assessment-layer features.

    python run_assessment_features.py gaze_log.csv

Reads   gaze_log.csv         (columns t_ns, screen_fx, screen_fy by default)
        gaze_log_meta.json   (screen px/cm, distance, line centres; see session_meta.py)
Writes  gaze_log_fixations.csv   fixations with line + transition labels
        gaze_log_per_line.csv    per-line metrics
        gaze_log_features.json   everything in one place (pymovements params + line summary)

Put this file in the same folder as movement_classifier.py, line_metrics.py and
session_meta.py (your src/ folder).

If there is no _meta.json you can pass the values on the command line instead,
see --help. Anything missing is an error, never a silent guess.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from movement_classifier import (DetectionConfig, GazeColumns, ReadingConfig,
                                 ScreenConfig, process_gaze_dataframe)
from line_metrics import line_metrics
from session_meta import load_session_meta, meta_path_for


def to_pymovements_frame(d, t_col, x_col, y_col, t_unit, width_px, height_px):
    """Our CSV (normalised 0-1 coords) -> the table pymovements' pipeline expects (px)."""
    missing = [c for c in (t_col, x_col, y_col) if c not in d.columns]
    if missing:
        raise SystemExit(f"Column(s) {missing} not in the CSV. Columns found: {list(d.columns)}\n"
                         f"Use --t-col / --x-col / --y-col to point at the right ones.")
    d = d.dropna(subset=[t_col, x_col, y_col]).sort_values(t_col)
    scale = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}[t_unit]   # -> microseconds
    return pd.DataFrame({
        "time": d[t_col].to_numpy(float) * scale,
        "gaze_x": d[x_col].to_numpy(float) * width_px,
        "gaze_y": d[y_col].to_numpy(float) * height_px,
    })


def run(csv_path, meta, t_col="t_ns", x_col="screen_fx", y_col="screen_fy", t_unit="ns",
        min_regress_px=20, return_px=150, line_tol=None, out_dir=None):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir else csv_path.parent
    stem = csv_path.stem

    raw = pd.read_csv(csv_path)
    a = to_pymovements_frame(raw, t_col, x_col, y_col, t_unit, meta["width_px"], meta["height_px"])
    if len(a) < 10:
        raise SystemExit(f"Only {len(a)} usable gaze rows, too few to analyse.")

    screen = ScreenConfig(meta["width_px"], meta["height_px"],
                          meta["width_cm"], meta["height_cm"], meta["distance_cm"])
    cols = GazeColumns(time="time", time_unit="us", x=("gaze_x",), y=("gaze_y",))
    spacing = float(np.median(np.diff(meta["line_centres"]))) if len(meta["line_centres"]) > 1 else 40.0
    params, fix = process_gaze_dataframe(a, columns=cols, screen=screen,
                                         reading=ReadingConfig(line_height_px=spacing),
                                         return_events=True)

    if len(fix) == 0:
        raise SystemExit("No fixations detected. Check the recording, sampling rate and geometry.")

    per_line, summary, labelled = line_metrics(fix, meta["line_centres"],
                                               min_regress_px=min_regress_px,
                                               return_px=return_px, tol=line_tol)

    labelled.round(2).to_csv(out_dir / f"{stem}_fixations.csv", index=False)
    per_line.round(2).to_csv(out_dir / f"{stem}_per_line.csv", index=False)
    features = {"recording": csv_path.name, "pymovements": params, "line_level": summary,
                "thresholds": {"min_regress_px": min_regress_px, "return_px": return_px,
                               "line_tol": line_tol}}
    (out_dir / f"{stem}_features.json").write_text(json.dumps(features, indent=2, default=float))
    return params, per_line, summary, labelled


def _print_report(params, per_line, summary):
    print(f"\nSampling rate: {params['sampling_rate_hz']:.1f} Hz | "
          f"usable samples: {params['usable_sample_pct']:.1f}%")
    if params["sampling_rate_hz"] < 60:
        print("  ! Low sampling rate: fixations shorter than ~130 ms are lost.")
    if params["usable_sample_pct"] < 80:
        print("  ! Under 80% usable samples: treat the numbers with caution.")
    print(f"Fixations: {summary['n_fix']} (off any line: {summary['off_line_fix']})")
    for k in ("within_line_regressions", "return_sweeps", "line_advances_no_sweep",
              "line_skips", "inter_line_regressions", "mean_fwd_amp_px"):
        v = summary[k]
        print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")
    if summary["n_fix"] and summary["off_line_fix"] / summary["n_fix"] > 0.3:
        print("  ! Over 30% of fixations are off any line: check that line centres and "
              "screen px in the meta file match what was on screen.")
    print("\nPer line:\n", per_line.round(1).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--meta", help="path to _meta.json (default: <csv>_meta.json)")
    ap.add_argument("--t-col", default="t_ns")
    ap.add_argument("--x-col", default="screen_fx")
    ap.add_argument("--y-col", default="screen_fy")
    ap.add_argument("--t-unit", default="ns", choices=["ns", "us", "ms", "s"])
    ap.add_argument("--min-regress-px", type=float, default=20)
    ap.add_argument("--return-px", type=float, default=150)
    ap.add_argument("--line-tol", type=float, default=None, help="max px from a line centre")
    ap.add_argument("--out-dir")
    a = ap.parse_args()

    mp = Path(a.meta) if a.meta else meta_path_for(a.csv)
    if not mp.exists():
        raise SystemExit(f"No meta file at {mp}.\nCreate one with session_meta.py "
                         f"(see its docstring), then rerun.")
    meta = load_session_meta(mp)
    params, per_line, summary, _ = run(a.csv, meta, a.t_col, a.x_col, a.y_col, a.t_unit,
                                       a.min_regress_px, a.return_px, a.line_tol, a.out_dir)
    _print_report(params, per_line, summary)