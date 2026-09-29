#!/usr/bin/env python
"""Exploratory sensitivity of the results to I-DT parameters.

Sweeps dispersion threshold x minimum duration on the chosen trials and writes
sensitivity_grid.csv (+ heatmaps). This is exploratory: it uses all trials you
give it, so do not quote its best cell as a validated result. Use
run_validation.py --tune for a train/test estimate.

    python scripts/sensitivity_grid.py --data-root data/ETDD70 --tasks Pseudo_Text --max-trials 30 --n-jobs 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.dataset import discover_trials  # noqa: E402
from src.movement_classifier import DetectionConfig, ReadingConfig  # noqa: E402
from src.validation import run_grid  # noqa: E402


def _floats(text):
    return [float(v) for v in text.split(",") if v.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/ETDD70")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--max-trials", type=int, default=None, help="use only the first N trials (sorted by path)")
    p.add_argument("--dispersions", default="0.5,1.0,1.5,2.0,2.2,2.5,3.0")
    p.add_argument("--durations", default="40,60,80,100,120,150")
    p.add_argument("--max-gap", type=float, default=DetectionConfig.max_interp_gap_ms)
    p.add_argument("--line-height", type=float, default=ReadingConfig.line_height_px)
    p.add_argument("--out", default="results")
    p.add_argument("--n-jobs", type=int, default=1)
    a = p.parse_args()

    trials = discover_trials(a.data_root, a.tasks)
    if a.max_trials:
        trials = trials[: a.max_trials]
    if not trials:
        sys.exit(f"No trials found under {a.data_root}. See scripts/fetch_dataset.py.")
    print(f"{len(trials)} trials")

    grid = run_grid(trials, _floats(a.dispersions), _floats(a.durations),
                    DetectionConfig(max_interp_gap_ms=a.max_gap),
                    reading=ReadingConfig(line_height_px=a.line_height),
                    n_jobs=a.n_jobs, verbose=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "sensitivity_grid.csv", index=False)
    print(grid.sort_values("objective").head(5).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping heatmaps")
        return
    panels = [("objective", "Objective (mean rel. error, lower = better)"),
              ("mean_fixation_count", "Mean fixation count"),
              ("mean_fixation_dur_mean_ms", "Mean fixation duration (ms)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5))
    for ax, (col, title) in zip(axes, panels):
        pv = grid.pivot(index="dispersion_deg", columns="min_duration_ms", values=col)
        im = ax.imshow(pv.to_numpy(float), aspect="auto", origin="lower")
        ax.set_xticks(range(len(pv.columns)), [f"{c:g}" for c in pv.columns])
        ax.set_yticks(range(len(pv.index)), [f"{i:g}" for i in pv.index])
        ax.set_xlabel("min duration (ms)")
        ax.set_ylabel("dispersion (deg)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax)
        for (i, j), v in np.ndenumerate(pv.to_numpy(float)):
            ax.text(j, i, f"{v:.2f}" if col == "objective" else f"{v:.0f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "sensitivity_grid.png", dpi=130)
    print(f"Wrote {out}/sensitivity_grid.csv and .png")


if __name__ == "__main__":
    main()
