#!/usr/bin/env python
"""Validate the I-DT pipeline against ETDD70 ground truth on many trials.

Examples
--------
Fixed parameters on every trial (except the subject the defaults were tuned on):
    python scripts/run_validation.py --data-root data/ETDD70 --tasks Pseudo_Text

Tune on train subjects, report on held-out test subjects:
    python scripts/run_validation.py --data-root data/ETDD70 --tasks Pseudo_Text --tune --n-jobs 4

Outputs (in --out): per-trial CSVs, validation_summary.json / .md, optional plots.
Trial-level agreement only: the ETDD70 metrics files hold per-trial aggregates,
not event timestamps, so event-level matching (src.evaluation.match_events)
needs a dataset with labelled events.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.dataset import discover_trials  # noqa: E402
from src.evaluation import split_subjects  # noqa: E402
from src.movement_classifier import DetectionConfig, ReadingConfig, ScreenConfig  # noqa: E402
from src.validation import METRICS, evaluate_trials, run_grid, summarize  # noqa: E402


def _floats(text):
    return [float(v) for v in text.split(",") if v.strip()]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/ETDD70")
    p.add_argument("--tasks", nargs="*", default=None, help="e.g. Pseudo_Text (default: all tasks)")
    p.add_argument("--out", default="results")
    p.add_argument("--exclude-subjects", nargs="*", default=["1322"],
                   help="subjects to leave out (default: 1322, the trial the defaults were first tuned on)")
    p.add_argument("--tune", action="store_true", help="grid-search on train subjects, evaluate on test subjects")
    p.add_argument("--test-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dispersion-grid", default="1.0,1.5,2.0,2.2,2.5,3.0")
    p.add_argument("--duration-grid", default="60,80,100,120,150")
    p.add_argument("--dispersion", type=float, default=DetectionConfig.dispersion_threshold_deg)
    p.add_argument("--min-duration", type=float, default=DetectionConfig.min_duration_ms)
    p.add_argument("--max-gap", type=float, default=DetectionConfig.max_interp_gap_ms)
    p.add_argument("--line-height", type=float, default=ReadingConfig.line_height_px)
    p.add_argument("--screen", nargs=5, type=float, metavar=("W_PX", "H_PX", "W_CM", "H_CM", "DIST_CM"),
                   default=None, help="override screen geometry")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _md_table(summary: dict) -> str:
    lines = ["| Metric | n | MAE | Mean rel. error | Bias | 95% limits | Pearson r |",
             "|---|---|---|---|---|---|---|"]
    for our_key, _, label, primary in METRICS:
        s = summary.get(our_key)
        if not s:
            continue
        tag = "" if primary else " *(informational)*"
        lines.append(f"| {label}{tag} | {s['n']} | {s['mae']:.2f} | {s['mean_rel_error']:.3f} | "
                     f"{s['bias']:+.2f} | [{s['loa_low']:+.1f}, {s['loa_high']:+.1f}] | {s['pearson_r']:.2f} |")
    return "\n".join(lines)


def _scatter(df, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    keys = [(k, l) for k, _, l, prim in METRICS if prim]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4.5))
    for ax, (key, label) in zip(np.atleast_1d(axes), keys):
        o, g = df[f"ours_{key}"].to_numpy(float), df[f"gt_{key}"].to_numpy(float)
        ax.scatter(g, o, s=18, alpha=0.7)
        lim = [np.nanmin([o, g]), np.nanmax([o, g])]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("Ground truth")
        ax.set_ylabel("This pipeline")
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    screen = ScreenConfig(int(a.screen[0]), int(a.screen[1]), *a.screen[2:]) if a.screen else ScreenConfig()
    reading = ReadingConfig(line_height_px=a.line_height)
    default_det = DetectionConfig(a.dispersion, a.min_duration, a.max_gap)

    trials = [t for t in discover_trials(a.data_root, a.tasks) if t.subject not in set(a.exclude_subjects)]
    if not trials:
        sys.exit(f"No trials with raw + metrics files found under {a.data_root} "
                 f"(tasks={a.tasks}, excluded={a.exclude_subjects}). See scripts/fetch_dataset.py.")
    subjects = sorted({t.subject for t in trials})
    print(f"{len(trials)} trials from {len(subjects)} subjects")

    report = {"n_trials": len(trials), "n_subjects": len(subjects), "tasks": a.tasks,
              "excluded_subjects": a.exclude_subjects, "screen": screen.__dict__,
              "reading": reading.__dict__}

    if a.tune:
        train_s, test_s = split_subjects(subjects, a.test_fraction, a.seed)
        train = [t for t in trials if t.subject in train_s]
        test = [t for t in trials if t.subject in test_s]
        print(f"train: {len(train_s)} subjects / {len(train)} trials; test: {len(test_s)} subjects / {len(test)} trials")
        grid = run_grid(train, _floats(a.dispersion_grid), _floats(a.duration_grid), default_det,
                        screen, reading, n_jobs=a.n_jobs, verbose=True)
        grid.to_csv(out / "tuning_grid_train.csv", index=False)
        best = grid.loc[grid["objective"].idxmin()]
        tuned = DetectionConfig(float(best["dispersion_deg"]), float(best["min_duration_ms"]), a.max_gap)
        print(f"best on train: dispersion={tuned.dispersion_threshold_deg} deg, min duration={tuned.min_duration_ms} ms")
        runs = {"tuned": (tuned, test), "default": (default_det, test)}
        report.update({"split": {"train_subjects": train_s, "test_subjects": test_s, "seed": a.seed},
                       "evaluated_on": "held-out test subjects"})
    else:
        runs = {"default": (default_det, trials)}
        report["evaluated_on"] = "all included trials (parameters not tuned in this run)"

    md = [f"# Validation summary\n\nEvaluated on: {report['evaluated_on']}\n"]
    report["runs"] = {}
    for label, (det, subset) in runs.items():
        df = evaluate_trials(subset, det, screen, reading, n_jobs=a.n_jobs)
        df.to_csv(out / f"per_trial_{label}.csv", index=False)
        summ = summarize(df)
        summ["config"] = det.__dict__
        report["runs"][label] = summ
        md.append(f"## {label}: dispersion {det.dispersion_threshold_deg} deg, min duration {det.min_duration_ms} ms "
                  f"({summ['n_trials']} trials, {summ['n_failed']} failed)\n\n{_md_table(summ)}\n")
        if not a.no_plots and summ["n_trials"] - summ["n_failed"] > 0:
            _scatter(df[df["error"] == ""], out / f"scatter_{label}.png")
        failed = df[df["error"] != ""]
        if len(failed):
            print(f"[{label}] {len(failed)} trials failed, first error: {failed.iloc[0]['error']}")

    (out / "validation_summary.json").write_text(json.dumps(report, indent=2, default=float))
    (out / "validation_summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"Wrote results to {out}/")


if __name__ == "__main__":
    main()
