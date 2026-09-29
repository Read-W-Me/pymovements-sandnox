# Read W/Me: Movement Classifier Validation

## Overview
This repository contains the `pymovements` gaze classification pipeline for the **Read W/Me** project. The core extraction logic uses a Dispersion-Threshold (I-DT) algorithm to group continuous raw gaze coordinates into fixations, and derives reading metrics from them: fixation count and duration, regressions, inter-fixation gaps and optional AOI metrics.

## Repository layout
```
src/movement_classifier.py   pipeline: load -> clean -> I-DT -> reading metrics (+ CLI)
src/dataset.py               ETDD70 file discovery, ground-truth loading, safe zip extraction
src/evaluation.py            agreement stats, event matching, Cohen's kappa, subject split
src/validation.py            shared validation / parameter-grid helpers
scripts/fetch_dataset.py     download or extract ETDD70 into data/ETDD70
scripts/run_validation.py    multi-trial validation, optional train/test tuning
scripts/sensitivity_grid.py  dispersion x duration sensitivity (exploratory)
tests/                       pytest suite on synthetic gaze (no dataset needed)
```

## Quick start
```bash
pip install -r requirements-dev.txt
pytest                                            # synthetic data only, ~2 s

python scripts/fetch_dataset.py --zip /path/to/ETDD70.zip      # or --url <direct link>
python scripts/run_validation.py --data-root data/ETDD70 --tasks Pseudo_Text
python scripts/run_validation.py --data-root data/ETDD70 --tasks Pseudo_Text --tune --n-jobs 4
python scripts/sensitivity_grid.py --data-root data/ETDD70 --tasks Pseudo_Text --max-trials 30
python -m src.movement_classifier path/to/gaze.csv --dispersion 2.2 --min-duration 100
```
The dataset is not bundled. Get the archive from the dataset's official page and follow its license and citation terms.

## Pipeline
1. **Timestamps.** Samples are placed on a uniform time grid at the detected sampling rate. Dropped frames become NaN gaps rather than silently shortening durations. (pymovements 0.28 I-DT needs evenly spaced integer timesteps, so this step is required.)
2. **Eyes.** The mean of the valid eyes is used per sample (left/right column names are configurable via `GazeColumns`).
3. **Gaps.** Only gaps of at most `max_interp_gap_ms` (default 75 ms) are linearly interpolated. Longer gaps (blinks, tracking loss) stay NaN and split fixations.
4. **Fixations.** pymovements I-DT with `dispersion_threshold_deg=2.2` and `min_duration_ms=100`, converted to samples with rounding up.
5. **Metrics.** Fixation count, mean/SD duration, inter-fixation gaps, regressions, optional AOI metrics. Screen geometry, thresholds and line height live in `ScreenConfig`, `DetectionConfig` and `ReadingConfig`.

### Metric definitions
* **Regression:** a leftward step larger than `regression_dx_px` on the same line (`|dy| <= 0.5 * line_height_px`), or an upward jump of more than half a line. Downward jumps of more than half a line (return sweeps) are never regressions. Set `line_height_px` to the real stimulus line spacing.
* **Gap metrics (`gap_count`, `gap_dur_mean_ms`):** I-DT only produces fixations. These are the intervals between consecutive fixations. They also contain tracking loss and any fixation the algorithm discarded, so they are **not** saccade counts or durations. (The earlier version reported them as saccades, so its saccade count was always fixations minus one and its saccade/fixation ratio was tautological. That ratio was removed.)
* **AOI metrics:** `dwell_time_ms`, `first_fixation_latency_ms` (from the start of the recording), `entry_count`, `revisit_count` (entries minus one). AOI metrics are only computed when `aoi_bounds` is given.

## Known limitations
* **Dispersion definition.** pymovements computes I-DT dispersion as the *sum* of the x-range and y-range of the window (Salvucci & Goldberg), not the Euclidean diameter. `2.2` is on that scale.
* **Boundary splits.** With a large threshold, the tail of a saccade can land within threshold of the next fixation. The window then barely passes at the start and breaks a few samples later, splitting one fixation into a short and a long one. This showed up on synthetic data with clean, noise-free saccades, so expect it on real data. A merge step for adjacent fixations is not implemented.
* **Off-by-one.** The sample that breaks the dispersion limit is included in the fixation, so each fixation ends one sample late (4 ms at 250 Hz). Durations follow pymovements (`offset - onset`).
* **The HMM layer is not in this repo and not validated here.** The upstream HMM filtering of jitter, blinks and tracker loss is a design assumption. The validation runs on raw ETDD70 data with no HMM stage, so it is untested whether parameters tuned on raw data transfer to HMM-filtered input.
* **Screen geometry is unverified.** The defaults (1920x1080 px, 53x30 cm, 60 cm) were carried over from the original script. Degree conversion, and therefore the dispersion threshold, depends on them.
* **"Ground truth" is another algorithm's output.** ETDD70 provides per-trial aggregates computed by the dataset's own tooling, not human-labelled events. Agreement with it shows consistency, not correctness. Event-level checks need a dataset with labelled events; `src/evaluation.py` provides `match_events` and `cohens_kappa` for that.

## Validation protocol
* Trial-level agreement over all trials (mean absolute error, mean relative error, Pearson r, Bland-Altman bias and 95% limits), not a single trial.
* Subject **1322** is excluded by default (`--exclude-subjects`) because the default parameters were first tuned on that trial.
* `--tune` chooses parameters on train subjects and reports both tuned and default parameters on **held-out test subjects** (subject-level split, seeded).
* The tuning objective is mean relative error over fixation count, mean fixation duration and regression count. Gap-vs-saccade rows are informational only.
* `scripts/sensitivity_grid.py` is exploratory. Do not quote its best cell as a validated result.

## Results
No multi-subject results are committed yet. Run `scripts/run_validation.py` on the full dataset and commit the generated `results/validation_summary.md` (the `.gitignore` allows it).

**Preliminary, single trial (earlier pipeline version, Subject 1322, T5 Pseudo Text, parameters tuned on that same trial):** fixation count 229 vs 218, mean fixation duration 610 ms vs 622 ms, regression count 15 vs 15. These numbers are anecdotal. They come from one trial, the parameters were tuned on it, and the pipeline changes in this version can move them.

## Future calibration requirements
The current `dispersion_threshold=2.2` and `minimum_duration=100 ms` were fitted to high-speed (250 Hz) clinical-tracker data from pediatric readers. **Further calibration is mandatory** for the final Read W/Me system: re-tune on our production hardware (frame rate, spatial resolution, screen geometry), on our stimuli (line height) and on our target demographic, using a held-out set as described above.

## License
MIT (see `LICENSE`); adjust the copyright holder to suit the project.
