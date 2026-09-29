"""ETDD70 file discovery, ground-truth loading and safe archive extraction."""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

RAW_PATTERN = re.compile(r"^Subject_(?P<subject>\d+)_(?P<trial>T\d+)_(?P<task>.+)_raw\.csv$", re.I)

# our key -> ground-truth column in *_metrics.csv (columns are lower-cased on load)
GT_COLUMNS = {
    "fixation_count": "n_fix_trial",
    "fixation_dur_mean_ms": "mean_fix_dur_trial",
    "regression_count": "n_regress_trial",
    "saccade_count": "n_sacc_trial",
    "saccade_dur_mean_ms": "mean_sacc_dur_trial",
}


@dataclass(frozen=True)
class Trial:
    subject: str
    trial: str
    task: str
    raw_path: Path
    metrics_path: Optional[Path]


def discover_trials(root, tasks: Optional[Iterable[str]] = None, require_metrics: bool = True):
    """Find *_raw.csv files (and their *_metrics.csv twin) under ``root``.

    ``tasks`` filters by task name, e.g. ["Pseudo_Text"]; matching is case-insensitive.
    """
    wanted = {t.lower() for t in tasks} if tasks else None
    trials = []
    for raw in sorted(Path(root).rglob("*_raw.csv")):
        m = RAW_PATTERN.match(raw.name)
        if not m:
            continue
        if wanted and m["task"].lower() not in wanted:
            continue
        metrics = raw.with_name(raw.name[: -len("_raw.csv")] + "_metrics.csv")
        metrics = metrics if metrics.exists() else None
        if require_metrics and metrics is None:
            continue
        trials.append(Trial(m["subject"], m["trial"], m["task"], raw, metrics))
    return trials


def load_ground_truth(metrics_path) -> dict:
    """Trial-level ground truth (first row; the file repeats trial values per AOI)."""
    df = pd.read_csv(metrics_path, sep=None, engine="python")
    df.columns = df.columns.str.strip().str.lower()
    first = df.iloc[0]
    out = {}
    for key, col in GT_COLUMNS.items():
        out[key] = float(pd.to_numeric(first.get(col), errors="coerce")) if col in df.columns else float("nan")
    return out


def extract_zip_safely(zip_path, dest) -> int:
    """Extract a zip, refusing members that would escape ``dest``. Returns member count."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"unsafe path in archive: {member}")
        zf.extractall(dest)
        return len(zf.namelist())
