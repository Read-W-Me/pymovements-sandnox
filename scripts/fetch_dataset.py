#!/usr/bin/env python
"""Download/extract ETDD70 into data/ETDD70 and check that it looks right.

The dataset is not bundled. Get the archive link from the dataset's official
page (Zenodo, per the ETDD70 paper), then either:

    python scripts/fetch_dataset.py --url <direct-download-url-to-zip>
    python scripts/fetch_dataset.py --zip /path/to/downloaded.zip

Both extract with path-traversal protection and report how many raw / metrics
files were found. Follow the dataset's own license and citation terms.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import discover_trials, extract_zip_safely  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--zip")
    p.add_argument("--dest", default="data/ETDD70")
    a = p.parse_args()

    if a.url:
        tmp = Path(tempfile.mkdtemp()) / "download.zip"
        print(f"Downloading {a.url} ...")
        urllib.request.urlretrieve(a.url, tmp)
        zip_path = tmp
    else:
        zip_path = Path(a.zip)
    if not Path(zip_path).exists():
        sys.exit(f"{zip_path} not found.")
    if not zipfile.is_zipfile(zip_path):
        sys.exit(f"{zip_path} is not a zip archive. Check the link points at the file itself, not a landing page.")

    n = extract_zip_safely(zip_path, a.dest)
    print(f"Extracted {n} entries into {a.dest}")
    with_metrics = discover_trials(a.dest, require_metrics=True)
    everything = discover_trials(a.dest, require_metrics=False)
    print(f"Found {len(everything)} raw trials, {len(with_metrics)} with a matching *_metrics.csv")
    if not everything:
        sys.exit("No Subject_<id>_T<n>_<task>_raw.csv files found; the archive layout may differ.")
    if len(with_metrics) < len(everything):
        print("Warning: some trials have no metrics file; validation will skip them.")


if __name__ == "__main__":
    main()
