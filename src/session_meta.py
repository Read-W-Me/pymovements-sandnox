"""
Side file that records the geometry of a gaze recording.

Why: the gaze CSV only has normalised screen coords (0-1). To get pixels and
degrees we need the screen size, the viewing distance, and where the text lines
are. Without this file the pipeline has to guess.

Two ways to create it:

1) From your recording script, right when you save the CSV (recommended):

       from session_meta import save_session_meta
       save_session_meta(
           "gaze_log.csv",                  # the recording path; writes gaze_log_meta.json
           width_px=SW, height_px=SH,       # real screen resolution
           width_cm=34.5, height_cm=19.4,   # measure your screen
           distance_cm=50,                  # eyes -> screen
           line_centres=reading["centres"], # y of each text line, in px
       )

2) By hand after the fact:

       python session_meta.py gaze_log.csv --width-px 1920 --height-px 1080 \
           --width-cm 34.5 --height-cm 19.4 --distance-cm 50 \
           --centres 200,300,400,500
"""
import argparse
import json
from pathlib import Path

REQUIRED = ["width_px", "height_px", "width_cm", "height_cm", "distance_cm", "line_centres"]


def meta_path_for(recording_path):
    p = Path(recording_path)
    return p.with_name(p.stem + "_meta.json")


def save_session_meta(recording_path, *, width_px, height_px, width_cm, height_cm,
                      distance_cm, line_centres, notes=""):
    meta = dict(width_px=int(width_px), height_px=int(height_px),
                width_cm=float(width_cm), height_cm=float(height_cm),
                distance_cm=float(distance_cm),
                line_centres=[float(c) for c in line_centres], notes=notes)
    out = meta_path_for(recording_path)
    out.write_text(json.dumps(meta, indent=2))
    return out


def load_session_meta(path):
    meta = json.loads(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(missing)}")
    if len(meta["line_centres"]) < 1:
        raise ValueError("line_centres is empty")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write the _meta.json side file for a recording.")
    ap.add_argument("recording")
    ap.add_argument("--width-px", type=int, required=True)
    ap.add_argument("--height-px", type=int, required=True)
    ap.add_argument("--width-cm", type=float, required=True)
    ap.add_argument("--height-cm", type=float, required=True)
    ap.add_argument("--distance-cm", type=float, required=True)
    ap.add_argument("--centres", required=True, help="comma-separated line centre y values in px")
    a = ap.parse_args()
    out = save_session_meta(a.recording, width_px=a.width_px, height_px=a.height_px,
                            width_cm=a.width_cm, height_cm=a.height_cm,
                            distance_cm=a.distance_cm,
                            line_centres=[float(c) for c in a.centres.split(",")])
    print("wrote", out)