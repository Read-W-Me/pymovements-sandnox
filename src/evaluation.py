"""Agreement statistics and event matching for validating a classifier."""
from __future__ import annotations

import numpy as np


def relative_error(pred, gt, floor: float = 1.0):
    """|pred - gt| / max(|gt|, floor). The floor keeps tiny/zero counts finite."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    return np.abs(pred - gt) / np.maximum(np.abs(gt), floor)


def agreement_stats(pred, gt) -> dict:
    """Trial-level agreement: MAE, mean relative error, Pearson r, Bland-Altman bias/limits."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    ok = ~(np.isnan(pred) | np.isnan(gt))
    pred, gt = pred[ok], gt[ok]
    n = int(len(pred))
    nan = float("nan")
    if n == 0:
        return {"n": 0, "mae": nan, "mean_rel_error": nan, "bias": nan,
                "loa_low": nan, "loa_high": nan, "pearson_r": nan}
    diff = pred - gt
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    r = nan
    if n >= 3 and pred.std() > 0 and gt.std() > 0:
        r = float(np.corrcoef(pred, gt)[0, 1])
    return {
        "n": n,
        "mae": float(np.abs(diff).mean()),
        "mean_rel_error": float(relative_error(pred, gt).mean()),
        "bias": float(diff.mean()),
        "loa_low": float(diff.mean() - 1.96 * sd),
        "loa_high": float(diff.mean() + 1.96 * sd),
        "pearson_r": r,
    }


def match_events(pred, gt, min_iou: float = 0.5) -> dict:
    """One-to-one event matching by interval IoU (greedy, best pairs first).

    pred, gt: arrays of shape (n, 2) with (onset, offset) in the same unit.
    """
    pred = np.asarray(pred, float).reshape(-1, 2)
    gt = np.asarray(gt, float).reshape(-1, 2)
    if len(pred) == 0 or len(gt) == 0:
        tp = 0
        ious = []
    else:
        inter = np.clip(np.minimum(pred[:, None, 1], gt[None, :, 1])
                        - np.maximum(pred[:, None, 0], gt[None, :, 0]), 0, None)
        union = (pred[:, None, 1] - pred[:, None, 0]) + (gt[None, :, 1] - gt[None, :, 0]) - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(union > 0, inter / union, 0.0)
        used_p, used_g, ious = set(), set(), []
        for flat in np.argsort(-iou, axis=None):
            i, j = np.unravel_index(flat, iou.shape)
            if iou[i, j] < min_iou:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            ious.append(float(iou[i, j]))
        tp = len(ious)
    fp, fn = len(pred) - tp, len(gt) - tp
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if tp and precision + recall > 0 else (0.0 if (tp + fp + fn) else float("nan")))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "mean_iou": float(np.mean(ious)) if ious else float("nan")}


def sample_labels(events, n_samples: int) -> np.ndarray:
    """Boolean per-sample array, True inside any (onset, offset) sample-index interval."""
    labels = np.zeros(n_samples, dtype=bool)
    for onset, offset in np.asarray(events, int).reshape(-1, 2):
        labels[max(onset, 0): min(offset, n_samples - 1) + 1] = True
    return labels


def cohens_kappa(a, b) -> float:
    """Cohen's kappa for two binary label arrays (sample-level event agreement)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("label arrays must be non-empty and the same shape")
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def split_subjects(subjects, test_fraction: float = 0.5, seed: int = 0):
    """Deterministic subject-level split -> (train_subjects, test_subjects)."""
    uniq = sorted(set(subjects))
    if len(uniq) < 2:
        raise ValueError("need at least 2 subjects to split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_test = min(max(1, int(round(len(uniq) * test_fraction))), len(uniq) - 1)
    test = sorted(uniq[i] for i in order[:n_test])
    train = sorted(uniq[i] for i in order[n_test:])
    return train, test
