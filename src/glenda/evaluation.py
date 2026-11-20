"""Metric primitives for predictions, including a cluster bootstrap.

These functions do not choose a threshold or load the held-out split.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def metrics(labels, scores, threshold: float) -> dict:
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and scores must be equally sized nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected binary labels and finite scores in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    pred = p >= threshold
    tp, fn = int(np.sum((y == 1) & pred)), int(np.sum((y == 1) & ~pred))
    fp, tn = int(np.sum((y == 0) & pred)), int(np.sum((y == 0) & ~pred))
    def ratio(a, b):
        return float(a / b) if b else None
    recall, specificity, precision = ratio(tp, tp + fn), ratio(tn, tn + fp), ratio(tp, tp + fp)
    return {"n": len(y), "prevalence": float(np.mean(y)), "threshold": threshold,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": (tp + tn) / len(y),
            "precision": precision, "recall": recall, "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
            "brier": float(brier_score_loss(y, p))}


def cluster_bootstrap(labels, scores, groups, threshold: float, samples: int = 1000, seed: int = 42) -> dict:
    """Percentile intervals by resampling entire clusters; undefined replicates are skipped."""
    y, p, g = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    if len(y) != len(p) or len(y) != len(g) or samples < 1:
        raise ValueError("Inputs must have equal lengths and samples must be positive")
    point = metrics(y, p, threshold)
    unique = np.unique(g)
    if len(unique) < 2:
        raise ValueError("At least two clusters are required")
    index = {key: np.flatnonzero(g == key) for key in unique}
    rng = np.random.default_rng(seed)
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision", "brier"]
    draws = {key: [] for key in keys}
    for _ in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([index[key] for key in sampled])
        result = metrics(y[indices], p[indices], threshold)
        for key in keys:
            if result[key] is not None:
                draws[key].append(result[key])
    intervals = {key: {"lower": float(np.percentile(values, 2.5)) if values else None,
                       "upper": float(np.percentile(values, 97.5)) if values else None,
                       "valid_replicates": len(values)} for key, values in draws.items()}
    return {"point": point, "intervals": intervals, "requested_replicates": samples, "seed": seed,
            "note": "Cluster bootstrap intervals depend on the supplied clusters; unknown patient links remain unaccounted for."}
