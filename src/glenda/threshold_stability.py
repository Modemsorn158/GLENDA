"""Explore threshold transfer between saved development-fold predictions only.

This is a diagnostic, not a new candidate or threshold selection protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

from glenda.evaluation import metrics


def _f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if set(labels) != {0, 1} or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected both classes and finite scores in [0, 1]")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator, out=np.zeros_like(denominator), where=denominator != 0)
    best = np.flatnonzero(np.isclose(f1, np.max(f1), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def analyze(manifest: Path, trial: Path, selection: Path, output: Path) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Refusing to overwrite diagnostic output")
    selected = json.loads(selection.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if manifest_hash != selected["manifest_sha256"] or trial.name != f"trial_{selected['trial_index']:03d}":
        raise ValueError("Selected threshold does not match manifest and trial")
    candidate = trial.parent / "selected_candidate.json"
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != selected["candidate_selection_sha256"]:
        raise ValueError("Candidate selection changed")
    with manifest.open(encoding="utf-8", newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    development = {row["path"]: row for row in manifest_rows if row["partition"] == "development"}
    if len(development) != sum(row["partition"] == "development" for row in manifest_rows):
        raise ValueError("Duplicate development paths")
    folds, seen = [], set()
    for fold in range(5):
        prediction_file = trial / f"fold_{fold}" / "validation_predictions.csv"
        if hashlib.sha256(prediction_file.read_bytes()).hexdigest() != selected["fold_prediction_sha256"][fold]:
            raise ValueError("Fold predictions changed after threshold selection")
        with prediction_file.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            path = row["path"]
            if path in seen or path not in development:
                raise ValueError("Duplicate or non-development prediction")
            expected = development[path]
            if expected["cv_fold"] != str(fold) or expected["label"] != row["label"] or expected["group"] != row["group"]:
                raise ValueError("Prediction does not match the development manifest")
            seen.add(path)
        labels = np.asarray([int(row["label"]) for row in rows])
        scores = np.asarray([float(row["score"]) for row in rows])
        _f1_threshold(labels, scores)  # Fail explicitly if a fold is unsuitable.
        folds.append((labels, scores))
    if seen != set(development):
        raise ValueError("Predictions do not cover development exactly once")

    diagnostics = []
    for held_out in range(5):
        train_labels = np.concatenate([f[0] for i, f in enumerate(folds) if i != held_out])
        train_scores = np.concatenate([f[1] for i, f in enumerate(folds) if i != held_out])
        threshold = _f1_threshold(train_labels, train_scores)
        labels, scores = folds[held_out]
        diagnostics.append({"held_out_fold": held_out, "threshold_from_other_folds": threshold,
                            "own_fold_f1_threshold": _f1_threshold(labels, scores),
                            "transferred_metrics": metrics(labels, scores, threshold)})
    result = {"manifest_sha256": manifest_hash, "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
              "original_pooled_threshold": selected["threshold"], "folds": diagnostics,
              "warning": "Exploratory development-only diagnostic; candidate selection used all folds, and models "
                         "producing the other-fold scores were trained on the target fold. It is not a nested estimate, "
                         "final-refit calibration, or an independent evaluation."}
    lines = ["# Development-only threshold transfer diagnostic", "",
             "For each fold, the F1-maximizing threshold is fitted on the other four saved out-of-fold score sets "
             "and applied to this fold. Candidate selection used all folds, and the models producing the other-fold scores "
             "were trained on the target fold. This is not a nested or independent estimate.", "",
             "| Held-out fold | Threshold from other folds | Fold-only threshold (descriptive) | Precision | Recall | F1 |",
             "|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "undefined" if value is None else f"{value:.4f}"
    for row in diagnostics:
        m = row["transferred_metrics"]
        lines.append(f"| {row['held_out_fold']} | {row['threshold_from_other_folds']:.8f} | {row['own_fold_f1_threshold']:.8f} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} |")
    lines.extend(["", "The fold-only thresholds use the same fold labels and are descriptive, not valid deployment thresholds. "
                  "This diagnostic does not measure threshold transfer to the final refit and does not establish calibration. "
                  "Only development rows are used for the analysis; no test images or predictions are used, and the original model and threshold are unchanged.\n"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path); p.add_argument("trial", type=Path)
    p.add_argument("selection", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); analyze(a.manifest, a.trial, a.selection, a.output)


if __name__ == "__main__":
    main()
