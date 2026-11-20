"""Describe score-scale variation at a frozen development-selected threshold."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from glenda.evaluation import metrics


def summarize(trial: Path, selection: Path, output: Path) -> None:
    chosen = json.loads(selection.read_text(encoding="utf-8")); threshold = chosen["threshold"]
    if trial.name != f"trial_{chosen['trial_index']:03d}":
        raise ValueError("Threshold belongs to a different trial")
    rows_out = []
    lines = ["# Out-of-fold score-scale diagnostics", "",
             f"The development-selected threshold is {threshold:.8f}; the pooled F1 at this selected threshold is {chosen['oof_f1']:.4f}. "
             "That F1 is selection data, not independent performance.", "",
             "| Fold | Median score | 99th percentile | Scores exactly 1 | Predicted positive | Recall at fixed threshold |",
             "|---:|---:|---:|---:|---:|---:|"]
    for fold in range(5):
        path = trial / f"fold_{fold}" / "validation_predictions.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != chosen["fold_prediction_sha256"][fold]:
            raise ValueError("Predictions changed after threshold selection")
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        scores = np.asarray([float(r["score"]) for r in rows]); labels = [int(r["label"]) for r in rows]
        result = metrics(labels, scores, threshold)
        item = {"fold": fold, "median_score": float(np.median(scores)), "p99_score": float(np.quantile(scores, 0.99)),
                "scores_exactly_one": int((scores == 1).sum()), "predicted_positive": result["tp"] + result["fp"],
                "recall": result["recall"]}
        rows_out.append(item)
        lines.append(f"| {fold} | {item['median_score']:.7f} | {item['p99_score']:.7f} | {item['scores_exactly_one']} | {item['predicted_positive']} | {item['recall']:.4f} |")
    lines.extend(["", "Scores are not calibrated probabilities. Differences between fold models and the final refit can make a pooled threshold unstable; "
                  "do not retune it using the holdout.\n"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps({"selection": chosen, "folds": rows_out}, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trial", type=Path); p.add_argument("selection", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); summarize(a.trial, a.selection, a.output)


if __name__ == "__main__":
    main()
