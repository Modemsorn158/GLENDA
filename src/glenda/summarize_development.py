"""Summarize completed development folds without publishing image-level data."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from sklearn.metrics import average_precision_score


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(search, manifest, output):
    identity_path, results_path = search / "search.json", search / "fold_results.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if digest(manifest) != identity["manifest_sha256"]:
        raise ValueError("Manifest differs from the recorded search")
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["partition"] == "development"]
    expected = {row["path"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate development path")
    if len(results) != len(identity["configurations"]) * 5:
        raise ValueError("Search has incomplete or unexpected fold results")
    trials = []
    for trial, configuration in enumerate(identity["configurations"]):
        folds = []
        for fold in range(5):
            matches = [row for row in results if row["trial"] == trial and row["fold"] == fold]
            if len(matches) != 1 or matches[0]["status"] != "complete":
                raise ValueError(f"Trial {trial} fold {fold} did not complete")
            folder = search / f"trial_{trial:03d}" / f"fold_{fold}"
            status = json.loads((folder.parent / f"fold_{fold}_status.json").read_text(encoding="utf-8"))
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            if status != {"status": "complete", "best_validation_average_precision": matches[0]["best_validation_average_precision"]}:
                raise ValueError("Fold status differs from search results")
            if (config["fold"] != fold or config["manifest_sha256"] != identity["manifest_sha256"] or
                any(config.get(k) != v for k, v in configuration.items())):
                raise ValueError("Fold configuration differs from search")
            history_file, prediction_file = folder / "epochs.csv", folder / "validation_predictions.csv"
            with history_file.open(encoding="utf-8", newline="") as handle:
                history = list(csv.DictReader(handle))
            with prediction_file.open(encoding="utf-8", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            paths = [row["path"] for row in predictions]
            fold_paths = {path for path, row in expected.items() if row["cv_fold"] == str(fold)}
            if len(paths) != len(set(paths)) or set(paths) != fold_paths:
                raise ValueError("Predictions do not exactly cover the validation fold")
            if any(row["label"] != expected[row["path"]]["label"] or row["group"] != expected[row["path"]]["group"] for row in predictions):
                raise ValueError("Prediction labels or groups differ from manifest")
            scores = [float(row["score"]) for row in predictions]
            if not scores or any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
                raise ValueError("Invalid prediction scores")
            if not history or not (folder / "best.pt").is_file():
                raise ValueError("Missing training history or checkpoint")
            best = max(history, key=lambda row: float(row["validation_average_precision"]))
            recorded = float(matches[0]["best_validation_average_precision"])
            calculated = average_precision_score([int(row["label"]) for row in predictions], scores)
            if not math.isclose(recorded, float(best["validation_average_precision"]), abs_tol=1e-12, rel_tol=0) or not math.isclose(recorded, calculated, abs_tol=1e-12, rel_tol=0):
                raise ValueError("Fold AP does not match its history and predictions")
            folds.append({"fold": fold, "best_ap": recorded, "best_epoch": int(best["epoch"]),
                          "epochs_run": len(history), "history_sha256": digest(history_file),
                          "predictions_sha256": digest(prediction_file), "checkpoint_sha256": digest(folder / "best.pt")})
        values = [item["best_ap"] for item in folds]
        trials.append({"configuration": configuration, "folds": folds, "mean_ap": mean(values), "sample_sd": stdev(values)})
    report = {"manifest_sha256": identity["manifest_sha256"], "search_sha256": digest(identity_path),
              "fold_results_sha256": digest(results_path), "trials": trials}
    lines = ["# Extended development training results", "",
             "These are adaptively selected development-fold scores, not an independent or clinical performance estimate. "
             "The original holdout was already consumed and was not reevaluated. Patient independence is not established.", "",
             "| Fine-tuning | Fold 0 AP | Fold 1 AP | Fold 2 AP | Fold 3 AP | Fold 4 AP | Mean AP | Sample SD | Best epochs | Epochs run |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for trial in trials:
        folds = trial["folds"]
        lines.append("| " + trial["configuration"]["fine_tuning"] + " | " + " | ".join(f"{f['best_ap']:.4f}" for f in folds) +
                     f" | {trial['mean_ap']:.4f} | {trial['sample_sd']:.4f} | " + ", ".join(str(f["best_epoch"]) for f in folds) +
                     " | " + ", ".join(str(f["epochs_run"]) for f in folds) + " |")
    lines.extend(["", "Each best epoch was chosen using its own validation fold; this is optimistic. "
                  "Fixed learning rate isolates the fine-tuning setting only within this limited comparison, and does not establish convergence or clinical utility. "
                  "See the [protocol](deep_development_protocol.md) and the adjacent JSON for configuration and artifact hashes.", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("search", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.search, args.manifest, args.output)


if __name__ == "__main__":
    main()
