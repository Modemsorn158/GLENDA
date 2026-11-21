"""Select an F1 threshold from a selected trial's out-of-fold predictions only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def choose_threshold(labels, scores, minimum_sensitivity=None):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom != 0)
    if minimum_sensitivity is None:
        best = np.flatnonzero(np.isclose(f1, np.max(f1), rtol=0, atol=1e-12))[-1]
    else:
        if not 0 < minimum_sensitivity <= 1:
            raise ValueError("Minimum sensitivity must be in (0, 1]")
        candidates = np.flatnonzero(recall[:-1] >= minimum_sensitivity)
        if not len(candidates):
            raise ValueError("No threshold meets the sensitivity requirement")
        best = candidates[-1]
    return float(thresholds[best]), float(f1[best])


def select(manifest: Path, trial: Path, output: Path, minimum_sensitivity=None) -> dict:
    if output.exists():
        raise FileExistsError("Refusing to overwrite threshold selection")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    search = json.loads((trial.parent / "search.json").read_text(encoding="utf-8"))
    if search["manifest_sha256"] != manifest_hash:
        raise ValueError("Trial search was run with a different manifest")
    if not trial.name.startswith("trial_") or not trial.name[6:].isdigit():
        raise ValueError("Unexpected trial directory name")
    trial_index = int(trial.name[6:])
    if trial_index >= len(search["configurations"]):
        raise ValueError("Trial index is not in search configuration")
    selected_config = search["configurations"][trial_index]
    candidate_file = trial.parent / "selected_candidate.json"
    candidate = json.loads(candidate_file.read_text(encoding="utf-8"))
    if candidate["trial_index"] != trial_index or candidate["configuration"] != selected_config:
        raise ValueError("Threshold trial is not the frozen selected candidate")
    if candidate["search_sha256"] != hashlib.sha256((trial.parent / "search.json").read_bytes()).hexdigest() or candidate["fold_results_sha256"] != hashlib.sha256((trial.parent / "fold_results.json").read_bytes()).hexdigest():
        raise ValueError("Search results changed after candidate selection")
    with manifest.open(encoding="utf-8", newline="") as f:
        development = {r["path"]: r for r in csv.DictReader(f) if r["partition"] == "development"}
    predictions, prediction_hashes, epoch_hashes, best_epochs = {}, [], [], []
    for fold in range(5):
        status = json.loads((trial / f"fold_{fold}_status.json").read_text(encoding="utf-8"))
        if status["status"] != "complete":
            raise ValueError(f"Fold {fold} did not complete")
        config = json.loads((trial / f"fold_{fold}" / "config.json").read_text(encoding="utf-8"))
        if config["manifest_sha256"] != manifest_hash or config["fold"] != fold or any(config.get(k) != v for k, v in selected_config.items()):
            raise ValueError("Fold configuration does not match the selected trial")
        if "inventory_sha256" in search and (config.get("inventory_sha256") != search["inventory_sha256"] or config.get("candidates_sha256") != search.get("candidates_sha256")):
            raise ValueError("Fold input provenance differs from the selected search")
        prediction_file = trial / f"fold_{fold}" / "validation_predictions.csv"
        prediction_hashes.append(hashlib.sha256(prediction_file.read_bytes()).hexdigest())
        fold_labels, fold_scores = [], []
        with prediction_file.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                path = row["path"]
                if path in predictions or path not in development:
                    raise ValueError("Predictions have duplicate or non-development paths")
                expected = development[path]
                if expected["cv_fold"] != str(fold) or row["label"] != expected["label"] or row["group"] != expected["group"]:
                    raise ValueError("Prediction does not match manifest fold, label or group")
                predictions[path] = row
                fold_labels.append(int(row["label"])); fold_scores.append(float(row["score"]))
        if set(fold_labels) != {0, 1} or not np.isfinite(fold_scores).all() or any(not 0 <= score <= 1 for score in fold_scores):
            raise ValueError("Fold predictions must contain both classes and finite probability scores")
        if not np.isclose(average_precision_score(fold_labels, fold_scores), status["best_validation_average_precision"], rtol=0, atol=1e-12):
            raise ValueError("Fold predictions do not reproduce recorded average precision")
        epoch_file = trial / f"fold_{fold}" / "epochs.csv"
        epoch_hashes.append(hashlib.sha256(epoch_file.read_bytes()).hexdigest())
        with epoch_file.open(encoding="utf-8", newline="") as f:
            history = list(csv.DictReader(f))
        if not history:
            raise ValueError("Fold epoch history is empty")
        best_row = max(history, key=lambda row: float(row["validation_average_precision"]))
        if not np.isclose(float(best_row["validation_average_precision"]), status["best_validation_average_precision"], rtol=0, atol=1e-12):
            raise ValueError("Epoch history does not match fold status")
        best_epoch = int(best_row["epoch"])
        if best_epoch < 1:
            raise ValueError("Invalid best epoch")
        best_epochs.append(best_epoch)
    if set(predictions) != set(development):
        raise ValueError("Predictions must cover the development set exactly once")
    labels = np.asarray([int(row["label"]) for row in predictions.values()])
    scores = np.asarray([float(row["score"]) for row in predictions.values()])
    if set(labels) != {0, 1} or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected both classes and finite probability scores")
    threshold, selected_f1 = choose_threshold(labels, scores, minimum_sensitivity)
    result = {"threshold": threshold, "oof_f1": selected_f1, "n": len(labels),
              "trial_index": trial_index, "trial_configuration": selected_config,
              "candidate_selection_sha256": hashlib.sha256(candidate_file.read_bytes()).hexdigest(),
              "fold_prediction_sha256": prediction_hashes,
              "fold_epoch_history_sha256": epoch_hashes, "fold_best_epochs": best_epochs,
              "rule": ("Maximize pooled development out-of-fold F1; ties choose highest threshold" if minimum_sensitivity is None else
                       "Highest pooled development out-of-fold threshold meeting minimum sensitivity"),
              "minimum_sensitivity": minimum_sensitivity,
              "manifest_sha256": manifest_hash,
              "warning": "Threshold and F1 are selected on the same out-of-fold predictions; F1 is optimistic and is not a test result."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path); parser.add_argument("trial", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--min-sensitivity", type=float, help="Prespecified development sensitivity requirement in (0, 1]")
    args = parser.parse_args()
    print(json.dumps(select(args.manifest, args.trial, args.output, args.min_sensitivity), indent=2))


if __name__ == "__main__":
    main()
