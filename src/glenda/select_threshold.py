"""Select an F1 threshold from a selected trial's out-of-fold predictions only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve


def select(manifest: Path, trial: Path, output: Path) -> dict:
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
    with manifest.open(encoding="utf-8", newline="") as f:
        development = {r["path"]: r for r in csv.DictReader(f) if r["partition"] == "development"}
    predictions, prediction_hashes = {}, []
    for fold in range(5):
        status = json.loads((trial / f"fold_{fold}_status.json").read_text(encoding="utf-8"))
        if status["status"] != "complete":
            raise ValueError(f"Fold {fold} did not complete")
        config = json.loads((trial / f"fold_{fold}" / "config.json").read_text(encoding="utf-8"))
        if config["manifest_sha256"] != manifest_hash or config["fold"] != fold or any(config.get(k) != v for k, v in selected_config.items()):
            raise ValueError("Fold configuration does not match the selected trial")
        prediction_file = trial / f"fold_{fold}" / "validation_predictions.csv"
        prediction_hashes.append(hashlib.sha256(prediction_file.read_bytes()).hexdigest())
        with prediction_file.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                path = row["path"]
                if path in predictions or path not in development:
                    raise ValueError("Predictions have duplicate or non-development paths")
                expected = development[path]
                if expected["cv_fold"] != str(fold) or row["label"] != expected["label"] or row["group"] != expected["group"]:
                    raise ValueError("Prediction does not match manifest fold, label or group")
                predictions[path] = row
    if set(predictions) != set(development):
        raise ValueError("Predictions must cover the development set exactly once")
    labels = np.asarray([int(row["label"]) for row in predictions.values()])
    scores = np.asarray([float(row["score"]) for row in predictions.values()])
    if set(labels) != {0, 1} or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected both classes and finite probability scores")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom != 0)
    best = np.flatnonzero(np.isclose(f1, np.max(f1), rtol=0, atol=1e-12))[-1]
    result = {"threshold": float(thresholds[best]), "oof_f1": float(f1[best]), "n": len(labels),
              "trial_index": trial_index, "trial_configuration": selected_config,
              "fold_prediction_sha256": prediction_hashes,
              "rule": "Maximize pooled development out-of-fold F1; ties choose highest threshold",
              "manifest_sha256": manifest_hash,
              "warning": "Threshold and F1 are selected on the same out-of-fold predictions; F1 is optimistic and is not a test result."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path); parser.add_argument("trial", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(select(args.manifest, args.trial, args.output), indent=2))


if __name__ == "__main__":
    main()
