"""Summarize recorded study hashes without publishing images or case identifiers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(reports):
    archive = read_json(reports / "archive_check.json")
    split = read_json(reports / "split_v2_summary.json")
    tuning = read_json(reports / "tuning_results.json")
    threshold = read_json(reports / "oof_score_diagnostics.json")["selection"]
    final = read_json(reports / "final_fit.json")
    error = read_json(reports / "error_analysis.json")
    test = read_json(reports / "test_results.json")
    provenance = test["provenance"]
    manifest = split["manifest_sha256"]
    if any(value != manifest for value in (tuning["identity"]["manifest_sha256"],
           threshold["manifest_sha256"], final["manifest_sha256"], error["manifest_sha256"],
           provenance["manifest_sha256"])):
        raise ValueError("Recorded manifest hashes disagree")
    if (split["holdout_sha256"] != provenance["holdout_sha256"] or
        final["final_checkpoint_sha256"] != error["checkpoint_sha256"] or
        final["final_checkpoint_sha256"] != provenance["checkpoint_sha256"] or
        final["threshold"] != threshold["threshold"] or
        final["threshold"] != test["point"]["threshold"] or
        final["cv_training_source_sha256"] != tuning["identity"]["training_source_sha256"]):
        raise ValueError("Recorded study provenance disagrees")
    return {
        "description": "Recorded SHA-256 identifiers; associated private data and artifacts are not distributed here.",
        "limitations": "Hashes support integrity checks, not patient independence, clinical validity, or reproducibility without the original artifacts.",
        "archives": archive["archive_sha256"],
        "inventory_sha256": archive["inventory_sha256"],
        "parent_manifest_sha256": split["parent_manifest_sha256"],
        "manifest_sha256": manifest,
        "holdout_sha256": split["holdout_sha256"],
        "search_sha256": tuning["selection"]["search_sha256"],
        "fold_results_sha256": tuning["selection"]["fold_results_sha256"],
        "fold_prediction_sha256": threshold["fold_prediction_sha256"],
        "threshold_file_sha256": final["threshold_file_sha256"],
        "pretrained_weights_sha256": final["pretrained_weights_sha256"],
        "checkpoint_sha256": final["final_checkpoint_sha256"],
        "test_predictions_sha256": error["predictions_sha256"],
        "training_source_sha256": final["cv_training_source_sha256"],
        "evaluation_source_sha256": provenance["evaluation_source_sha256"],
    }


def verify_local(root, manifest):
    outputs = root / "outputs"
    expected = {
        "audit/inventory.csv": manifest["inventory_sha256"],
        "splits/v1/manifest.csv": manifest["parent_manifest_sha256"],
        "splits/v2/manifest.csv": manifest["manifest_sha256"],
        "search/tuning_v2/search.json": manifest["search_sha256"],
        "search/tuning_v2/fold_results.json": manifest["fold_results_sha256"],
        "search/tuning_v2/selected_threshold.json": manifest["threshold_file_sha256"],
        "final/selected/final.pt": manifest["checkpoint_sha256"],
        "test/selected/predictions.csv": manifest["test_predictions_sha256"],
    }
    for fold, digest in enumerate(manifest["fold_prediction_sha256"]):
        expected[f"search/tuning_v2/trial_001/fold_{fold}/validation_predictions.csv"] = digest
    for relative, digest in expected.items():
        path = outputs / relative
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != digest:
            raise ValueError(f"Hash mismatch: {relative}")
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--verify-local", action="store_true", help="Hash existing private artifacts without running inference")
    parser.add_argument("--write", action="store_true", help="Write the public hash summary")
    args = parser.parse_args()
    manifest = build_manifest(args.root / "reports")
    if args.verify_local:
        print(f"Verified {verify_local(args.root, manifest)} local artifact hashes")
    if args.write:
        (args.root / "reports" / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    elif not args.verify_local:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
