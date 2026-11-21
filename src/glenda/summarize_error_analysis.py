"""Post-hoc descriptive analysis of saved holdout predictions; never refits a model."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss

from glenda.evaluation import metrics
from glenda.evaluation_integrity import verify_results


def summarize(test: Path, ledger: Path, manifest: Path, output: Path, *, allow_legacy_unverified=False) -> None:
    result = json.loads((test / "metrics.json").read_text(encoding="utf-8")); provenance = result["provenance"]
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != provenance["manifest_sha256"]:
        raise ValueError("Manifest differs from evaluated model")
    lock = json.loads((ledger / f"{provenance['holdout_sha256']}.json").read_text(encoding="utf-8"))
    if lock["status"] != "complete" or lock["checkpoint_sha256"] != provenance["checkpoint_sha256"] or Path(lock["output"]).resolve() != test.resolve():
        raise ValueError("Ledger does not match the completed evaluation")
    verify_results(test, lock, allow_legacy_unverified=allow_legacy_unverified)
    with manifest.open(encoding="utf-8", newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    expected = {r["path"]: (r["label"], r["group"]) for r in manifest_rows if r["partition"] == "test"}
    with (test / "predictions.csv").open(encoding="utf-8", newline="") as f:
        predictions = list(csv.DictReader(f))
    if len(predictions) != len(expected) or len({r["path"] for r in predictions}) != len(expected) or any(expected.get(r["path"]) != (r["label"], r["group"]) for r in predictions):
        raise ValueError("Predictions do not match held-out manifest")
    labels = np.asarray([int(r["label"]) for r in predictions]); scores = np.asarray([float(r["score"]) for r in predictions])
    threshold = float(result["point"]["threshold"])
    computed = metrics(labels, scores, threshold)
    if any(computed[key] != result["point"][key] for key in ("tp", "fp", "tn", "fn")):
        raise ValueError("Predictions do not reproduce recorded confusion counts")
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision", "brier"):
        actual, recorded = computed[key], result["point"][key]
        if (actual is None) != (recorded is None) or (actual is not None and not np.isclose(actual, recorded, rtol=0, atol=1e-12)):
            raise ValueError(f"Predictions do not reproduce recorded {key}")
    development = [r for r in manifest_rows if r["partition"] == "development"]
    prevalence = sum(int(r["label"]) for r in development) / len(development)
    baseline = float(brier_score_loss(labels, np.full(len(labels), prevalence)))
    false_negatives = Counter(r["group"] for r in predictions if r["label"] == "1" and float(r["score"]) < threshold)
    quantiles = {str(q): float(np.quantile(scores, q)) for q in (0.25, 0.5, 0.75, 0.99)}
    summary = {"manifest_sha256": provenance["manifest_sha256"], "checkpoint_sha256": provenance["checkpoint_sha256"],
               "predictions_sha256": hashlib.sha256((test / "predictions.csv").read_bytes()).hexdigest(),
               "development_prevalence": prevalence, "constant_prevalence_brier": baseline, "model_brier": computed["brier"],
               "score_quantiles": quantiles, "false_negative_groups": len(false_negatives),
               "largest_false_negative_group": max(false_negatives.values(), default=0), "false_negative_frames": computed["fn"]}
    negative_group_count = len({r["group"] for r in predictions if r["label"] == "0"})
    summary["negative_group_count"] = negative_group_count
    fp_note = "No false positives were observed; this does not prove perfect specificity." if computed["fp"] == 0 else f"There were {computed['fp']} observed false-positive frames."
    lines = ["# Holdout score and error analysis", "",
             "This is descriptive analysis of the saved one-time evaluation; no threshold or model was changed.", "",
             f"The model Brier score was **{computed['brier']:.4f}**. A constant score equal to the development positive-frame fraction ({prevalence:.4f}) has Brier score **{baseline:.4f}** on the same holdout. This is a calibration reference, not a clinical comparator.", "",
             f"The median model score was {quantiles['0.5']:.4f}; its 25th and 75th percentiles were {quantiles['0.25']:.4f} and {quantiles['0.75']:.4f}. The model scores are not calibrated probabilities.", "",
             f"The {computed['fn']} false-negative frames came from {len(false_negatives)} known linked groups; the largest such group contributed {max(false_negatives.values(), default=0)} frames. These are grouped frame errors, not confirmed independent patients.", "",
             f"{fp_note} Only {negative_group_count} known groups supplied negative frames, and the threshold was selected from development-fold score scales; it may not transfer reliably to a refitted model.\n"]
    if "artifact_sha256" not in lock:
        lines[2:2] = ["**Legacy result:** the original ledger did not record output hashes; result-file integrity has not been verified by that ledger.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("test", "ledger", "manifest", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--allow-legacy-unverified", action="store_true", help="Acknowledge that old ledgers have no output hashes")
    a = p.parse_args(); summarize(a.test, a.ledger, a.manifest, a.output, allow_legacy_unverified=a.allow_legacy_unverified)


if __name__ == "__main__":
    main()
