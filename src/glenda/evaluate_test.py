"""Evaluate a fixed final checkpoint once, using a manifest-specific ledger."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from glenda.dataset import DevelopmentDataset
from glenda.evaluation import cluster_bootstrap, metrics
from glenda.inference import load_final, predict_image


def evaluate(raw: Path, manifest: Path, checkpoint: Path, output: Path, ledger: Path, bootstrap: int = 1000) -> None:
    raw, output, ledger = raw.resolve(), output.resolve(), ledger.resolve()
    if any(path == raw or raw in path.parents for path in (output, ledger)):
        raise ValueError("Output and ledger must be outside raw data")
    if output.exists():
        raise FileExistsError("Refusing to overwrite test evaluation")
    if bootstrap < 1:
        raise ValueError("Bootstrap sample count must be positive")
    os.environ["MPLCONFIGDIR"] = str(output.parent / ".mpl-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_final(checkpoint, device)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if config["manifest_sha256"] != digest:
        raise ValueError("Checkpoint and manifest differ")
    # Validates uniqueness and known group separation without opening images.
    DevelopmentDataset(raw, manifest, 0, True)
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["partition"] == "test"]
    if not rows or {r["label"] for r in rows} != {"0", "1"}:
        raise ValueError("Held-out split must contain both labels")
    if len({r["group"] for r in rows}) < 2:
        raise ValueError("Held-out split needs at least two independent groups for bootstrap")
    ledger.mkdir(parents=True, exist_ok=True)
    lock = ledger / f"{digest}.json"
    with lock.open("x", encoding="utf-8") as f:
        json.dump({"manifest_sha256": digest, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   "output": str(output), "threshold": config["threshold"], "status": "started"}, f, indent=2)
    # An interrupted evaluation remains locked and requires documented investigation.
    output.mkdir(parents=True)
    predictions = []
    for row in rows:
        path = (raw / row["path"]).resolve()
        if raw not in path.parents:
            raise ValueError("Image path escapes dataset root")
        with Image.open(path) as im:
            score = predict_image(model, im, device)
        predictions.append({"path": row["path"], "label": int(row["label"]), "score": score, "group": row["group"]})
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "score", "group"]); writer.writeheader(); writer.writerows(predictions)
    labels = np.asarray([p["label"] for p in predictions]); scores = np.asarray([p["score"] for p in predictions])
    groups = [p["group"] for p in predictions]; threshold = float(config["threshold"])
    result = cluster_bootstrap(labels, scores, groups, threshold, samples=bootstrap)
    result["fixed_threshold_sensitivity"] = [metrics(labels, scores, t) for t in (0.1, 0.25, 0.5, 0.75, 0.9)]
    result["warning"] = "Dataset frame prevalence is curated and is not population prevalence; unknown patient links can affect uncertainty."
    point = result["point"]
    result["illustration_per_1000_at_test_frame_prevalence"] = {
        "false_positives": 1000 * (1 - point["prevalence"]) * (1 - point["specificity"]),
        "false_negatives": 1000 * point["prevalence"] * (1 - point["recall"]),
        "note": "Descriptive re-expression of this test set, not a clinical population estimate."}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, predicate in (("false_positives", lambda p: p["label"] == 0 and p["score"] >= threshold),
                            ("false_negatives", lambda p: p["label"] == 1 and p["score"] < threshold)):
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "score", "group"]); writer.writeheader()
            writer.writerows(sorted((p for p in predictions if predicate(p)), key=lambda p: p["score"], reverse=True))
    fpr, tpr, _ = roc_curve(labels, scores); precision, recall, _ = precision_recall_curve(labels, scores)
    observed, predicted = calibration_curve(labels, scores, n_bins=10, strategy="uniform")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], "--", color="gray"); axes[0].set(xlabel="False positive rate", ylabel="Sensitivity", title="ROC")
    axes[1].plot(recall, precision); axes[1].axhline(float(labels.mean()), linestyle="--", color="gray"); axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[2].plot(predicted, observed, marker="o"); axes[2].plot([0, 1], [0, 1], "--", color="gray"); axes[2].set(xlabel="Mean score", ylabel="Observed fraction", title="Calibration")
    fig.tight_layout(); fig.savefig(output / "evaluation.png", dpi=150); plt.close(fig)
    lock.write_text(json.dumps({"manifest_sha256": digest, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                "output": str(output), "threshold": threshold, "status": "complete"}, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "manifest", "checkpoint", "output", "ledger"):
        p.add_argument(name, type=Path)
    p.add_argument("--bootstrap", type=int, default=1000)
    a = p.parse_args(); evaluate(a.raw, a.manifest, a.checkpoint, a.output, a.ledger, a.bootstrap)


if __name__ == "__main__":
    main()
