"""Evaluate a fixed final checkpoint once, using a manifest-specific ledger."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
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
        all_rows = list(csv.DictReader(f))
    if any(r["label"] not in {"0", "1"} for r in all_rows):
        raise ValueError("Manifest labels must be binary")
    release_prevalence = sum(int(r["label"]) for r in all_rows) / len(all_rows)
    rows = [r for r in all_rows if r["partition"] == "test"]
    if not rows or {r["label"] for r in rows} != {"0", "1"}:
        raise ValueError("Held-out split must contain both labels")
    if len({r["group"] for r in rows}) < 2:
        raise ValueError("Held-out split needs at least two independent groups for bootstrap")
    heldout_paths = sorted(row["path"] for row in rows)
    heldout_digest = hashlib.sha256(json.dumps(heldout_paths, separators=(",", ":")).encode("utf-8")).hexdigest()
    source_dir = Path(__file__).parent
    source_hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                     for name in ("evaluate_test.py", "evaluation.py", "inference.py", "dataset.py", "models.py")}
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    audit = {"manifest_sha256": digest, "holdout_sha256": heldout_digest, "checkpoint_sha256": checkpoint_digest,
             "evaluation_source_sha256": source_hashes, "device": str(device), "torch_version": str(torch.__version__),
             "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__, "python_version": platform.python_version()}
    ledger.mkdir(parents=True, exist_ok=True)
    lock = ledger / f"{heldout_digest}.json"
    with lock.open("x", encoding="utf-8") as f:
        json.dump({**audit, "output": str(output), "threshold": config["threshold"], "status": "started"}, f, indent=2)
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
    result["provenance"] = {**audit,
                            "known_groups": len(set(groups)), "groups_with_negative_frames": len({groups[i] for i in range(len(groups)) if labels[i] == 0})}
    result["fixed_threshold_sensitivity"] = [metrics(labels, scores, t) for t in (0.1, 0.25, 0.5, 0.75, 0.9)]
    group_array = np.asarray(groups)
    negative_groups = sorted({groups[i] for i in range(len(groups)) if labels[i] == 0})
    result["leave_one_negative_group_out"] = [
        {"excluded_group": group, "metrics": metrics(labels[group_array != group], scores[group_array != group], threshold)}
        for group in negative_groups if np.any(group_array != group)]
    result["leave_one_negative_group_out_note"] = "Descriptive source sensitivity at the fixed threshold; not an independent test or a basis for retuning."
    result["warning"] = "Dataset frame prevalence is curated and is not population prevalence; unknown patient links can affect uncertainty."
    point = result["point"]
    result["illustration_per_1000_at_test_frame_prevalence"] = {
        "false_positives": 1000 * (1 - point["prevalence"]) * (1 - point["specificity"]),
        "false_negatives": 1000 * point["prevalence"] * (1 - point["recall"]),
        "note": "Descriptive re-expression of this test set, not a clinical population estimate."}
    result["illustration_per_1000_at_release_frame_prevalence"] = {
        "prevalence": release_prevalence,
        "false_positives": 1000 * (1 - release_prevalence) * (1 - point["specificity"]),
        "false_negatives": 1000 * release_prevalence * (1 - point["recall"]),
        "note": "Applies held-out sensitivity and specificity to curated release-frame prevalence; not a clinical forecast."}
    estimated_tp = release_prevalence * point["recall"]
    estimated_fp = (1 - release_prevalence) * (1 - point["specificity"])
    result["illustration_per_1000_at_release_frame_prevalence"]["precision"] = estimated_tp / (estimated_tp + estimated_fp) if estimated_tp + estimated_fp else None
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, predicate in (("false_positives", lambda p: p["label"] == 0 and p["score"] >= threshold),
                            ("false_negatives", lambda p: p["label"] == 1 and p["score"] < threshold)):
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "score", "group"]); writer.writeheader()
            writer.writerows(sorted((p for p in predictions if predicate(p)), key=lambda p: p["score"], reverse=True))
    fp = sorted((p for p in predictions if p["label"] == 0 and p["score"] >= threshold), key=lambda p: p["score"], reverse=True)[:8]
    fn = sorted((p for p in predictions if p["label"] == 1 and p["score"] < threshold), key=lambda p: p["score"])[:8]
    examples = [("FP", p) for p in fp] + [("FN", p) for p in fn]
    if examples:
        width, height = 240, 160
        sheet = Image.new("RGB", (width * 4, height * ((len(examples) + 3) // 4)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (kind, prediction) in enumerate(examples):
            with Image.open(raw / prediction["path"]) as image:
                thumbnail = image.convert("RGB"); thumbnail.thumbnail((width, height - 25))
            x, y = (index % 4) * width, (index // 4) * height
            sheet.paste(thumbnail, (x, y)); draw.text((x + 2, y + height - 21), f"{kind}  score={prediction['score']:.4f}", fill="black")
        sheet.save(output / "error_examples.png")
    by_group = defaultdict(list)
    for prediction in predictions:
        by_group[prediction["group"]].append(prediction)
    with (output / "group_errors.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "frames", "positives", "false_positives", "false_negatives"]); writer.writeheader()
        for group, items in sorted(by_group.items()):
            writer.writerow({"group": group, "frames": len(items), "positives": sum(p["label"] for p in items),
                             "false_positives": sum(p["label"] == 0 and p["score"] >= threshold for p in items),
                             "false_negatives": sum(p["label"] == 1 and p["score"] < threshold for p in items)})
    fpr, tpr, _ = roc_curve(labels, scores); precision, recall, _ = precision_recall_curve(labels, scores)
    observed, predicted = calibration_curve(labels, scores, n_bins=10, strategy="uniform")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9)); axes = axes.flatten()
    axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], "--", color="gray"); axes[0].set(xlabel="False positive rate", ylabel="Sensitivity", title="ROC")
    axes[1].step(recall[::-1], precision[::-1], where="pre"); axes[1].axhline(float(labels.mean()), linestyle="--", color="gray"); axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[2].plot(predicted, observed, marker="o"); axes[2].plot([0, 1], [0, 1], "--", color="gray"); axes[2].set(xlabel="Mean score", ylabel="Observed fraction", title="Calibration")
    grid = np.linspace(0, 1, 101)
    grid_precision, grid_recall = [], []
    for t in grid:
        positive = scores >= t
        true_positive = np.sum((labels == 1) & positive)
        grid_precision.append(true_positive / positive.sum() if positive.any() else np.nan)
        grid_recall.append(true_positive / np.sum(labels == 1))
    axes[3].plot(grid, grid_precision, label="Precision"); axes[3].plot(grid, grid_recall, label="Recall")
    axes[3].axvline(threshold, linestyle="--", color="gray", label="Fixed threshold")
    axes[3].set(xlabel="Threshold", ylabel="Metric", title="Threshold sensitivity", ylim=(0, 1)); axes[3].legend()
    fig.tight_layout(); fig.savefig(output / "evaluation.png", dpi=150); plt.close(fig)
    lock.write_text(json.dumps({**audit, "output": str(output), "threshold": threshold, "status": "complete"}, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "manifest", "checkpoint", "output", "ledger"):
        p.add_argument(name, type=Path)
    p.add_argument("--bootstrap", type=int, default=1000)
    a = p.parse_args(); evaluate(a.raw, a.manifest, a.checkpoint, a.output, a.ledger, a.bootstrap)


if __name__ == "__main__":
    main()
