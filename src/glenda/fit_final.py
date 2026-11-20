"""Fit a selected model on development only for a fixed CV-derived epoch count."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader

from glenda.dataset import DevelopmentDataset
from glenda.models import build_model, pretrained_url, set_training_mode


def fit(raw: Path, manifest: Path, trial: Path, threshold_file: Path, output: Path) -> None:
    raw, output = raw.resolve(), output.resolve()
    if output == raw or raw in output.parents:
        raise ValueError("Output must be outside raw data")
    if output.exists():
        raise FileExistsError("Refusing to overwrite final fit")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    selection = json.loads(threshold_file.read_text(encoding="utf-8"))
    if selection["manifest_sha256"] != manifest_hash or trial.name != f"trial_{selection['trial_index']:03d}":
        raise ValueError("Threshold selection does not match manifest and trial")
    search = json.loads((trial.parent / "search.json").read_text(encoding="utf-8"))
    if search["manifest_sha256"] != manifest_hash or search["configurations"][selection["trial_index"]] != selection["trial_configuration"]:
        raise ValueError("Selected trial does not match the original search")
    fold_configs, best_epochs, labels, scores = [], [], [], []
    for fold in range(5):
        if json.loads((trial / f"fold_{fold}_status.json").read_text(encoding="utf-8"))["status"] != "complete":
            raise ValueError("All five folds must complete")
        folder = trial / f"fold_{fold}"
        if hashlib.sha256((folder / "validation_predictions.csv").read_bytes()).hexdigest() != selection["fold_prediction_sha256"][fold]:
            raise ValueError("Validation predictions changed after threshold selection")
        with (folder / "validation_predictions.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                labels.append(int(row["label"])); scores.append(float(row["score"]))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if config["manifest_sha256"] != manifest_hash or config["fold"] != fold:
            raise ValueError("Fold configuration does not match manifest")
        fold_configs.append(config)
        with (folder / "epochs.csv").open(encoding="utf-8", newline="") as f:
            epochs = list(csv.DictReader(f))
        best_epochs.append(int(max(epochs, key=lambda row: float(row["validation_average_precision"]))["epoch"]))
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator, out=np.zeros_like(denominator), where=denominator != 0)
    expected_threshold = thresholds[np.flatnonzero(np.isclose(f1, np.max(f1), rtol=0, atol=1e-12))[-1]]
    if not np.isclose(selection["threshold"], expected_threshold, rtol=0, atol=1e-12):
        raise ValueError("Threshold does not match the prespecified selection rule")
    keys = ["architecture", "fine_tuning", "lr", "weight_decay", "batch_size", "seed", "weighted", "random_init", "cpu"]
    base = {key: fold_configs[0][key] for key in keys}
    if any(any(config[key] != base[key] for key in keys) for config in fold_configs):
        raise ValueError("Fold training configurations differ")
    if any(fold_configs[0].get(key) != value for key, value in selection["trial_configuration"].items()):
        raise ValueError("Fold configuration differs from selected trial")
    if base["random_init"]:
        raise ValueError("Random initialization is reserved for smoke tests")
    # Median of the five best epochs, chosen without the held-out split.
    epoch_count = sorted(best_epochs)[2]
    random.seed(base["seed"]); np.random.seed(base["seed"]); torch.manual_seed(base["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and not fold_configs[0]["cpu"] else "cpu")
    data = DevelopmentDataset(raw, manifest, None, True)
    output.mkdir(parents=True)
    cache = trial.parent / ".torch-cache"
    os.environ["TORCH_HOME"] = str(cache.resolve())
    model = build_model(base["architecture"], base["fine_tuning"], pretrained=True).to(device)
    url = pretrained_url(base["architecture"])
    weights = cache / "hub/checkpoints" / url.rsplit("/", 1)[-1]
    weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    if any(config["pretrained_weights_sha256"] != weight_hash for config in fold_configs):
        raise ValueError("Pretrained weights differ from those used in CV")
    metadata = {**base, "epochs": epoch_count, "cv_best_epochs": best_epochs,
                "manifest_sha256": manifest_hash, "threshold": selection["threshold"],
                "threshold_file_sha256": hashlib.sha256(threshold_file.read_bytes()).hexdigest(),
                "pretrained_weights_url": url, "pretrained_weights_sha256": weight_hash,
                "device": str(device), "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}
    (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    generator = torch.Generator().manual_seed(base["seed"])
    loader = DataLoader(data, batch_size=base["batch_size"], shuffle=True, generator=generator, num_workers=0)
    positives = sum(int(row["label"]) for row in data.rows)
    weight = torch.tensor([(len(data) - positives) / positives], device=device) if base["weighted"] else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=base["lr"], weight_decay=base["weight_decay"])
    with (output / "epochs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "training_loss"]); writer.writeheader()
        for epoch in range(1, epoch_count + 1):
            set_training_mode(model); total = 0.0
            for images, labels, _ in loader:
                images, labels = images.to(device), labels.to(device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images).flatten(), labels)
                loss.backward(); optimizer.step(); total += loss.item() * len(labels)
            writer.writerow({"epoch": epoch, "training_loss": total / len(data)}); f.flush()
            print(f"epoch={epoch}/{epoch_count}", flush=True)
    torch.save({"state_dict": model.state_dict(), "config": metadata}, output / "final.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "manifest", "trial", "threshold", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args(); fit(a.raw, a.manifest, a.trial, a.threshold, a.output)


if __name__ == "__main__":
    main()
