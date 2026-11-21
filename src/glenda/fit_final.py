"""Fit a selected model on development only for a fixed CV-derived epoch count."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import sklearn
from torch import nn
from torch.utils.data import DataLoader

from glenda.dataset import DevelopmentDataset
from glenda.models import build_model, pretrained_url, set_training_mode
from glenda.select_threshold import choose_threshold


def fit(raw: Path, manifest: Path, trial: Path, threshold_file: Path, output: Path, inventory: Path = None, inventory_sha256: str = None) -> None:
    raw, output = raw.resolve(), output.resolve()
    if output == raw or raw in output.parents:
        raise ValueError("Output must be outside raw data")
    if output.exists():
        raise FileExistsError("Refusing to overwrite final fit")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    selection = json.loads(threshold_file.read_text(encoding="utf-8"))
    if selection["manifest_sha256"] != manifest_hash or trial.name != f"trial_{selection['trial_index']:03d}":
        raise ValueError("Threshold selection does not match manifest and trial")
    if selection["candidate_selection_sha256"] != hashlib.sha256((trial.parent / "selected_candidate.json").read_bytes()).hexdigest():
        raise ValueError("Candidate selection changed after threshold selection")
    search = json.loads((trial.parent / "search.json").read_text(encoding="utf-8"))
    if search["manifest_sha256"] != manifest_hash or search["configurations"][selection["trial_index"]] != selection["trial_configuration"]:
        raise ValueError("Selected trial does not match the original search")
    source_names = ("dataset.py", "models.py", "train_fold.py", "verify_data.py")
    current_sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_names}
    if search.get("training_source_sha256") != current_sources:
        raise ValueError("Training source differs from the cross-validation search or was not recorded; start a new study")
    if (inventory is None) != (inventory_sha256 is None):
        raise ValueError("Inventory and its trusted hash must be supplied together")
    if inventory is not None:
        if search.get("inventory_sha256") != inventory_sha256.lower():
            raise ValueError("Inventory differs from cross-validation")
        from glenda.verify_data import verify
        verify(raw, manifest, inventory, inventory_sha256, "development")
    if len(selection.get("fold_epoch_history_sha256", [])) != 5 or len(selection.get("fold_best_epochs", [])) != 5:
        raise ValueError("Threshold selection does not freeze all epoch histories; start a new study")
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
        epoch_file = folder / "epochs.csv"
        if hashlib.sha256(epoch_file.read_bytes()).hexdigest() != selection["fold_epoch_history_sha256"][fold]:
            raise ValueError("Epoch history changed after threshold selection")
        with epoch_file.open(encoding="utf-8", newline="") as f:
            epochs = list(csv.DictReader(f))
        best_epochs.append(int(max(epochs, key=lambda row: float(row["validation_average_precision"]))["epoch"]))
        if best_epochs[-1] != selection["fold_best_epochs"][fold]:
            raise ValueError("Best epoch differs from frozen selection")
    expected_threshold, _ = choose_threshold(labels, scores, selection.get("minimum_sensitivity"))
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
                "inventory_sha256": inventory_sha256.lower() if inventory_sha256 else None,
                "threshold_file_sha256": hashlib.sha256(threshold_file.read_bytes()).hexdigest(),
                "pretrained_weights_url": url, "pretrained_weights_sha256": weight_hash,
                "device": str(device), "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
                "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
                "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "python_version": platform.python_version(),
                "cuda_version": torch.version.cuda, "cv_training_source_sha256": search.get("training_source_sha256")}
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
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--inventory-sha256", required=True)
    a = p.parse_args(); fit(a.raw, a.manifest, a.trial, a.threshold, a.output, a.inventory, a.inventory_sha256)


if __name__ == "__main__":
    main()
