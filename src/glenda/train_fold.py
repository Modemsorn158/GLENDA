"""Train one development fold; this command cannot evaluate the test partition."""
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
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader

from glenda.dataset import DevelopmentDataset
from glenda.models import build_model, set_training_mode


def train(args):
    raw_root, output = args.raw.resolve(), args.output.resolve()
    if output == raw_root or raw_root in output.parents:
        raise ValueError("Experiment output must be outside the raw dataset")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_data = DevelopmentDataset(args.raw, args.manifest, args.fold, True)
    val_data = DevelopmentDataset(args.raw, args.manifest, args.fold, False)
    args.output.mkdir(parents=True)
    os.environ["TORCH_HOME"] = str(args.output.parent.resolve() / ".torch-cache")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(device=str(device), torch_version=torch.__version__, torchvision_version=torchvision.__version__,
                  sklearn_version=sklearn.__version__, python_version=platform.python_version(),
                  cuda_version=torch.version.cuda, manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  transforms="resize 224x224; training horizontal flip; ImageNet normalization")
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = build_model(args.architecture, args.fine_tuning, not args.random_init).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    positives = sum(int(r["label"]) for r in train_data.rows)
    negatives = len(train_data) - positives
    weight = torch.tensor([negatives / positives], device=device) if args.weighted else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    best, stale = -1.0, 0
    with (args.output / "epochs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "training_loss", "validation_average_precision"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            set_training_mode(model)
            total_loss = 0.0
            for x, y, _ in train_loader:
                x, y = x.to(device), y.to(device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x).flatten()
                loss = criterion(logits, y)
                loss.backward(); optimizer.step()
                total_loss += loss.item() * len(y)
            model.eval(); labels, scores = [], []
            with torch.no_grad():
                for x, y, _ in val_loader:
                    labels.extend(y.tolist()); scores.extend(torch.sigmoid(model(x.to(device)).flatten()).cpu().tolist())
            ap = float(average_precision_score(labels, scores))
            writer.writerow(dict(epoch=epoch, training_loss=total_loss / len(train_data), validation_average_precision=ap)); f.flush()
            print(f"epoch={epoch} validation_ap={ap:.6f}", flush=True)
            if ap > best:
                best, stale = ap, 0
                torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch, "validation_ap": ap}, args.output / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path); p.add_argument("manifest", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--fold", type=int, required=True); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--architecture", choices=["efficientnet_b0", "resnet50"], default="efficientnet_b0")
    p.add_argument("--fine-tuning", choices=["frozen", "partial", "full"], default="frozen")
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight-decay", type=float, default=0.0001); p.add_argument("--weighted", action="store_true")
    p.add_argument("--random-init", action="store_true", help="Untrained backbone for smoke tests only")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.lr <= 0 or args.weight_decay < 0:
        p.error("Epochs, patience, batch size and learning rate must be positive; weight decay must be nonnegative")
    train(args)


if __name__ == "__main__":
    main()
