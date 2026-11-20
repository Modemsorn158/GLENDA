"""Read-only image datasets restricted to a selected development fold."""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocessing(training: bool):
    steps = [transforms.Resize((224, 224))]
    if training:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(steps)


class DevelopmentDataset(Dataset):
    """Expose a development fold or, for final fitting, all development rows."""

    def __init__(self, root: Path, manifest: Path, fold: int | None, training: bool):
        if fold is None and not training:
            raise ValueError("All-development data is only available for fitting")
        if fold is not None and fold not in range(5):
            raise ValueError("Fold must be between 0 and 4")
        self.root = root.resolve()
        self.transform = preprocessing(training)
        with manifest.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        seen_paths, group_assignments = set(), {}
        for row in rows:
            if row["path"] in seen_paths:
                raise ValueError("Duplicate path in manifest")
            seen_paths.add(row["path"])
            if row["partition"] not in {"development", "test"} or not row["group"]:
                raise ValueError("Invalid partition or group")
            fold_value = row["cv_fold"]
            if row["partition"] == "development" and fold_value not in {str(i) for i in range(5)}:
                raise ValueError("Invalid development fold")
            if row["partition"] == "test" and fold_value:
                raise ValueError("Test rows must not have a CV fold")
            assignment = (row["partition"], fold_value)
            previous = group_assignments.setdefault(row["group"], assignment)
            if previous != assignment:
                raise ValueError("Group spans partitions or validation folds")
        self.rows = [r for r in rows if r["partition"] == "development" and
                     (fold is None or ((int(r["cv_fold"]) != fold) if training else (int(r["cv_fold"]) == fold)))]
        if not self.rows or {r["label"] for r in self.rows} != {"0", "1"}:
            raise ValueError("The selected fold must contain both labels")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.root / row["path"]).resolve()
        if self.root not in path.parents:
            raise ValueError("Image path escapes dataset root")
        with Image.open(path) as image:
            value = self.transform(image.convert("RGB"))
        return value, int(row["label"]), row["path"]
