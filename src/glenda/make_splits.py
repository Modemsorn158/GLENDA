"""Create a frozen grouped holdout and development CV manifests.

This is an internal video/case-linked evaluation, not patient-independent validation.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from glenda.splitting import assert_disjoint, connected_groups


def make(inventory: Path, candidates: Path, output: Path, seed: int = 42) -> None:
    if output.exists():
        raise FileExistsError("Refusing to overwrite a split directory")
    with inventory.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with candidates.open(encoding="utf-8", newline="") as f:
        pairs = [(x["path_a"], x["path_b"]) for x in csv.DictReader(f)]
    groups = np.asarray(connected_groups(rows, pairs))
    labels = np.asarray([int(row["label"]) for row in rows])
    if set(labels) != {0, 1}:
        raise ValueError("Inventory must contain both binary classes")
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev, test = next(outer.split(np.zeros(len(rows)), labels, groups))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    folds = np.full(len(rows), -1, dtype=int)
    for fold, (_, val) in enumerate(inner.split(np.zeros(len(dev)), labels[dev], groups[dev])):
        folds[dev[val]] = fold
    test_indices = set(test)
    assignments = ["test" if i in test_indices else "development" for i in range(len(rows))]
    assert_disjoint(rows, assignments, pairs)
    dev_paths = {rows[i]["path"] for i in dev}
    dev_pairs = [(a, b) for a, b in pairs if a in dev_paths and b in dev_paths]
    for f in range(5):
        indices = dev[folds[dev] == f]
        train = dev[folds[dev] != f]
        if set(labels[indices]) != {0, 1} or set(labels[train]) != {0, 1}:
            raise ValueError(f"Fold {f} lacks a class")
        assert_disjoint([rows[i] for i in dev], ["validation" if folds[i] == f else "train" for i in dev],
                        dev_pairs)
    if set(labels[test]) != {0, 1}:
        raise ValueError("Test split lacks a class")
    output.mkdir(parents=True)
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "group", "partition", "cv_fold"])
        w.writeheader()
        for i, row in enumerate(rows):
            w.writerow({"path": row["path"], "label": row["label"], "group": groups[i],
                        "partition": assignments[i], "cv_fold": folds[i] if folds[i] >= 0 else ""})
    summary = {"seed": seed, "method": "first shuffled stratified-group fivefold outer split; fivefold development CV",
               "numpy_version": np.__version__, "scikit_learn_version": sklearn.__version__,
               "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
               "candidate_pairs_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
               "counts": {name: dict(Counter(str(labels[i]) for i in range(len(rows)) if assignments[i] == name)) for name in ("development", "test")},
               "validation_counts": {str(f): dict(Counter(str(labels[i]) for i in dev if folds[i] == f)) for f in range(5)},
               "warning": "Unmapped negative patient identities and near-duplicate false negatives prevent a guarantee of patient-level independence."}
    (output / "protocol.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventory", type=Path); p.add_argument("candidates", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    make(args.inventory, args.candidates, args.output, args.seed)


if __name__ == "__main__":
    main()
