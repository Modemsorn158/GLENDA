"""Freeze the best complete primary-model trial using development AP."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def select(search: Path) -> dict:
    destination = search / "selected_candidate.json"
    if destination.exists():
        raise FileExistsError("Candidate selection already exists")
    identity_file, folds_file = search / "search.json", search / "fold_results.json"
    identity = json.loads(identity_file.read_text(encoding="utf-8"))
    folds = json.loads(folds_file.read_text(encoding="utf-8"))
    configurations = identity["configurations"]
    if not configurations or any(c.get("architecture", "efficientnet_b0") != "efficientnet_b0" for c in configurations):
        raise ValueError("Candidate search must contain EfficientNet-B0 configurations")
    means = []
    for trial in range(len(configurations)):
        rows = [r for r in folds if r["trial"] == trial]
        if len(rows) != 5 or {r["fold"] for r in rows} != set(range(5)) or any(r["status"] != "complete" for r in rows):
            raise ValueError("All five folds of every candidate must complete")
        values = [r["best_validation_average_precision"] for r in rows]
        if not np.isfinite(values).all() or any(not 0 <= value <= 1 for value in values):
            raise ValueError("Invalid fold average precision")
        means.append(float(np.mean(values)))
    if len(folds) != len(configurations) * 5:
        raise ValueError("Unexpected fold results")
    best = int(np.argmax(means))
    result = {"trial_index": best, "configuration": configurations[best], "mean_average_precision": means,
              "rule": "Highest mean development average precision; exact ties use the first configuration",
              "search_sha256": hashlib.sha256(identity_file.read_bytes()).hexdigest(),
              "fold_results_sha256": hashlib.sha256(folds_file.read_bytes()).hexdigest(),
              "warning": "Candidate selection uses development folds and is not a test estimate."}
    with destination.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("search", type=Path)
    a = p.parse_args(); print(json.dumps(select(a.search), indent=2))


if __name__ == "__main__":
    main()
