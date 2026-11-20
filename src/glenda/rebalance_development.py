"""Reassign development folds by known group counts; preserve the held-out partition."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def rebalance(manifest: Path, output: Path, seed: int = 43) -> None:
    if output.exists():
        raise FileExistsError("Refusing to overwrite a split directory")
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    path_set = set()
    for row in rows:
        if row["path"] in path_set or row["partition"] not in {"development", "test"} or row["label"] not in {"0", "1"} or not row["group"]:
            raise ValueError("Invalid or repeated manifest row")
        path_set.add(row["path"]); groups[row["group"]].append(row)
    if any(len({row["partition"] for row in members}) != 1 for members in groups.values()):
        raise ValueError("Group crosses the held-out boundary")
    dev = {group: members for group, members in groups.items() if members[0]["partition"] == "development"}
    counts = {group: Counter(row["label"] for row in members) for group, members in dev.items()}
    negative_groups = [group for group, c in counts.items() if c["0"]]
    positive_only = [group for group, c in counts.items() if not c["0"]]
    if len(negative_groups) < 10:
        raise ValueError("At least ten negative-containing groups are required for two per fold")
    rng = random.Random(seed)
    tie = {group: rng.random() for group in dev}
    negative_groups.sort(key=lambda group: (-counts[group]["0"], tie[group]))
    positive_only.sort(key=lambda group: (-counts[group]["1"], tie[group]))
    neg_groups = [0] * 5; neg_frames = [0] * 5; pos_frames = [0] * 5
    assignment = {}
    for group in negative_groups:
        choices = list(range(5)); rng.shuffle(choices)
        fold = min(choices, key=lambda f: (neg_groups[f], neg_frames[f], pos_frames[f]))
        assignment[group] = fold; neg_groups[fold] += 1
        neg_frames[fold] += counts[group]["0"]; pos_frames[fold] += counts[group]["1"]
    for group in positive_only:
        choices = list(range(5)); rng.shuffle(choices)
        fold = min(choices, key=lambda f: (pos_frames[f], neg_frames[f]))
        assignment[group] = fold; pos_frames[fold] += counts[group]["1"]
    if any(n < 2 for n in neg_groups) or any(n == 0 for n in pos_frames):
        raise ValueError("Development fold lacks sufficient known groups or positives")
    holdout_before = [dict(row) for row in rows if row["partition"] == "test"]
    for row in rows:
        if row["partition"] == "development":
            row["cv_fold"] = str(assignment[row["group"]])
    if [row for row in rows if row["partition"] == "test"] != holdout_before:
        raise AssertionError("Held-out rows changed")
    output.mkdir(parents=True)
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "group", "partition", "cv_fold"])
        writer.writeheader(); writer.writerows(rows)
    summary = {"seed": seed, "method": "greedy balance of negative-group counts, negative frames, then positive frames",
               "parent_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "manifest_sha256": hashlib.sha256((output / "manifest.csv").read_bytes()).hexdigest(),
               "holdout_sha256": hashlib.sha256(json.dumps(sorted(row["path"] for row in holdout_before), separators=(",", ":")).encode("utf-8")).hexdigest(),
               "holdout_unchanged": True,
               "counts": {name: dict(Counter(row["label"] for row in rows if row["partition"] == name)) for name in ("development", "test")},
               "group_counts": {name: {"all": len({r["group"] for r in rows if r["partition"] == name}),
                                        "with_positive": len({r["group"] for r in rows if r["partition"] == name and r["label"] == "1"}),
                                        "with_negative": len({r["group"] for r in rows if r["partition"] == name and r["label"] == "0"})}
                                for name in ("development", "test")},
               "validation_counts": {str(f): dict(Counter(r["label"] for r in rows if r["partition"] == "development" and r["cv_fold"] == str(f))) for f in range(5)},
               "validation_groups_by_label": {str(f): {str(label): len({r["group"] for r in rows if r["partition"] == "development" and r["cv_fold"] == str(f) and r["label"] == str(label)}) for label in (0, 1)} for f in range(5)},
               "warning": "Known groups are not guaranteed to correspond to independent patients."}
    (output / "protocol.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("manifest", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--seed", type=int, default=43)
    a = p.parse_args(); rebalance(a.manifest, a.output, a.seed)


if __name__ == "__main__":
    main()
