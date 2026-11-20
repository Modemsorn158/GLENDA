"""Generate a split-composition table from the saved v2 protocol."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(protocol: Path, manifest: Path, output: Path) -> None:
    data = json.loads(protocol.read_text(encoding="utf-8"))
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != data["manifest_sha256"]:
        raise ValueError("Protocol and manifest hashes differ")
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for part in ("development", "test"):
        subset = [row for row in rows if row["partition"] == part]
        groups = data["group_counts"][part]
        if dict(Counter(row["label"] for row in subset)) != data["counts"][part] or any(
            len({row["group"] for row in subset if row["label"] == label}) != groups[key]
            for label, key in (("0", "with_negative"), ("1", "with_positive"))
        ):
            raise ValueError("Partition counts do not match the manifest")
    for fold in range(5):
        subset = [row for row in rows if row["partition"] == "development" and row["cv_fold"] == str(fold)]
        if dict(Counter(row["label"] for row in subset)) != data["validation_counts"][str(fold)] or any(
            len({row["group"] for row in subset if row["label"] == label}) != data["validation_groups_by_label"][str(fold)][label]
            for label in ("0", "1")
        ):
            raise ValueError("Fold counts do not match the manifest")
    negative_groups = Counter(row["group"] for row in rows if row["partition"] == "test" and row["label"] == "0")
    if not negative_groups:
        raise ValueError("Holdout has no negative-bearing groups")
    largest_share = max(negative_groups.values()) / sum(negative_groups.values())
    lines = ["# Split composition", "", "Group counts reflect known linked source components, not confirmed independent patients.", "",
             "| Partition | Positive frames | Negative frames | Positive-bearing groups | Negative-bearing groups |",
             "|---|---:|---:|---:|---:|"]
    for part in ("development", "test"):
        counts, groups = data["counts"][part], data["group_counts"][part]
        lines.append(f"| {part} | {counts.get('1', 0)} | {counts.get('0', 0)} | {groups['with_positive']} | {groups['with_negative']} |")
    lines.extend(["", "| Validation fold | Positive frames | Negative frames | Positive-bearing groups | Negative-bearing groups |",
                  "|---:|---:|---:|---:|---:|"])
    for fold in range(5):
        counts, groups = data["validation_counts"][str(fold)], data["validation_groups_by_label"][str(fold)]
        lines.append(f"| {fold} | {counts.get('1', 0)} | {counts.get('0', 0)} | {groups['1']} | {groups['0']} |")
    lines.extend(["", "Groups can contain frames from both labels, so the two group columns need not sum to the total. "
                  f"The holdout is unchanged from v1; it contains {len(negative_groups)} negative-bearing groups, and the largest contributes {largest_share:.1%} of its negative frames.\n"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("protocol", type=Path); p.add_argument("manifest", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); summarize(a.protocol, a.manifest, a.output)


if __name__ == "__main__":
    main()
