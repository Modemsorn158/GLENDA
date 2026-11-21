"""Verify manifest-selected images against a separately trusted inventory digest."""
import argparse
import csv
import hashlib
import re
from pathlib import Path
from glenda.splitting import connected_groups


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(raw, manifest, inventory, expected_inventory_sha256, partition, candidates, expected_candidates_sha256):
    raw = raw.resolve()
    if sha256(inventory) != expected_inventory_sha256.lower():
        raise ValueError("Inventory hash does not match trusted digest")
    if sha256(candidates) != expected_candidates_sha256.lower():
        raise ValueError("Candidate-pair hash does not match trusted digest")
    with inventory.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label", "case_id", "video_id", "sha256"} <= set(reader.fieldnames or []):
            raise ValueError("Inventory is missing grouping fields")
        rows = list(reader)
    if any(not row["video_id"] or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in rows):
        raise ValueError("Inventory has invalid video or image identity")
    indexed = {row["path"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate inventory path")
    with manifest.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assignments = {row["path"]: row for row in manifest_rows}
    if len(assignments) != len(manifest_rows) or set(assignments) != set(indexed):
        raise ValueError("Manifest must cover the trusted inventory exactly once")
    with candidates.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"path_a", "path_b"} <= set(reader.fieldnames or []):
            raise ValueError("Candidate file is missing pair fields")
        pairs = [(row["path_a"], row["path_b"]) for row in reader]
    components = connected_groups(rows, pairs)
    seen = {}
    for row, component in zip(rows, components):
        entry = assignments[row["path"]]
        if raw not in (raw / row["path"]).resolve().parents:
            raise ValueError("Inventory path escapes the dataset root")
        if row["label"] not in {"0", "1"} or entry["label"] != row["label"]:
            raise ValueError("Manifest label does not match trusted inventory")
        if entry["group"] != component:
            raise ValueError("Manifest group differs from canonical trusted component")
        partition_name, fold = entry["partition"], entry["cv_fold"]
        if partition_name not in {"development", "test"} or (partition_name == "development" and fold not in {str(i) for i in range(5)}) or (partition_name == "test" and fold):
            raise ValueError("Invalid partition or fold")
        assignment = (partition_name, fold)
        if seen.setdefault(component, assignment) != assignment:
            raise ValueError("Connected source or duplicate crosses partitions or folds")
    selected = [row for row in manifest_rows if row["partition"] == partition]
    if not selected or len({row["path"] for row in selected}) != len(selected):
        raise ValueError("Empty partition or duplicate manifest path")
    for row in selected:
        reference = indexed.get(row["path"])
        path = (raw / row["path"]).resolve()
        if raw not in path.parents or reference is None or reference["label"] != row["label"]:
            raise ValueError("Manifest entry is unsafe or does not match inventory")
        if sha256(path) != reference["sha256"]:
            raise ValueError(f"Image hash mismatch: {row['path']}")
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "manifest", "inventory"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--inventory-sha256", required=True, help="Digest obtained from a trusted immutable audit record")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidates-sha256", required=True)
    parser.add_argument("--partition", choices=("development", "test"), default="development")
    args = parser.parse_args()
    count = verify(args.raw, args.manifest, args.inventory, args.inventory_sha256, args.partition, args.candidates, args.candidates_sha256)
    print(f"Verified {count} {args.partition} images against the trusted inventory")


if __name__ == "__main__":
    main()
