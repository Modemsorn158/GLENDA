"""Verify manifest-selected images against a separately trusted inventory digest."""
import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(raw, manifest, inventory, expected_inventory_sha256, partition):
    raw = raw.resolve()
    if sha256(inventory) != expected_inventory_sha256.lower():
        raise ValueError("Inventory hash does not match trusted digest")
    with inventory.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["path"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate inventory path")
    with manifest.open(encoding="utf-8", newline="") as handle:
        selected = [row for row in csv.DictReader(handle) if row["partition"] == partition]
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
    parser.add_argument("--partition", choices=("development", "test"), default="development")
    args = parser.parse_args()
    count = verify(args.raw, args.manifest, args.inventory, args.inventory_sha256, args.partition)
    print(f"Verified {count} {args.partition} images against the trusted inventory")


if __name__ == "__main__":
    main()
