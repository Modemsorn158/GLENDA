"""Compare audited frame hashes with the supplied GLENDA release archives."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def digest(stream) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def compare(inventory: Path, positive_zip: Path, negative_zip: Path, raw: Path, output: Path) -> dict:
    if output.resolve() == raw.resolve() or raw.resolve() in output.resolve().parents:
        raise ValueError("Output must be outside raw data")
    with inventory.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {row["path"]: row["sha256"] for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate paths in inventory")
    seen, mismatches, duplicate_entries = set(), [], []
    annotation_match = None
    for path, prefix, destination in ((positive_zip, "Glenda_v1.5_classes/frames/", "pathology/frames/"),
                                      (negative_zip, "no_pathology/frames/", "no_pathology/frames/")):
        with ZipFile(path) as archive:
            names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
            if len(names) != len(set(names)):
                duplicate_entries.append(path.name)
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                if entry.filename == "Glenda_v1.5_classes/coco.json":
                    with archive.open(entry) as stream, (raw / "pathology/coco.json").open("rb") as reference:
                        annotation_match = digest(stream) == digest(reference)
                if not entry.filename.startswith(prefix) or not entry.filename.lower().endswith(".jpg"):
                    continue
                relative = destination + entry.filename[len(prefix):]
                seen.add(relative)
                with archive.open(entry) as stream:
                    value = digest(stream)
                if expected.get(relative) != value:
                    mismatches.append(relative)
    archive_hashes = {}
    for path in (positive_zip, negative_zip):
        with path.open("rb") as stream:
            archive_hashes[path.name] = digest(stream)
    result = {"inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(), "archive_sha256": archive_hashes,
              "archive_frames": len(seen), "inventory_frames": len(expected),
              "missing_archive_frames": sorted(set(expected) - seen), "unexpected_or_mismatched_frames": mismatches,
              "duplicate_archive_entry_names": duplicate_entries, "coco_byte_match": annotation_match}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["missing_archive_frames"] or mismatches or duplicate_entries or annotation_match is not True:
        raise ValueError("Archive comparison failed; see output")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "positive_zip", "negative_zip", "raw", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args(); print(json.dumps(compare(a.inventory, a.positive_zip, a.negative_zip, a.raw, a.output), indent=2))


if __name__ == "__main__":
    main()
