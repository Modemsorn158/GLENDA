"""Build a traceable, read-only inventory of the GLENDA v1.5 release."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image

POSITIVE = re.compile(r"c_(\d+)_v_\(video_(\d+)\.mp4\)_f_(\d+)\.jpg")
NEGATIVE = re.compile(r"v_(\d+)_s_(\d+)-(\d+)")


def audit(root: Path, out: Path) -> dict:
    root = root.resolve()
    out = out.resolve()
    if out == root or root in out.parents:
        raise ValueError("Output must be outside the raw dataset")
    coco = json.loads((root / "pathology/coco.json").read_text(encoding="utf-8"))
    images = {i["file_name"]: i for i in coco["images"]}
    if len(images) != len(coco["images"]):
        raise ValueError("Duplicate COCO filenames")
    ann = collections.defaultdict(list)
    for a in coco["annotations"]:
        ann[a["image_id"]].append(a["category_id"])
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    by_video = collections.defaultdict(set)
    for name in images:
        m = POSITIVE.fullmatch(name)
        if not m:
            raise ValueError(f"Unrecognized positive filename: {name}")
        by_video[m[2]].add(m[1])
    if any(len(cases) != 1 for cases in by_video.values()):
        raise ValueError("A positive video maps to multiple case IDs")
    positives = sorted((root / "pathology/frames").glob("*.jpg"))
    if {p.name for p in positives} != set(images):
        raise ValueError("Positive frame files and COCO image list disagree")
    negatives = sorted((root / "no_pathology/frames").rglob("*.jpg"))
    rows, errors = [], []
    dimensions, digests, video_counts = collections.Counter(), collections.defaultdict(list), collections.Counter()
    for path in positives + negatives:
        positive = path.parent.name == "frames" and path.parent.parent.name == "pathology"
        if positive:
            m = POSITIVE.fullmatch(path.name)
            case, video, frame = m.groups()
            image = images[path.name]
            classes = ann[image["id"]]
            if not classes or any(c not in categories for c in classes):
                raise ValueError(f"Missing or invalid annotation: {path}")
            provenance = "coco_annotation"
        else:
            m = NEGATIVE.fullmatch(path.parent.name)
            if not m or not re.fullmatch(r"f_\d+\.jpg", path.name):
                raise ValueError(f"Unrecognized negative path: {path}")
            video, _, _ = m.groups()
            frame = path.stem[2:]
            case = next(iter(by_video[video])) if video in by_video else ""
            classes = []
            provenance = "release_no_pathology_sequence"
        rel = path.relative_to(root).as_posix()
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as im:
                im.load()
                size, fmt = im.size, im.format
            if positive and size != (image["width"], image["height"]):
                raise ValueError("Dimensions disagree with COCO")
        except Exception as exc:
            errors.append({"path": rel, "error": str(exc)})
            continue
        dimensions[f"{size[0]}x{size[1]} {fmt}"] += 1
        video_counts[video] += 1
        digests[digest].append((rel, int(positive)))
        rows.append({"path": rel, "label": int(positive), "label_source": provenance,
                     "case_id": case, "video_id": video, "frame_id": frame,
                     "annotation_categories": "|".join(sorted({categories[c] for c in classes})),
                     "sha256": digest, "width": size[0], "height": size[1], "format": fmt})
    duplicates = [v for v in digests.values() if len(v) > 1]
    conflicts = [v for v in duplicates if len({label for _, label in v}) > 1]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "inventory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["path"])
        writer.writeheader(); writer.writerows(rows)
    summary = {"frames": len(rows), "positive": sum(r["label"] for r in rows),
               "negative": sum(not r["label"] for r in rows), "coco_annotations": len(coco["annotations"]),
               "positive_cases": len({r['case_id'] for r in rows if r['label']}),
               "positive_videos": len({r['video_id'] for r in rows if r['label']}),
               "negative_videos": len({r['video_id'] for r in rows if not r['label']}),
               "negative_videos_with_known_positive_case": sorted(set(by_video) & {r['video_id'] for r in rows if not r['label']}),
               "negative_videos_without_case_mapping": sorted({r['video_id'] for r in rows if not r['label'] and not r['case_id']}),
               "dimensions": dict(dimensions), "exact_duplicate_sets": len(duplicates),
               "cross_label_duplicate_sets": len(conflicts), "errors": errors}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "exact_duplicates.json").write_text(json.dumps(duplicates, indent=2), encoding="utf-8")
    if errors or conflicts:
        raise ValueError(f"Audit failed: {len(errors)} unreadable files, {len(conflicts)} cross-label duplicate sets; see {out}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.raw, args.output), indent=2))


if __name__ == "__main__":
    main()
