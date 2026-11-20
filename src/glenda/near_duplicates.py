"""Screen for visually similar images across known source components.

Difference hashes are a screening tool, not proof of common source or identity.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

from glenda.splitting import connected_groups


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        image = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    result = 0
    for row in range(8):
        for col in range(8):
            result = (result << 1) | (pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return result


def screen(raw: Path, inventory: Path, output: Path, max_distance: int = 3) -> int:
    if not 0 <= max_distance <= 3:
        raise ValueError("Four 16-bit bands guarantee candidates only for distances <= 3")
    with inventory.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    groups = connected_groups(rows)
    buckets = defaultdict(list)
    hashes = []
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path_a", "path_b", "label_a", "label_b", "hamming_distance"])
        for i, row in enumerate(rows):
            value = dhash(raw / row["path"])
            keys = [(band, (value >> (16 * band)) & 65535) for band in range(4)]
            candidates = set(j for key in keys for j in buckets[key])
            for j in sorted(candidates):
                if groups[j] == groups[i]:
                    continue
                distance = (value ^ hashes[j]).bit_count()
                if distance <= max_distance:
                    writer.writerow([rows[j]["path"], row["path"], rows[j]["label"], row["label"], distance])
                    count += 1
            hashes.append(value)
            for key in keys:
                buckets[key].append(i)
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("inventory", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.raw.resolve() in args.output.resolve().parents:
        raise ValueError("Output must be outside the raw dataset")
    print(f"Cross-component candidates: {screen(args.raw, args.inventory, args.output)}")


if __name__ == "__main__":
    main()
