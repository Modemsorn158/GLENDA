"""Plot descriptive dataset composition from an audited inventory."""
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


def plot(inventory: Path, output: Path) -> None:
    os.environ["MPLCONFIGDIR"] = str(output.parent / ".mpl-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with inventory.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    labels = Counter(row["label"] for row in rows)
    videos = {label: Counter(row["video_id"] for row in rows if row["label"] == label) for label in ("0", "1")}
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for label in ("0", "1"):
        counts = sorted(videos[label].values(), reverse=True)
        summary[label] = {"frames": labels[label], "videos": len(counts), "largest_video_frames": counts[0] if counts else 0,
                          "largest_five_video_frames": sum(counts[:5])}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bars = axes[0].bar(["No visible pathology", "Annotated pathology"], [labels["0"], labels["1"]], color=["#4c78a8", "#e45756"])
    axes[0].set_yscale("log"); axes[0].set_ylabel("Frames (log scale)"); axes[0].set_title("Release composition")
    axes[0].bar_label(bars)
    for label, title, color in (("0", "Negative sequences", "#4c78a8"), ("1", "Positive annotations", "#e45756")):
        axes[1].plot(sorted(videos[label].values(), reverse=True), label=title, color=color)
    axes[1].set_yscale("log"); axes[1].set(xlabel="Video rank within class", ylabel="Frames per video (log scale)", title="Source-video imbalance")
    axes[1].legend(); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventory", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); plot(a.inventory, a.output)


if __name__ == "__main__":
    main()
