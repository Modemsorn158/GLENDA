"""Summarize matched frozen-backbone feasibility runs on the same folds."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(primary: Path, baseline: Path, output: Path) -> None:
    identities = [json.loads((d / "search.json").read_text(encoding="utf-8")) for d in (primary, baseline)]
    if identities[0]["manifest_sha256"] != identities[1]["manifest_sha256"]:
        raise ValueError("Search manifests differ")
    if any(len(x["configurations"]) != 1 for x in identities):
        raise ValueError("Expected one configuration in each search")
    configs = [dict(x["configurations"][0]) for x in identities]
    if configs[0].pop("architecture", None) != "efficientnet_b0" or configs[1].pop("architecture", None) != "resnet50" or configs[0] != configs[1]:
        raise ValueError("Searches must differ only by the expected architecture")
    results = []
    for directory in (primary, baseline):
        rows = json.loads((directory / "fold_results.json").read_text(encoding="utf-8"))
        if len(rows) != 5 or {r["fold"] for r in rows} != set(range(5)) or any(r["status"] != "complete" for r in rows):
            raise ValueError("All five folds must complete")
        results.append({r["fold"]: r["best_validation_average_precision"] for r in rows})
    keys = ("fold", "seed", "fine_tuning", "lr", "batch_size", "epochs", "patience", "weight_decay", "weighted",
            "manifest_sha256", "torch_version", "torchvision_version", "transforms")
    for fold in range(5):
        data = [json.loads((directory / f"trial_000/fold_{fold}/config.json").read_text(encoding="utf-8")) for directory in (primary, baseline)]
        if any(key not in row for row in data for key in keys) or any(data[0][key] != data[1][key] for key in keys):
            raise ValueError(f"Fold {fold} configurations differ beyond architecture")
        if data[0]["architecture"] != "efficientnet_b0" or data[1]["architecture"] != "resnet50":
            raise ValueError("Unexpected architecture")
    lines = ["# Architecture feasibility comparison", "",
             "Both frozen ImageNet models used the same folds, preprocessing and one-epoch optimization settings. "
             "Different architectures and weights can require different tuning; this is not a superiority test.", "",
             "| Fold | EfficientNet-B0 AP | ResNet50 AP | ResNet minus EfficientNet |", "|---:|---:|---:|---:|"]
    for fold in range(5):
        x, y = results[0][fold], results[1][fold]
        lines.append(f"| {fold} | {x:.4f} | {y:.4f} | {y - x:+.4f} |")
    differences = [results[1][i] - results[0][i] for i in range(5)]
    lines.extend(["", f"Mean paired difference: **{np.mean(differences):+.4f}**, descriptive only.", "",
                  "One epoch is a feasibility budget, not evidence of convergence; folds are dependent, "
                  "and unknown patient overlap and source effects limit interpretation.\n"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps({"searches": identities, "folds": results,
                                                       "mean_paired_difference": float(np.mean(differences))}, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("primary", type=Path); p.add_argument("baseline", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); compare(a.primary, a.baseline, a.output)


if __name__ == "__main__":
    main()
