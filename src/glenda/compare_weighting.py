"""Summarize a matched one-epoch class-weighting feasibility ablation."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(unweighted: Path, weighted: Path, output: Path) -> None:
    left = json.loads((unweighted / "search.json").read_text(encoding="utf-8"))
    right = json.loads((weighted / "search.json").read_text(encoding="utf-8"))
    if left["manifest_sha256"] != right["manifest_sha256"] or len(left["configurations"]) != 1 or len(right["configurations"]) != 1:
        raise ValueError("Runs must share a manifest and have exactly one configuration")
    a, b = dict(left["configurations"][0]), dict(right["configurations"][0])
    if a.pop("weighted", False) is not False or b.pop("weighted", False) is not True or a != b:
        raise ValueError("Configurations must differ only in class weighting")
    results = []
    for directory in (unweighted, weighted):
        rows = json.loads((directory / "fold_results.json").read_text(encoding="utf-8"))
        if len(rows) != 5 or {r["fold"] for r in rows} != set(range(5)) or any(r["status"] != "complete" for r in rows):
            raise ValueError("All five folds must complete")
        results.append({r["fold"]: r["best_validation_average_precision"] for r in rows})
    matching_keys = ("fold", "seed", "architecture", "fine_tuning", "lr", "batch_size", "epochs", "patience", "weight_decay",
                     "manifest_sha256", "pretrained_weights_sha256", "torch_version", "torchvision_version", "transforms")
    for fold in range(5):
        configs = [json.loads((directory / f"trial_000/fold_{fold}/config.json").read_text(encoding="utf-8"))
                   for directory in (unweighted, weighted)]
        if any(key not in config for config in configs for key in matching_keys) or any(configs[0][key] != configs[1][key] for key in matching_keys) or configs[0]["weighted"] or not configs[1]["weighted"]:
            raise ValueError(f"Fold {fold} runtime configurations do not match the planned ablation")
    lines = ["# Class-weighting feasibility comparison", "",
             "One epoch per fold with frozen ImageNet EfficientNet-B0; the runs use the same manifest, "
             "seed and configuration except for class-weighted loss. These are preliminary development results.", "",
             "| Fold | Unweighted AP | Weighted AP | Difference |", "|---:|---:|---:|---:|"]
    for fold in range(5):
        x, y = results[0][fold], results[1][fold]
        lines.append(f"| {fold} | {x:.4f} | {y:.4f} | {y - x:+.4f} |")
    diffs = [results[1][i] - results[0][i] for i in range(5)]
    lines += ["", f"Mean paired difference: **{np.mean(diffs):+.4f}** (descriptive, not a significance test).", "",
              "One epoch does not establish convergence or superiority. Fold results are dependent, "
              "and unknown patient overlap, curated prevalence and source-video effects limit inference.\n"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    runtime = []
    for directory in (unweighted, weighted):
        config = json.loads((directory / "trial_000/fold_0/config.json").read_text(encoding="utf-8"))
        for path_key in ("raw", "manifest", "output", "weights_cache"):
            config.pop(path_key, None)
        runtime.append(config)
    output.with_suffix(".json").write_text(json.dumps({"unweighted": left, "weighted": right, "folds": results,
                                                       "runtime": runtime, "mean_paired_difference": float(np.mean(diffs))}, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("unweighted", type=Path); p.add_argument("weighted", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); compare(a.unweighted, a.weighted, a.output)


if __name__ == "__main__":
    main()
