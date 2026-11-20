"""Generate a report from complete development tuning and frozen selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(search: Path, output: Path) -> None:
    identity = json.loads((search / "search.json").read_text(encoding="utf-8"))
    selected = json.loads((search / "selected_candidate.json").read_text(encoding="utf-8"))
    folds = json.loads((search / "fold_results.json").read_text(encoding="utf-8"))
    if selected["search_sha256"] != hashlib.sha256((search / "search.json").read_bytes()).hexdigest() or selected["fold_results_sha256"] != hashlib.sha256((search / "fold_results.json").read_bytes()).hexdigest():
        raise ValueError("Search or fold results changed after selection")
    lines = ["# Development tuning results", "",
             "The candidates were chosen after the feasibility experiments and evaluated on the same development folds. "
             "The table is selection data, not an unbiased performance estimate. Both learning rate and fine-tuning depth differ.", "",
             "| Trial | Fine-tuning | Learning rate | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean AP | SD |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    results = []
    for trial, config in enumerate(identity["configurations"]):
        rows = sorted((r for r in folds if r["trial"] == trial), key=lambda row: row["fold"])
        if len(rows) != 5 or [r["fold"] for r in rows] != list(range(5)) or any(r["status"] != "complete" for r in rows):
            raise ValueError("Every trial must have five completed folds")
        scores = [float(r["best_validation_average_precision"]) for r in rows]
        mean, sd = float(np.mean(scores)), float(np.std(scores, ddof=1))
        lines.append(f"| {trial} | {config['fine_tuning']} | {config['lr']:.4g} | " + " | ".join(f"{s:.4f}" for s in scores) + f" | {mean:.4f} | {sd:.4f} |")
        results.append({"trial": trial, "scores": scores, "mean": mean, "sample_sd": sd})
    if selected["trial_index"] != int(np.argmax([r["mean"] for r in results])):
        raise ValueError("Frozen selection does not match fold scores")
    lines.extend(["", f"Selected trial: **{selected['trial_index']}**, using the prespecified highest-mean-AP rule.", "",
                  "Variation is descriptive. A short training budget, source imbalance and unknown patient overlap limit interpretation.\n"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps({"identity": identity, "selection": selected, "results": results}, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("search", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); summarize(a.search, a.output)


if __name__ == "__main__":
    main()
