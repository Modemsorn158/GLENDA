"""Run predeclared configurations on all development folds, recording failures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


OPTIONS = {"architecture", "fine_tuning", "epochs", "patience", "batch_size", "lr", "weight_decay", "seed", "weighted", "cpu", "random_init"}


def run(raw: Path, manifest: Path, config_path: Path, output: Path) -> None:
    raw, output = raw.resolve(), output.resolve()
    if output == raw or raw in output.parents:
        raise ValueError("Experiment output must be outside raw data")
    configurations = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("Search configuration must be a nonempty JSON list")
    for config in configurations:
        if not isinstance(config, dict) or set(config) - OPTIONS:
            raise ValueError("Unknown configuration option")
        if any(not isinstance(config.get(key, False), bool) for key in ("weighted", "cpu", "random_init")):
            raise ValueError("Boolean options must be true or false")
    identity = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "configurations": configurations}
    output.mkdir(parents=True, exist_ok=True)
    identity_file = output / "search.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing search uses different configurations or manifest")
    else:
        identity_file.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    results, summaries = [], []
    for trial, config in enumerate(configurations):
        for fold in range(5):
            run_dir = output / f"trial_{trial:03d}" / f"fold_{fold}"
            status_file = run_dir.parent / f"fold_{fold}_status.json"
            if status_file.exists():
                status = json.loads(status_file.read_text(encoding="utf-8"))
            elif run_dir.exists():
                status = {"status": "interrupted", "reason": "Run directory exists without completion status"}
                status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
            else:
                run_dir.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-m", "glenda.train_fold", str(raw), str(manifest.resolve()), str(run_dir), "--fold", str(fold), "--weights-cache", str(output / ".torch-cache")]
                for key, value in config.items():
                    flag = "--" + key.replace("_", "-")
                    if isinstance(value, bool):
                        if value:
                            command.append(flag)
                    else:
                        command.extend([flag, str(value)])
                with (run_dir.parent / f"fold_{fold}.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                status = {"status": "failed", "exit_code": result.returncode}
                if result.returncode == 0 and (run_dir / "best.pt").is_file():
                    with (run_dir / "epochs.csv").open(encoding="utf-8", newline="") as f:
                        scores = [float(row["validation_average_precision"]) for row in csv.DictReader(f)]
                    if scores:
                        status = {"status": "complete", "best_validation_average_precision": max(scores)}
                status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
            results.append({"trial": trial, "fold": fold, **status})
            (output / "fold_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        completed = [r["best_validation_average_precision"] for r in results if r["trial"] == trial and r["status"] == "complete"]
        summaries.append({"trial": trial, "configuration": config, "completed_folds": len(completed),
                          "mean_average_precision": float(np.mean(completed)) if len(completed) == 5 else None,
                          "std_average_precision": float(np.std(completed, ddof=1)) if len(completed) == 5 else None})
        (output / "trial_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"trial={trial} completed={len(completed)}/5" + (f" mean_ap={np.mean(completed):.6f}" if len(completed) == 5 else ""), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path); parser.add_argument("manifest", type=Path)
    parser.add_argument("config", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.raw, args.manifest, args.config, args.output)


if __name__ == "__main__":
    main()
