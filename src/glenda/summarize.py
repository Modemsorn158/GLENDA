"""Generate a descriptive development-fold table from recorded outputs."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(search: Path, protocol: Path, output: Path) -> None:
    folds = json.loads((search / "fold_results.json").read_text(encoding="utf-8"))
    summaries = json.loads((search / "trial_summary.json").read_text(encoding="utf-8"))
    split = json.loads(protocol.read_text(encoding="utf-8"))
    identity = json.loads((search / "search.json").read_text(encoding="utf-8"))
    if identity["manifest_sha256"] != hashlib.sha256((protocol.parent / "manifest.csv").read_bytes()).hexdigest():
        raise ValueError("Search and split manifest differ")
    if len(summaries) != 1 or summaries[0]["completed_folds"] != 5 or len(folds) != 5 or any(row["status"] != "complete" for row in folds):
        raise ValueError("Expected exactly one configuration with five complete folds")
    s = summaries[0]
    lines = ["# Five-fold feasibility results", "",
             "Generated from `outputs/search/feasibility/` and the recorded split protocol. "
             "This is a single one-epoch configuration, not a comparison or held-out test.", "",
             "The constant-score baseline has average precision equal to the fold's positive-frame fraction.", "",
             "| Development fold | Positive frames | Negative frames | Constant-score AP | Model AP |", "|---:|---:|---:|---:|---:|"]
    for row in sorted(folds, key=lambda item: item["fold"]):
        count = split["validation_counts"][str(row["fold"])]
        with (search / "trial_000" / f"fold_{row['fold']}" / "validation_predictions.csv").open(encoding="utf-8", newline="") as f:
            observed = Counter(item["label"] for item in csv.DictReader(f))
        if dict(observed) != count:
            raise ValueError("Prediction counts do not match the split protocol")
        positive, negative = count.get("1", 0), count.get("0", 0)
        lines.append(f"| {row['fold']} | {positive} | {negative} | {positive / (positive + negative):.4f} | {row['best_validation_average_precision']:.4f} |")
    lines.extend(["", f"Mean average precision: **{s['mean_average_precision']:.4f}**; sample standard deviation: **{s['std_average_precision']:.4f}**.",
                  "", "Fold variation is descriptive and does not establish statistical significance. "
                  "Unknown patient links, curated prevalence and model selection limit interpretation.\n"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    run_config = json.loads((search / "trial_000/fold_0/config.json").read_text(encoding="utf-8"))
    for path_key in ("raw", "manifest", "output", "weights_cache"):
        run_config.pop(path_key, None)
    record = {"search": identity, "split": split, "folds": folds, "summary": s, "runtime": run_config}
    output.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("search", type=Path); p.add_argument("protocol", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); summarize(a.search, a.protocol, a.output)


if __name__ == "__main__":
    main()
