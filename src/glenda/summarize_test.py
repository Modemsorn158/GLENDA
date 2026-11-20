"""Create a de-identified report from a completed locked test evaluation."""
import argparse
import json
from pathlib import Path


def summarize(test: Path, ledger: Path, output: Path) -> None:
    data = json.loads((test / "metrics.json").read_text(encoding="utf-8"))
    provenance = data["provenance"]
    lock = json.loads((ledger / f"{provenance['manifest_sha256']}.json").read_text(encoding="utf-8"))
    if lock["status"] != "complete" or lock["checkpoint_sha256"] != provenance["checkpoint_sha256"] or Path(lock["output"]).resolve() != test.resolve():
        raise ValueError("Completed test ledger does not match results")
    point, intervals = data["point"], data["intervals"]
    def fmt(value):
        return "undefined" if value is None else f"{value:.4f}"
    keys = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("precision", "Precision"),
            ("recall", "Recall / sensitivity"), ("specificity", "Specificity"), ("f1", "F1"),
            ("roc_auc", "ROC-AUC"), ("average_precision", "Average precision"), ("brier", "Brier score")]
    lines = ["# Locked holdout evaluation", "", f"Frames: {point['n']}; positive-frame fraction: {point['prevalence']:.4f}; fixed threshold: {point['threshold']:.4f}.",
             f"Known linked groups: {provenance['known_groups']}; groups containing negative frames: {provenance['groups_with_negative_frames']}.", "",
             f"Confusion counts: TP {point['tp']}, FP {point['fp']}, TN {point['tn']}, FN {point['fn']}.", "",
             "| Metric | Estimate | Cluster bootstrap 95% interval | Valid replicates |", "|---|---:|---:|---:|"]
    for key, label in keys:
        interval = intervals[key]
        bounds = f"{fmt(interval['lower'])}–{fmt(interval['upper'])}" if interval["lower"] is not None else "undefined"
        lines.append(f"| {label} | {fmt(point[key])} | {bounds} | {interval['valid_replicates']} |")
    illustration = data["illustration_per_1000_at_release_frame_prevalence"]
    lines.extend(["", f"At the curated release-frame prevalence of {illustration['prevalence']:.4f}, applying these held-out rates gives an illustrative {illustration['false_positives']:.1f} false positives and {illustration['false_negatives']:.1f} false negatives per 1,000 frames, and precision {fmt(illustration['precision'])}. This is not a clinical forecast.", "",
                  "| Excluded negative-containing group | Remaining specificity | Remaining sensitivity |", "|---|---:|---:|"])
    for index, row in enumerate(data["leave_one_negative_group_out"], 1):
        lines.append(f"| Group {index} | {fmt(row['metrics']['specificity'])} | {fmt(row['metrics']['recall'])} |")
    lines.extend(["", "The leave-one-group-out results are descriptive sensitivity analyses at the fixed threshold, not independent tests or a basis for retuning. "
                  "Only three known groups contribute negative frames, one dominates the negative test frames, and unmapped patient relationships remain possible. "
                  "Intervals are conditional on the available groups and may be unstable. This dataset has been used previously; external validation is absent.\n"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    safe = {key: data[key] for key in ("point", "intervals", "requested_replicates", "seed", "provenance", "illustration_per_1000_at_release_frame_prevalence")}
    output.with_suffix(".json").write_text(json.dumps(safe, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("test", type=Path); p.add_argument("ledger", type=Path); p.add_argument("output", type=Path)
    a = p.parse_args(); summarize(a.test, a.ledger, a.output)


if __name__ == "__main__":
    main()
