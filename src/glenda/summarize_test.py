"""Create a de-identified report from a completed locked test evaluation."""
import argparse
import csv
import json
from pathlib import Path
from glenda.evaluation_integrity import verify_results


def summarize(test: Path, ledger: Path, output: Path, *, allow_legacy_unverified=False) -> None:
    data = json.loads((test / "metrics.json").read_text(encoding="utf-8"))
    provenance = data["provenance"]
    lock = json.loads((ledger / f"{provenance['holdout_sha256']}.json").read_text(encoding="utf-8"))
    if lock["status"] != "complete" or lock["checkpoint_sha256"] != provenance["checkpoint_sha256"] or Path(lock["output"]).resolve() != test.resolve():
        raise ValueError("Completed test ledger does not match results")
    verify_results(test, lock, allow_legacy_unverified=allow_legacy_unverified)
    point, intervals = data["point"], data["intervals"]
    with (test / "group_errors.csv").open(encoding="utf-8", newline="") as f:
        group_rows = list(csv.DictReader(f))
    negative_sizes = [int(row["frames"]) - int(row["positives"]) for row in group_rows]
    negative_sizes = [count for count in negative_sizes if count > 0]
    if len(negative_sizes) != provenance["groups_with_negative_frames"] or sum(negative_sizes) != point["tn"] + point["fp"]:
        raise ValueError("Group counts do not match the recorded test results")
    largest_negative_share = max(negative_sizes) / sum(negative_sizes)
    def fmt(value):
        return "undefined" if value is None else f"{value:.4f}"
    keys = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("precision", "Precision"),
            ("recall", "Recall / sensitivity"), ("specificity", "Specificity"), ("f1", "F1"),
            ("roc_auc", "ROC-AUC"), ("average_precision", "Average precision"), ("brier", "Brier score")]
    lines = ["# Locked holdout evaluation", "", f"Frames: {point['n']}; positive-frame fraction: {point['prevalence']:.4f}; fixed threshold: {point['threshold']:.8f}.",
             f"Known linked groups: {provenance['known_groups']}; groups containing negative frames: {provenance['groups_with_negative_frames']}.", "",
             f"Confusion counts: TP {point['tp']}, FP {point['fp']}, TN {point['tn']}, FN {point['fn']}.",
             f"An always-negative classifier would have accuracy {(point['tn'] + point['fp']) / point['n']:.4f} and sensitivity 0 on this holdout.", "",
             "| Metric | Estimate | Cluster bootstrap 95% interval | Valid replicates |", "|---|---:|---:|---:|"]
    if "artifact_sha256" not in lock:
        lines[2:2] = ["**Legacy result:** the original ledger did not record output hashes; result-file integrity has not been verified by that ledger.", ""]
    for key, label in keys:
        interval = intervals[key]
        bounds = f"{fmt(interval['lower'])}–{fmt(interval['upper'])}" if interval["lower"] is not None else "undefined"
        lines.append(f"| {label} | {fmt(point[key])} | {bounds} | {interval['valid_replicates']} |")
    default = next(item for item in data["fixed_threshold_sensitivity"] if item["threshold"] == 0.5)
    lines.extend(["", "| Fixed operating point | Precision | Recall | Specificity | F1 |", "|---|---:|---:|---:|---:|",
                  f"| Development-selected ({point['threshold']:.8f}) | {fmt(point['precision'])} | {fmt(point['recall'])} | {fmt(point['specificity'])} | {fmt(point['f1'])} |",
                  f"| 0.5 reference | {fmt(default['precision'])} | {fmt(default['recall'])} | {fmt(default['specificity'])} | {fmt(default['f1'])} |"])
    illustration = data["illustration_per_1000_at_release_frame_prevalence"]
    lines.extend(["", f"At the curated release-frame prevalence of {illustration['prevalence']:.4f}, applying these held-out rates gives an illustrative {illustration['false_positives']:.1f} false positives and {illustration['false_negatives']:.1f} false negatives per 1,000 frames, and precision {fmt(illustration['precision'])}. This is not a clinical forecast.", "",
                  "| Excluded negative-containing group | Remaining specificity | Remaining sensitivity |", "|---|---:|---:|"])
    for index, row in enumerate(data["leave_one_negative_group_out"], 1):
        lines.append(f"| Group {index} | {fmt(row['metrics']['specificity'])} | {fmt(row['metrics']['recall'])} |")
    zero_event_note = " With zero observed false positives, the empirical bootstrap produces degenerate intervals for some metrics; those intervals do not establish perfect specificity or precision on unseen data." if point["fp"] == 0 else ""
    lines.extend(["", "The leave-one-group-out results are descriptive sensitivity analyses at the fixed threshold, not independent tests or a basis for retuning. "
                  f"{len(negative_sizes)} known groups contribute negative frames, with {largest_negative_share:.1%} of those frames in the largest group; unmapped patient relationships remain possible. "
                  f"Intervals are conditional on the available groups and may be unstable.{zero_event_note} This dataset has been used previously; external validation is absent.\n"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    safe = {key: data[key] for key in ("point", "intervals", "requested_replicates", "seed", "provenance", "illustration_per_1000_at_release_frame_prevalence")}
    output.with_suffix(".json").write_text(json.dumps(safe, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("test", type=Path); p.add_argument("ledger", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--allow-legacy-unverified", action="store_true", help="Acknowledge that old ledgers have no output hashes")
    a = p.parse_args(); summarize(a.test, a.ledger, a.output, allow_legacy_unverified=a.allow_legacy_unverified)


if __name__ == "__main__":
    main()
