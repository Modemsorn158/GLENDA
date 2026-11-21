import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from glenda.summarize_development import summarize


class DevelopmentSummaryTests(unittest.TestCase):
    def test_completed_folds_and_tampered_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.csv"
            rows = [dict(path=f"{fold}-{label}", group=f"g{fold}-{label}", label=str(label),
                         partition="development", cv_fold=str(fold)) for fold in range(5) for label in range(2)]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            config = {"fine_tuning": "full"}
            (root / "search.json").write_text(json.dumps({"manifest_sha256": manifest_hash, "configurations": [config]}))
            results = [dict(trial=0, fold=fold, status="complete", best_validation_average_precision=1.0) for fold in range(5)]
            (root / "fold_results.json").write_text(json.dumps(results))
            for fold in range(5):
                folder = root / "trial_000" / f"fold_{fold}"; folder.mkdir(parents=True)
                (folder.parent / f"fold_{fold}_status.json").write_text(json.dumps({"status": "complete", "best_validation_average_precision": 1.0}))
                (folder / "config.json").write_text(json.dumps({**config, "fold": fold, "manifest_sha256": manifest_hash}))
                (folder / "epochs.csv").write_text("epoch,validation_average_precision\n1,1.0\n")
                (folder / "best.pt").write_bytes(b"synthetic checkpoint")
                with (folder / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["path", "label", "group", "score"]); writer.writeheader()
                    for row in rows:
                        if row["cv_fold"] == str(fold):
                            writer.writerow({**{key: row[key] for key in ("path", "label", "group")}, "score": 0.2 if row["label"] == "0" else 0.8})
            output = root / "report.md"
            summarize(root, manifest, output)
            self.assertIn("1.0000", output.read_text(encoding="utf-8"))
            published = output.with_suffix(".json").read_text(encoding="utf-8")
            self.assertNotIn('"path"', published)
            self.assertAlmostEqual(json.loads(published)["trials"][0]["folds"][0]["brier"], 0.04)
            prediction_file = root / "trial_000/fold_0/validation_predictions.csv"
            prediction_file.write_text(prediction_file.read_text().replace("0.8", "0.1"))
            with self.assertRaisesRegex(ValueError, "AP does not match"):
                summarize(root, manifest, root / "tampered.md")


if __name__ == "__main__":
    unittest.main()
