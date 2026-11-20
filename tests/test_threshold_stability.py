import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from glenda.threshold_stability import analyze


class ThresholdStabilityTests(unittest.TestCase):
    def test_development_only_transfer_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); trial = root / "trial_001"; trial.mkdir()
            manifest = root / "manifest.csv"
            rows, hashes = [], []
            for fold in range(5):
                folder = trial / f"fold_{fold}"; folder.mkdir()
                predictions = folder / "validation_predictions.csv"
                with predictions.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["path", "label", "score", "group"]); writer.writeheader()
                    for label in (0, 1):
                        path = f"{fold}-{label}"
                        rows.append({"path": path, "label": label, "group": path, "partition": "development", "cv_fold": fold})
                        writer.writerow({"path": path, "label": label, "score": 0.1 + label * 0.7, "group": path})
                hashes.append(hashlib.sha256(predictions.read_bytes()).hexdigest())
            rows.append({"path": "test", "label": 1, "group": "test", "partition": "test", "cv_fold": ""})
            with manifest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "label", "group", "partition", "cv_fold"]); writer.writeheader(); writer.writerows(rows)
            candidate = root / "selected_candidate.json"; candidate.write_text("{}", encoding="utf-8")
            selection = root / "threshold.json"
            selection.write_text(json.dumps({"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                             "trial_index": 1, "candidate_selection_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                                             "fold_prediction_sha256": hashes, "threshold": 0.8}), encoding="utf-8")
            result = analyze(manifest, trial, selection, root / "report.md")
            self.assertEqual(len(result["folds"]), 5)
            self.assertTrue(all(row["transferred_metrics"]["recall"] == 1 for row in result["folds"]))
            (trial / "fold_0/validation_predictions.csv").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze(manifest, trial, selection, root / "another.md")


if __name__ == "__main__":
    unittest.main()
