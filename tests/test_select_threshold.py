import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from glenda.select_threshold import select
from glenda.select_candidate import select as select_candidate


class ThresholdTests(unittest.TestCase):
    def test_selects_only_complete_development_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); trial = root / "trial_000"; trial.mkdir()
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "label", "group", "partition", "cv_fold"])
                writer.writeheader()
                for fold in range(5):
                    for label in range(2):
                        path = f"{fold}-{label}"
                        writer.writerow(dict(path=path, label=label, group=path, partition="development", cv_fold=fold))
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (root / "search.json").write_text(json.dumps({"manifest_sha256": manifest_hash, "configurations": [{}]}))
            (root / "fold_results.json").write_text(json.dumps([{"trial": 0, "fold": fold, "status": "complete", "best_validation_average_precision": 1.0} for fold in range(5)]))
            for fold in range(5):
                (trial / f"fold_{fold}_status.json").write_text('{"status": "complete", "best_validation_average_precision": 1.0}')
                folder = trial / f"fold_{fold}"; folder.mkdir()
                (folder / "config.json").write_text(json.dumps({"manifest_sha256": manifest_hash, "fold": fold}))
                (folder / "epochs.csv").write_text("validation_average_precision\n1.0\n")
                (folder / "best.pt").write_bytes(b"checkpoint")
                with (folder / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["path", "label", "score", "group"]); writer.writeheader()
                    for label in range(2):
                        path = f"{fold}-{label}"
                        writer.writerow(dict(path=path, label=label, score=0.2 if label == 0 else 0.8, group=path))
            select_candidate(root)
            result = select(manifest, trial, root / "threshold.json")
            self.assertEqual(result["threshold"], 0.8)
            self.assertEqual(result["oof_f1"], 1.0)
            with self.assertRaises(FileExistsError):
                select(manifest, trial, root / "threshold.json")


if __name__ == "__main__":
    unittest.main()
