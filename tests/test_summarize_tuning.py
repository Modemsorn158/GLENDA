import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from glenda.summarize_tuning import summarize


class TuningReportTests(unittest.TestCase):
    def test_report_requires_unchanged_selection_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = root / "search.json"; identity.write_text(json.dumps({"configurations": [{"fine_tuning": "frozen", "lr": 0.001}]}))
            folds = root / "fold_results.json"
            folds.write_text(json.dumps([{"trial": 0, "fold": i, "status": "complete", "best_validation_average_precision": 0.5} for i in range(5)]))
            (root / "selected_candidate.json").write_text(json.dumps({"trial_index": 0, "search_sha256": hashlib.sha256(identity.read_bytes()).hexdigest(), "fold_results_sha256": hashlib.sha256(folds.read_bytes()).hexdigest()}))
            for fold in range(5):
                directory = root / f"trial_000/fold_{fold}"; directory.mkdir(parents=True)
                (directory / "epochs.csv").write_text("epoch,training_loss,validation_average_precision\n1,0.4,0.5\n")
            summarize(root, root / "report.md")
            self.assertIn("0.5000", (root / "report.md").read_text())
            self.assertTrue((root / "report.epochs.csv").exists())
            folds.write_text("[]")
            with self.assertRaises(ValueError):
                summarize(root, root / "other.md")


if __name__ == "__main__":
    unittest.main()
