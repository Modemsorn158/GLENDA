import json
import tempfile
import unittest
from pathlib import Path

from glenda.select_candidate import select


class CandidateSelectionTests(unittest.TestCase):
    def test_complete_trials_only_and_tie_break(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "search.json").write_text(json.dumps({"configurations": [{"architecture": "efficientnet_b0"}, {"architecture": "efficientnet_b0"}]}))
            rows = [{"trial": trial, "fold": fold, "status": "complete", "best_validation_average_precision": 0.5} for trial in range(2) for fold in range(5)]
            (root / "fold_results.json").write_text(json.dumps(rows[:-1]))
            with self.assertRaises(ValueError):
                select(root)
            (root / "fold_results.json").write_text(json.dumps(rows))
            self.assertEqual(select(root)["trial_index"], 0)
            with self.assertRaises(FileExistsError):
                select(root)


if __name__ == "__main__":
    unittest.main()
