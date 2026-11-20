import json
import tempfile
import unittest
from pathlib import Path

from glenda.compare_architectures import compare


class ArchitectureComparisonTests(unittest.TestCase):
    def test_requires_matched_budget_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, architecture in (("a", "efficientnet_b0"), ("b", "resnet50")):
                run = root / name; run.mkdir()
                config = {"architecture": architecture, "epochs": 1}
                (run / "search.json").write_text(json.dumps({"manifest_sha256": "same", "configurations": [config]}))
                rows = []
                for fold in range(5):
                    folder = run / f"trial_000/fold_{fold}"; folder.mkdir(parents=True)
                    keys = ("seed", "fine_tuning", "lr", "batch_size", "epochs", "patience", "weight_decay", "weighted", "manifest_sha256", "torch_version", "torchvision_version", "transforms")
                    base = {key: "same" for key in keys}
                    (folder / "config.json").write_text(json.dumps({**base, "fold": fold, "architecture": architecture}))
                    rows.append({"fold": fold, "status": "complete", "best_validation_average_precision": 0.2})
                (run / "fold_results.json").write_text(json.dumps(rows))
            compare(root / "a", root / "b", root / "comparison.md")
            self.assertIn("+0.0000", (root / "comparison.md").read_text())
            (root / "b/search.json").write_text(json.dumps({"manifest_sha256": "same", "configurations": [{"architecture": "resnet50", "epochs": 2}]}))
            with self.assertRaises(ValueError):
                compare(root / "a", root / "b", root / "other.md")


if __name__ == "__main__":
    unittest.main()
