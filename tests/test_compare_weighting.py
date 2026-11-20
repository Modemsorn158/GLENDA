import json
import tempfile
import unittest
from pathlib import Path

from glenda.compare_weighting import compare


class ComparisonTests(unittest.TestCase):
    def test_requires_matched_manifests_and_configs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, weighted in (("a", False), ("b", True)):
                run = root / name; run.mkdir()
                config = {"weighted": weighted, "epochs": 1}
                (run / "search.json").write_text(json.dumps({"manifest_sha256": "same", "configurations": [config]}))
                results = []
                for fold in range(5):
                    folder = run / f"trial_000/fold_{fold}"; folder.mkdir(parents=True)
                    base = {key: "same" for key in ("seed", "architecture", "fine_tuning", "lr", "batch_size", "epochs", "patience", "weight_decay", "manifest_sha256", "pretrained_weights_sha256", "torch_version", "torchvision_version", "transforms")}
                    (folder / "config.json").write_text(json.dumps({**base, "fold": fold, "weighted": weighted}))
                    results.append({"fold": fold, "status": "complete", "best_validation_average_precision": 0.2})
                (run / "fold_results.json").write_text(json.dumps(results))
            compare(root / "a", root / "b", root / "comparison.md")
            self.assertIn("+0.0000", (root / "comparison.md").read_text())
            (root / "b/search.json").write_text(json.dumps({"manifest_sha256": "different", "configurations": [{"weighted": True, "epochs": 1}]}))
            with self.assertRaises(ValueError):
                compare(root / "a", root / "b", root / "other.md")


if __name__ == "__main__":
    unittest.main()
