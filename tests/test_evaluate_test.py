import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from glenda.evaluate_test import evaluate
from glenda.summarize_test import summarize


class TestEvaluationTests(unittest.TestCase):
    def test_once_only_ledger_and_heldout_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw"; raw.mkdir()
            manifest = root / "manifest.csv"
            rows = []
            for partition, folds in (("development", (0, 1)), ("test", (None, None))):
                for i, fold in enumerate(folds):
                    for label in (0, 1):
                        name = f"{partition}-{i}-{label}.png"
                        if partition == "test":
                            Image.new("RGB", (12, 12), (label * 255, 0, 0)).save(raw / name)
                        rows.append(dict(path=name, label=label, group=name, partition=partition, cv_fold=fold if fold is not None else ""))
            with manifest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            checkpoint = root / "final.pt"; checkpoint.write_bytes(b"checkpoint")
            config = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "threshold": 0.5}
            with patch("glenda.evaluate_test.load_final", return_value=(None, config)), \
                 patch("glenda.evaluate_test.predict_image", side_effect=lambda model, image, device: 0.4 if image.getpixel((0, 0))[0] else 0.6):
                evaluate(raw, manifest, checkpoint, root / "result", root / "ledger", bootstrap=10)
                with self.assertRaises(FileExistsError):
                    evaluate(raw, manifest, checkpoint, root / "second", root / "ledger", bootstrap=10)
                revised = root / "revised.csv"
                updated = [dict(row, cv_fold=(1 if row["cv_fold"] == 0 and row["partition"] == "development" else row["cv_fold"])) for row in rows]
                with revised.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(updated)
                config["manifest_sha256"] = hashlib.sha256(revised.read_bytes()).hexdigest()
                with self.assertRaises(FileExistsError):
                    evaluate(raw, revised, checkpoint, root / "third", root / "ledger", bootstrap=10)
            self.assertTrue((root / "result/metrics.json").exists())
            self.assertTrue((root / "result/evaluation.png").exists())
            self.assertTrue((root / "result/group_errors.csv").exists())
            self.assertTrue((root / "result/error_examples.png").exists())
            with (root / "result/predictions.csv").open(encoding="utf-8", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)
            self.assertFalse((root / "second").exists())
            summarize(root / "result", root / "ledger", root / "report.md")
            self.assertIn("Locked holdout evaluation", (root / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
