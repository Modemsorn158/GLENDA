import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from glenda.evaluate_test import evaluate
from glenda.summarize_test import summarize
from glenda.summarize_error_analysis import summarize as summarize_error_analysis


class TestEvaluationTests(unittest.TestCase):
    def test_rejects_unverified_library_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Trusted inventory"):
                evaluate(root / "raw", root / "manifest", root / "checkpoint", root / "out", root / "ledger")

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
                            Image.new("RGB", (12, 12), (label * 255, i * 60, 0)).save(raw / name)
                        rows.append(dict(path=name, label=label, group=name, partition=partition, cv_fold=fold if fold is not None else ""))
            with manifest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            inventory = root / "inventory.csv"
            with inventory.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "label", "case_id", "video_id", "sha256"]); writer.writeheader()
                for row in rows:
                    image = raw / row["path"]
                    digest = hashlib.sha256(image.read_bytes() if image.exists() else row["path"].encode()).hexdigest()
                    writer.writerow(dict(path=row["path"], label=row["label"], case_id="", video_id=row["path"], sha256=digest))
            candidates = root / "pairs.csv"; candidates.write_text("path_a,path_b\n", encoding="utf-8")
            inventory_hash = hashlib.sha256(inventory.read_bytes()).hexdigest()
            candidate_hash = hashlib.sha256(candidates.read_bytes()).hexdigest()
            checkpoint = root / "final.pt"; checkpoint.write_bytes(b"checkpoint")
            config = {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "threshold": 0.5,
                      "inventory_sha256": inventory_hash, "candidates_sha256": candidate_hash}
            with patch("glenda.evaluate_test.load_final", return_value=(None, config)), \
                 patch("glenda.evaluate_test.predict_image", side_effect=lambda model, image, device: 0.4 if image.getpixel((0, 0))[0] else 0.6):
                evaluate(raw, manifest, checkpoint, root / "result", root / "ledger", bootstrap=10,
                         inventory=inventory, inventory_sha256=inventory_hash, candidates=candidates, candidates_sha256=candidate_hash)
                with self.assertRaises(FileExistsError):
                    evaluate(raw, manifest, checkpoint, root / "second", root / "ledger", bootstrap=10,
                             inventory=inventory, inventory_sha256=inventory_hash, candidates=candidates, candidates_sha256=candidate_hash)
                revised = root / "revised.csv"
                updated = [dict(row, cv_fold=(1 if row["cv_fold"] == 0 and row["partition"] == "development" else row["cv_fold"])) for row in rows]
                with revised.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(updated)
                config["manifest_sha256"] = hashlib.sha256(revised.read_bytes()).hexdigest()
                with self.assertRaises(FileExistsError):
                    evaluate(raw, revised, checkpoint, root / "third", root / "ledger", bootstrap=10,
                             inventory=inventory, inventory_sha256=inventory_hash, candidates=candidates, candidates_sha256=candidate_hash)
            self.assertTrue((root / "result/metrics.json").exists())
            self.assertTrue((root / "result/evaluation.png").exists())
            self.assertTrue((root / "result/group_errors.csv").exists())
            self.assertTrue((root / "result/error_examples.png").exists())
            self.assertTrue((root / "result/error_examples.csv").exists())
            with (root / "result/predictions.csv").open(encoding="utf-8", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)
            self.assertFalse((root / "second").exists())
            summarize(root / "result", root / "ledger", root / "report.md")
            self.assertIn("Locked holdout evaluation", (root / "report.md").read_text(encoding="utf-8"))
            summarize_error_analysis(root / "result", root / "ledger", manifest, root / "error_report.md")
            self.assertIn("2 observed false-positive frames", (root / "error_report.md").read_text(encoding="utf-8"))
            for name in ("metrics.json", "predictions.csv"):
                path = root / "result" / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "hashes do not match"):
                    summarize(root / "result", root / "ledger", root / "changed.md")
                with self.assertRaisesRegex(ValueError, "hashes do not match"):
                    summarize_error_analysis(root / "result", root / "ledger", manifest, root / "changed-error.md")
                path.write_bytes(original)
            lock = next((root / "ledger").glob("*.json"))
            saved = lock.read_bytes()
            legacy = json.loads(saved); legacy.pop("artifact_sha256")
            lock.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Legacy ledger"):
                summarize(root / "result", root / "ledger", root / "legacy.md")
            summarize(root / "result", root / "ledger", root / "legacy.md", allow_legacy_unverified=True)
            self.assertIn("result-file integrity has not been verified", (root / "legacy.md").read_text())
            lock.write_bytes(saved)


if __name__ == "__main__":
    unittest.main()
