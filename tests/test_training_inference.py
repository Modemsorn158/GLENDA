"""Exercise training and prediction using only generated images."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from glenda.inference import load_final, predict_image


class TrainingInferenceTest(unittest.TestCase):
    def test_random_training_checkpoint_and_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "images"
            raw.mkdir()
            rows = []
            for fold in range(2):
                for label in range(2):
                    name = f"fold{fold}_class{label}.png"
                    Image.new("RGB", (32, 24), (30 + label * 180, 40 + fold * 50, 80)).save(raw / name)
                    rows.append(dict(path=name, label=label, group=name, partition="development", cv_fold=fold))
            rows.append(dict(path="must_not_be_opened.png", label=1, group="must_not_be_opened.png", partition="test", cv_fold=""))
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "label", "group", "partition", "cv_fold"])
                writer.writeheader()
                writer.writerows(rows)
            inventory = root / "inventory.csv"
            with inventory.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "label", "case_id", "video_id", "sha256"])
                writer.writeheader()
                for row in rows:
                    path = raw / row["path"]
                    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "0" * 64
                    writer.writerow(dict(path=row["path"], label=row["label"], case_id="", video_id=row["path"], sha256=digest))
            candidates = root / "pairs.csv"; candidates.write_text("path_a,path_b\n", encoding="utf-8")
            inventory_hash = hashlib.sha256(inventory.read_bytes()).hexdigest()
            candidate_hash = hashlib.sha256(candidates.read_bytes()).hexdigest()
            output = root / "run"
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            result = subprocess.run([sys.executable, "-m", "glenda.train_fold", str(raw), str(manifest),
                                     str(output), "--fold", "0", "--epochs", "1", "--batch-size", "2",
                                     "--random-init", "--cpu", "--inventory", str(inventory), "--inventory-sha256", inventory_hash,
                                     "--candidates", str(candidates), "--candidates-sha256", candidate_hash], capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["random_init"])
            self.assertEqual(config["device"], "cpu")
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            checkpoint["config"]["threshold"] = 0.5
            final = root / "synthetic_final.pt"
            torch.save(checkpoint, final)
            model, _ = load_final(final, torch.device("cpu"))
            with (output / "validation_predictions.csv").open(encoding="utf-8", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual({row["path"] for row in predictions}, {"fold0_class0.png", "fold0_class1.png"})
            old_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                for row in predictions:
                    with Image.open(raw / row["path"]) as image:
                        score = predict_image(model, image, torch.device("cpu"))
                    self.assertAlmostEqual(score, float(row["score"]), places=5)
            finally:
                torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
