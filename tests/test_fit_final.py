import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from glenda.fit_final import fit
from glenda.select_threshold import select
from glenda.select_candidate import select as select_candidate


class FinalFitTests(unittest.TestCase):
    def test_rejects_unverified_library_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Trusted inventory"):
                fit(root / "raw", root / "manifest", root / "trial", root / "threshold", root / "output")

    def test_final_fit_uses_development_and_selected_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw"; raw.mkdir()
            search = root / "search"; search.mkdir(); trial = search / "trial_000"; trial.mkdir()
            cache = search / ".torch-cache/hub/checkpoints"; cache.mkdir(parents=True)
            weights = cache / "fake.pth"; weights.write_bytes(b"fake-pretrained-weights")
            weights_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
            manifest = root / "manifest.csv"; rows = []
            for fold in range(5):
                for label in range(2):
                    name = f"{fold}-{label}.png"
                    Image.new("RGB", (16, 16), (label * 255, 0, 0)).save(raw / name)
                    rows.append(dict(path=name, label=label, group=name, partition="development", cv_fold=fold))
            rows.append(dict(path="must-not-open.png", label=0, group="test", partition="test", cv_fold=""))
            with manifest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            chosen = {"architecture": "efficientnet_b0", "fine_tuning": "frozen", "lr": 0.001,
                      "weight_decay": 0.0, "batch_size": 2, "seed": 42, "weighted": False,
                      "random_init": False, "cpu": True}
            source_dir = Path(__file__).resolve().parents[1] / "src" / "glenda"
            source_hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest() for name in ("dataset.py", "models.py", "train_fold.py", "verify_data.py", "splitting.py")}
            (search / "search.json").write_text(json.dumps({"manifest_sha256": digest, "configurations": [chosen], "training_source_sha256": source_hashes}))
            (search / "fold_results.json").write_text(json.dumps([{"trial": 0, "fold": fold, "status": "complete", "best_validation_average_precision": 1.0} for fold in range(5)]))
            for fold in range(5):
                (trial / f"fold_{fold}_status.json").write_text('{"status":"complete", "best_validation_average_precision":1.0}')
                folder = trial / f"fold_{fold}"; folder.mkdir()
                (folder / "config.json").write_text(json.dumps({**chosen, "manifest_sha256": digest, "fold": fold, "pretrained_weights_sha256": weights_hash}))
                (folder / "epochs.csv").write_text("epoch,validation_average_precision\n1,1.0\n", encoding="utf-8")
                (folder / "best.pt").write_bytes(b"checkpoint")
                with (folder / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["path", "label", "score", "group"]); writer.writeheader()
                    for label in range(2):
                        name = f"{fold}-{label}.png"
                        writer.writerow(dict(path=name, label=label, score=0.2 + label * 0.6, group=name))
            select_candidate(search)
            threshold = root / "threshold.json"
            select(manifest, trial, threshold)
            history = trial / "fold_0" / "epochs.csv"
            original = history.read_bytes()
            history.write_bytes(original + b"2,0.1\n")
            with self.assertRaisesRegex(ValueError, "Epoch history changed"):
                fit(raw, manifest, trial, threshold, root / "tampered", _synthetic_test_only=True)
            history.write_bytes(original)
            search_file = search / "search.json"
            saved_search = search_file.read_bytes()
            changed = json.loads(saved_search)
            changed["training_source_sha256"]["models.py"] = "changed"
            search_file.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Training source differs"):
                fit(raw, manifest, trial, threshold, root / "wrong-source", _synthetic_test_only=True)
            search_file.write_bytes(saved_search)
            with patch("glenda.fit_final.pretrained_url", return_value="https://example.org/fake.pth"), \
                 patch("glenda.fit_final.build_model", side_effect=lambda *args, **kwargs: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 224 * 224, 1))), \
                 contextlib.redirect_stdout(io.StringIO()):
                fit(raw, manifest, trial, threshold, root / "final", _synthetic_test_only=True)
            metadata = json.loads((root / "final/config.json").read_text())
            self.assertEqual(metadata["epochs"], 1)
            self.assertEqual(metadata["threshold"], 0.8)
            self.assertTrue((root / "final/final.pt").exists())


if __name__ == "__main__":
    unittest.main()
