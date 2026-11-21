import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from glenda.cross_validate import run


class CrossValidationTests(unittest.TestCase):
    def test_legacy_search_cannot_silently_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / "raw"; raw.mkdir()
            manifest = root / "manifest.csv"; manifest.write_text("manifest", encoding="utf-8")
            config = root / "config.json"; config.write_text('[{"epochs": 1}]', encoding="utf-8")
            output = root / "runs"; output.mkdir()
            (output / "search.json").write_text(json.dumps({"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "configurations": [{"epochs": 1}]}))
            with self.assertRaisesRegex(ValueError, "Legacy search"):
                run(raw, manifest, config, output, _synthetic_test_only=True)

    def test_rejects_unverified_library_call(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "Trusted inventory"):
                run(root / "raw", root / "manifest", root / "config", root / "output")

    def test_complete_folds_resume_without_retraining(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"; raw.mkdir()
            manifest = root / "manifest.csv"; manifest.write_text("manifest", encoding="utf-8")
            config = root / "config.json"; config.write_text('[{"epochs": 1}]', encoding="utf-8")
            def fake_run(command, **kwargs):
                destination = Path(command[5]); destination.mkdir()
                (destination / "best.pt").write_bytes(b"checkpoint")
                (destination / "epochs.csv").write_text("validation_average_precision\n0.25\n", encoding="utf-8")
                (destination / "validation_predictions.csv").write_text("path,label,score,group\n", encoding="utf-8")
                return SimpleNamespace(returncode=0)
            with patch("glenda.cross_validate.subprocess.run", side_effect=fake_run) as worker, contextlib.redirect_stdout(io.StringIO()):
                run(raw, manifest, config, root / "runs", _synthetic_test_only=True)
                run(raw, manifest, config, root / "runs", _synthetic_test_only=True)
                self.assertEqual(worker.call_count, 5)
            statuses = json.loads((root / "runs/fold_results.json").read_text())
            self.assertEqual(len(statuses), 5)
            self.assertTrue(all(s["status"] == "complete" for s in statuses))
            config.write_text('[{"epochs": 2}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                run(raw, manifest, config, root / "runs", _synthetic_test_only=True)


if __name__ == "__main__":
    unittest.main()
