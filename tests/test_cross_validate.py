import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from glenda.cross_validate import run


class CrossValidationTests(unittest.TestCase):
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
                run(raw, manifest, config, root / "runs")
                run(raw, manifest, config, root / "runs")
                self.assertEqual(worker.call_count, 5)
            statuses = json.loads((root / "runs/fold_results.json").read_text())
            self.assertEqual(len(statuses), 5)
            self.assertTrue(all(s["status"] == "complete" for s in statuses))
            config.write_text('[{"epochs": 2}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                run(raw, manifest, config, root / "runs")


if __name__ == "__main__":
    unittest.main()
