import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from glenda.artifact_manifest import build_manifest, verify_local


class ArtifactManifestTests(unittest.TestCase):
    def test_summary_consistency_and_tampering(self):
        source = Path(__file__).resolve().parents[1] / "reports"
        files = ("archive_check.json", "split_v2_summary.json", "tuning_results.json",
                 "oof_score_diagnostics.json", "final_fit.json", "error_analysis.json", "test_results.json")
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp)
            for name in files:
                shutil.copyfile(source / name, reports / name)
            manifest = build_manifest(reports)
            self.assertEqual(manifest, json.loads((source / "artifact_manifest.json").read_text(encoding="utf-8")))
            final_path = reports / "final_fit.json"
            final = json.loads(final_path.read_text(encoding="utf-8"))
            final["manifest_sha256"] = "wrong"
            final_path.write_text(json.dumps(final), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest hashes disagree"):
                build_manifest(reports)

    def test_local_hash_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "outputs" / "audit" / "inventory.csv"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"changed")
            digest = hashlib.sha256(b"expected").hexdigest()
            manifest = dict.fromkeys(("inventory_sha256", "parent_manifest_sha256", "manifest_sha256",
                "search_sha256", "fold_results_sha256", "threshold_file_sha256", "checkpoint_sha256",
                "test_predictions_sha256"), digest)
            manifest["fold_prediction_sha256"] = []
            with self.assertRaisesRegex(ValueError, "Hash mismatch: audit/inventory.csv"):
                verify_local(root, manifest)


if __name__ == "__main__":
    unittest.main()
