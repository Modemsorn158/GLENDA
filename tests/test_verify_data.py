import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from glenda.verify_data import verify


class DataIntegrityTests(unittest.TestCase):
    def test_rejects_unsafe_paths_before_opening_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw"; raw.mkdir()
            inventory = root / "inventory.csv"
            inventory.write_text("path,label,case_id,video_id,sha256\n../escape.jpg,1,,video," + "a" * 64 + "\n")
            manifest = root / "manifest.csv"
            manifest.write_text("path,label,group,partition,cv_fold\n../escape.jpg,1,../escape.jpg,development,0\n")
            candidates = root / "pairs.csv"; candidates.write_text("path_a,path_b\n")
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify(raw, manifest, inventory, hashlib.sha256(inventory.read_bytes()).hexdigest(), "development",
                       candidates, hashlib.sha256(candidates.read_bytes()).hexdigest())

    def test_trusted_grouping_and_selected_image_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw"; raw.mkdir()
            (raw / "a.jpg").write_bytes(b"original")
            (raw / "b.jpg").write_bytes(b"second")
            inventory = root / "inventory.csv"
            rows = [dict(path="a.jpg", label="1", case_id="case", video_id="v1", sha256=hashlib.sha256(b"original").hexdigest()),
                    dict(path="b.jpg", label="0", case_id="", video_id="v2", sha256=hashlib.sha256(b"second").hexdigest())]
            with inventory.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            manifest = root / "manifest.csv"
            entries = [dict(path="a.jpg", label="1", partition="development", cv_fold="0", group="a.jpg"),
                       dict(path="b.jpg", label="0", partition="development", cv_fold="1", group="b.jpg")]
            def write_manifest():
                with manifest.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=entries[0].keys()); writer.writeheader(); writer.writerows(entries)
            write_manifest()
            candidates = root / "pairs.csv"; candidates.write_text("path_a,path_b\n", encoding="utf-8")
            trusted = hashlib.sha256(inventory.read_bytes()).hexdigest()
            pair_hash = hashlib.sha256(candidates.read_bytes()).hexdigest()
            def check():
                return verify(raw, manifest, inventory, trusted, "development", candidates, pair_hash)
            self.assertEqual(check(), 2)
            (raw / "a.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Image hash mismatch"):
                check()
            (raw / "a.jpg").write_bytes(b"original")
            entries[0]["group"] = "forged"
            with self.assertRaisesRegex(ValueError, "canonical"):
                write_manifest(); check()
            entries[0]["group"] = "a.jpg"
            rows[1]["video_id"] = "v1"
            with inventory.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            trusted = hashlib.sha256(inventory.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "canonical|crosses"):
                write_manifest(); check()
            rows[1]["video_id"] = "v2"
            with inventory.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            trusted = hashlib.sha256(inventory.read_bytes()).hexdigest()
            candidates.write_text("path_a,path_b\na.jpg,b.jpg\n", encoding="utf-8")
            pair_hash = hashlib.sha256(candidates.read_bytes()).hexdigest()
            entries[1]["group"] = "a.jpg"
            with self.assertRaisesRegex(ValueError, "crosses"):
                write_manifest(); check()
            with self.assertRaisesRegex(ValueError, "Candidate-pair hash"):
                verify(raw, manifest, inventory, trusted, "development", candidates, "0" * 64)
            candidates.write_text("path_a,path_b\n", encoding="utf-8")
            pair_hash = hashlib.sha256(candidates.read_bytes()).hexdigest()
            for field in ("case_id", "sha256"):
                original = rows[1][field]
                rows[1][field] = rows[0][field]
                with inventory.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
                trusted = hashlib.sha256(inventory.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, "crosses"):
                    write_manifest(); check()
                rows[1][field] = original
            entries.pop()
            with self.assertRaisesRegex(ValueError, "cover"):
                write_manifest(); check()


if __name__ == "__main__":
    unittest.main()
