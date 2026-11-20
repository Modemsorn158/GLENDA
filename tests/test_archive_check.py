import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from glenda.archive_check import compare


class ArchiveTests(unittest.TestCase):
    def test_frame_and_annotation_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw = root / "raw"; (raw / "pathology").mkdir(parents=True)
            (raw / "pathology/coco.json").write_bytes(b"annotation")
            positive, negative = root / "positive.zip", root / "negative.zip"
            with ZipFile(positive, "w") as z:
                z.writestr("Glenda_v1.5_classes/coco.json", b"annotation")
                z.writestr("Glenda_v1.5_classes/frames/a.jpg", b"positive")
            with ZipFile(negative, "w") as z:
                z.writestr("no_pathology/frames/v_1/f.jpg", b"negative")
            inventory = root / "inventory.csv"
            with inventory.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "sha256"]); writer.writeheader()
                for path, content in (("pathology/frames/a.jpg", b"positive"), ("no_pathology/frames/v_1/f.jpg", b"negative")):
                    writer.writerow({"path": path, "sha256": hashlib.sha256(content).hexdigest()})
            result = compare(inventory, positive, negative, raw, root / "result.json")
            self.assertEqual(result["archive_frames"], 2)
            self.assertTrue(result["coco_byte_match"])
            (raw / "pathology/coco.json").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                compare(inventory, positive, negative, raw, root / "mismatch.json")


if __name__ == "__main__":
    unittest.main()
