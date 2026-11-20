import csv
import tempfile
import unittest
from pathlib import Path

from glenda.make_splits import make


class ManifestTests(unittest.TestCase):
    def test_determinism_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = root / "inventory.csv"
            with inventory.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "label", "case_id", "video_id", "sha256"])
                writer.writeheader()
                for i in range(40):
                    writer.writerow(dict(path=f"{i}.jpg", label=i % 2, case_id=i, video_id=i, sha256=i))
            pairs = root / "pairs.csv"
            pairs.write_text("path_a,path_b\n", encoding="utf-8")
            make(inventory, pairs, root / "one")
            make(inventory, pairs, root / "two")
            self.assertEqual((root / "one/manifest.csv").read_bytes(), (root / "two/manifest.csv").read_bytes())
            with self.assertRaises(FileExistsError):
                make(inventory, pairs, root / "one")


if __name__ == "__main__":
    unittest.main()
