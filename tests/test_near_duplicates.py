import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from glenda.near_duplicates import screen


class ScreenTests(unittest.TestCase):
    def test_cross_component_pair_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGB", (20, 20), "red").save(root / "a.png")
            Image.new("RGB", (20, 20), "blue").save(root / "b.png")
            inventory = root / "inventory.csv"
            with inventory.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "label", "case_id", "video_id", "sha256"])
                writer.writeheader()
                writer.writerow(dict(path="a.png", label="1", case_id="1", video_id="1", sha256="a"))
                writer.writerow(dict(path="b.png", label="0", case_id="2", video_id="2", sha256="b"))
            result = root / "pairs.csv"
            self.assertEqual(screen(root, inventory, result), 1)
            with result.open(encoding="utf-8", newline="") as f:
                self.assertEqual(next(csv.DictReader(f))["hamming_distance"], "0")


if __name__ == "__main__":
    unittest.main()
