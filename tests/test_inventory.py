import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from glenda.inventory import audit


class InventoryTests(unittest.TestCase):
    def test_annotation_provenance_and_shared_video_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw"
            positive = raw / "pathology/frames"; positive.mkdir(parents=True)
            negative = raw / "no_pathology/frames/v_9_s_0-2"; negative.mkdir(parents=True)
            name = "c_2_v_(video_9.mp4)_f_1.jpg"
            Image.new("RGB", (12, 12), "red").save(positive / name)
            Image.new("RGB", (12, 12), "blue").save(negative / "f_1.jpg")
            coco = {"images": [{"id": 1, "file_name": name, "width": 12, "height": 12}],
                    "categories": [{"id": 1, "name": "lesion"}], "annotations": [{"image_id": 1, "category_id": 1}]}
            (raw / "pathology/coco.json").write_text(json.dumps(coco), encoding="utf-8")
            result = audit(raw, root / "audit")
            self.assertEqual((result["positive"], result["negative"]), (1, 1))
            with (root / "audit/inventory.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({row["case_id"] for row in rows}, {"2"})
            self.assertEqual({row["label_source"] for row in rows}, {"coco_annotation", "release_no_pathology_sequence"})
            Image.new("RGB", (12, 12), "green").save(negative / "f_2.jpg")
            with self.assertRaises(ValueError):
                audit(raw, root / "audit2")


if __name__ == "__main__":
    unittest.main()
