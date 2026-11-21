import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from glenda.verify_data import verify


class DataIntegrityTests(unittest.TestCase):
    def test_checks_trusted_inventory_and_selected_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / "raw"; raw.mkdir()
            (raw / "dev.jpg").write_bytes(b"original")
            inventory = root / "inventory.csv"
            with inventory.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "label", "sha256"]); writer.writeheader()
                writer.writerow(dict(path="dev.jpg", label=1, sha256=hashlib.sha256(b"original").hexdigest()))
                writer.writerow(dict(path="test.jpg", label=0, sha256="unused"))
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "label", "partition"]); writer.writeheader()
                writer.writerow(dict(path="dev.jpg", label=1, partition="development"))
                writer.writerow(dict(path="test.jpg", label=0, partition="test"))
            trusted = hashlib.sha256(inventory.read_bytes()).hexdigest()
            self.assertEqual(verify(raw, manifest, inventory, trusted, "development"), 1)
            (raw / "dev.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Image hash mismatch"):
                verify(raw, manifest, inventory, trusted, "development")
            with self.assertRaisesRegex(ValueError, "Inventory hash"):
                verify(raw, manifest, inventory, "0" * 64, "development")
