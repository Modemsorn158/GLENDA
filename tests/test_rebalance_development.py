import csv
import json
import tempfile
import unittest
from pathlib import Path

from glenda.rebalance_development import rebalance


class RebalanceTests(unittest.TestCase):
    def test_holdout_unchanged_groups_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root / "manifest.csv"
            rows = []
            for label in (0, 1):
                for i in range(15):
                    rows.append(dict(path=f"{label}-{i}", label=label, group=f"{label}-{i}", partition="development", cv_fold=i % 5))
            rows.append(dict(path="heldout", label=0, group="heldout", partition="test", cv_fold=""))
            with manifest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            rebalance(manifest, root / "one"); rebalance(manifest, root / "two")
            self.assertEqual((root / "one/manifest.csv").read_bytes(), (root / "two/manifest.csv").read_bytes())
            protocol = json.loads((root / "one/protocol.json").read_text())
            self.assertTrue(protocol["holdout_unchanged"])
            self.assertTrue(all(v["0"] == 3 and v["1"] > 0 for v in protocol["validation_groups_by_label"].values()))
            with (root / "one/manifest.csv").open(encoding="utf-8", newline="") as f:
                output = list(csv.DictReader(f))
            self.assertEqual(next(row for row in output if row["path"] == "heldout")["partition"], "test")
            self.assertEqual(next(row for row in output if row["path"] == "heldout")["cv_fold"], "")


if __name__ == "__main__":
    unittest.main()
