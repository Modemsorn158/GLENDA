"""Check the technical guide's study commands through their actual parsers."""
import importlib
import re
import shlex
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class ReadmeCommandTests(unittest.TestCase):
    def test_documented_commands_supply_inventory_arguments(self):
        readme = (Path(__file__).resolve().parents[1] / "docs" / "technical-guide.md").read_text(encoding="utf-8")
        commands = {"train_fold": "train", "cross_validate": "run", "fit_final": "fit", "evaluate_test": "evaluate"}
        for module_name, entry in commands.items():
            with self.subTest(module=module_name):
                examples = re.findall(r"`(python -m glenda\." + module_name + r" [^`]+)`", readme)
                self.assertEqual(len(examples), 1, "Expected one documented command")
                tokens = shlex.split(examples[0], posix=False)
                tokens = ["a" * 64 if token == "$TrustedInventorySha256" else "b" * 64 if token == "$TrustedCandidatesSha256" else token for token in tokens]
                self.assertEqual(tokens[:3], ["python", "-m", f"glenda.{module_name}"])
                self.assertIn("--inventory", tokens)
                self.assertIn("--inventory-sha256", tokens)
                self.assertIn("--candidates", tokens)
                self.assertIn("--candidates-sha256", tokens)
                module = importlib.import_module(f"glenda.{module_name}")
                with patch.object(sys, "argv", [module_name, *tokens[3:]]), \
                     patch.object(module, entry) as operation, \
                     patch("glenda.verify_data.verify") as verify:
                    module.main()
                operation.assert_called_once()
                if module_name == "train_fold":
                    args = operation.call_args.args[0]
                    self.assertEqual(args.inventory_sha256, "a" * 64)
                    self.assertEqual(args.candidates_sha256, "b" * 64)
                    self.assertIsNotNone(args.inventory)
                else:
                    self.assertEqual(operation.call_args.args[-1], "b" * 64)
                    self.assertEqual(operation.call_args.args[-3], "a" * 64)
                    self.assertIsInstance(operation.call_args.args[-2], Path)


if __name__ == "__main__":
    unittest.main()
