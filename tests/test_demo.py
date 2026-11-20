import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from glenda.demo import classify, main


class DemoTests(unittest.TestCase):
    def test_fixed_threshold_and_missing_input(self):
        with patch("glenda.demo.predict_image", return_value=0.7):
            label, score, threshold = classify(None, {"threshold": 0.6}, Image.new("RGB", (4, 4)), None)
        self.assertEqual(label, "Model result: above threshold")
        self.assertEqual(score, "0.70000000")
        self.assertEqual(threshold, "0.60000000")
        self.assertEqual(classify(None, {}, None, None)[0], "Please provide an image.")

    def test_below_threshold_does_not_offer_reassurance(self):
        with patch("glenda.demo.predict_image", return_value=0.2):
            label, _, _ = classify(None, {"threshold": 0.6}, Image.new("RGB", (4, 4)), None)
        self.assertIn("pathology is not ruled out", label)

    def test_invalid_score_does_not_become_a_negative_result(self):
        with patch("glenda.demo.predict_image", return_value=float("nan")):
            label, score, _ = classify(None, {"threshold": 0.6}, Image.new("RGB", (4, 4)), None)
        self.assertIn("unavailable", label)
        self.assertEqual(score, "")

    def test_upload_directory_is_cleaned_on_return_and_exception(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temp:
                checkpoint = Path(temp) / "model.pt"; checkpoint.touch()
                seen = {}
                def interface(**kwargs):
                    seen["description"] = kwargs["description"]
                    def launch(**launch_args):
                        seen["launch_args"] = launch_args
                        seen["upload_dir"] = Path(os.environ["GRADIO_TEMP_DIR"])
                        (seen["upload_dir"] / "upload.bin").write_bytes(b"temporary")
                        if fail:
                            raise RuntimeError("simulated failure")
                    return SimpleNamespace(launch=launch)
                fake_gradio = SimpleNamespace(Interface=interface, Image=lambda **kw: None, Textbox=lambda **kw: None)
                with patch.dict(sys.modules, {"gradio": fake_gradio}), patch.dict(os.environ, {}, clear=False), \
                     patch.object(sys, "argv", ["demo", str(checkpoint)]), patch("glenda.demo.load_final", return_value=(None, {})), \
                     patch("glenda.demo.torch.cuda.is_available", return_value=False):
                    if fail:
                        with self.assertRaises(RuntimeError):
                            main()
                    else:
                        main()
                self.assertFalse(seen["upload_dir"].exists())
                self.assertEqual(seen["launch_args"], {"server_name": "127.0.0.1", "share": False})
                self.assertIn("does not rule out pathology", seen["description"])


if __name__ == "__main__":
    unittest.main()
