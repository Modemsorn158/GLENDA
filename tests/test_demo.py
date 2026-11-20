import unittest
from unittest.mock import patch

from PIL import Image

from glenda.demo import classify


class DemoTests(unittest.TestCase):
    def test_fixed_threshold_and_missing_input(self):
        with patch("glenda.demo.predict_image", return_value=0.7):
            label, score, threshold = classify(None, {"threshold": 0.6}, Image.new("RGB", (4, 4)), None)
        self.assertEqual(label, "Pathology-positive")
        self.assertEqual(score, "0.7000")
        self.assertEqual(threshold, "0.6000")
        self.assertEqual(classify(None, {}, None, None)[0], "Please provide an image.")


if __name__ == "__main__":
    unittest.main()
