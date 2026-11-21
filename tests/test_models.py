import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from glenda.dataset import DevelopmentDataset, inference_contract
from glenda.models import build_model, set_training_mode
from glenda.train_fold import train
from glenda.inference import load_final, predict_image
from PIL import Image


def changed_preprocessing(training):
    return None


class ModelTests(unittest.TestCase):
    def test_synthetic_golden_preprocessing_and_logit(self):
        class Probe(torch.nn.Module):
            def forward(self, x):
                return (0.2 * x[:, 0, 0, 0] + 0.3 * x[:, 1, 112, 112] - 0.1 * x[:, 2, 223, 223] + 0.4).unsqueeze(1)
        image = Image.new("RGB", (2, 2))
        image.putdata([(0, 40, 80), (255, 100, 20), (30, 200, 60), (220, 250, 240)])
        self.assertAlmostEqual(predict_image(Probe(), image, torch.device("cpu")), 0.47630763, delta=0.0005)
        with self.assertRaisesRegex(ValueError, "one finite logit"):
            predict_image(torch.nn.Identity(), image, torch.device("cpu"))
        class Invalid(torch.nn.Module):
            def forward(self, x):
                return torch.tensor([float("nan")], device=x.device)
        with self.assertRaisesRegex(ValueError, "one finite logit"):
            predict_image(Invalid(), image, torch.device("cpu"))

    def test_frozen_efficientnet_forward_and_batchnorm(self):
        model = build_model("efficientnet_b0", "frozen", pretrained=False)
        set_training_mode(model)
        self.assertTrue(all(not p.requires_grad for p in model.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        self.assertTrue(all(not m.training for m in model.features.modules() if isinstance(m, torch.nn.BatchNorm2d)))
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(2, 3, 64, 64)).shape), (2, 1))

    def test_manifest_group_crossing_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "label", "group", "partition", "cv_fold"])
                writer.writeheader()
                writer.writerow(dict(path="a", label=0, group="same", partition="development", cv_fold=0))
                writer.writerow(dict(path="b", label=1, group="same", partition="test", cv_fold=""))
            with self.assertRaises(ValueError):
                DevelopmentDataset(Path(temp), path, 0, True)

    def test_partial_finetuning_leaves_earlier_features_frozen(self):
        model = build_model("efficientnet_b0", "partial", pretrained=False)
        self.assertTrue(all(not p.requires_grad for p in model.features[:-3].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.features[-3:].parameters()))
        set_training_mode(model)
        self.assertTrue(all(not block.training for block in model.features[:-3]))
        self.assertTrue(all(block.training for block in model.features[-3:]))

    def test_frozen_features_are_deterministic_in_training_mode(self):
        model = build_model("efficientnet_b0", "frozen", pretrained=False)
        set_training_mode(model)
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            a, b = model.features(x), model.features(x)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(model.classifier.training)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_synthetic_cpu_gpu_inference_scores_match(self):
        torch.manual_seed(17)
        cpu_model = build_model("efficientnet_b0", "frozen", pretrained=False).eval()
        gpu_model = build_model("efficientnet_b0", "frozen", pretrained=False).eval()
        gpu_model.load_state_dict(cpu_model.state_dict())
        gpu_model.to("cuda")
        image = Image.new("RGB", (31, 19), (40, 120, 210))
        cpu = predict_image(cpu_model, image, torch.device("cpu"))
        gpu = predict_image(gpu_model, image, torch.device("cuda"))
        self.assertAlmostEqual(cpu, gpu, delta=1e-5)

    def test_training_cannot_write_into_raw_data(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                train(SimpleNamespace(raw=Path(temp), output=Path(temp) / "output"))

    def test_final_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            model = build_model("efficientnet_b0", "frozen", pretrained=False)
            path = Path(temp) / "final.pt"
            config = {"architecture": "efficientnet_b0", "fine_tuning": "frozen", "threshold": 0.5, "inference_contract": inference_contract()}
            torch.save({"state_dict": model.state_dict(), "config": config}, path)
            loaded, config = load_final(path, torch.device("cpu"))
            score = predict_image(loaded, Image.new("RGB", (24, 24), "gray"), torch.device("cpu"))
            self.assertTrue(0 <= score <= 1)
            self.assertEqual(config["threshold"], 0.5)
            with patch("glenda.dataset.preprocessing", changed_preprocessing):
                with self.assertRaisesRegex(ValueError, "inference contract"):
                    load_final(path, torch.device("cpu"))
            config["inference_contract"]["color"] = "BGR"
            torch.save({"state_dict": model.state_dict(), "config": config}, path)
            with self.assertRaisesRegex(ValueError, "inference contract"):
                load_final(path, torch.device("cpu"))
            config.pop("inference_contract")
            torch.save({"state_dict": model.state_dict(), "config": config}, path)
            with self.assertRaisesRegex(ValueError, "Legacy checkpoint"):
                load_final(path, torch.device("cpu"))
            load_final(path, torch.device("cpu"), allow_legacy=True)
            config["threshold"] = float("nan")
            torch.save({"state_dict": model.state_dict(), "config": config}, path)
            with self.assertRaisesRegex(ValueError, "threshold"):
                load_final(path, torch.device("cpu"), allow_legacy=True)


if __name__ == "__main__":
    unittest.main()
