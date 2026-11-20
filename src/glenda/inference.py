"""Load a final checkpoint and apply the evaluation preprocessing."""
from pathlib import Path

import torch
from PIL import Image

from glenda.dataset import preprocessing
from glenda.models import build_model


def load_final(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    config = payload["config"]
    if not 0 <= float(config["threshold"]) <= 1:
        raise ValueError("Invalid threshold in final checkpoint")
    model = build_model(config["architecture"], config["fine_tuning"], pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, config


def predict_image(model, image: Image.Image, device: torch.device) -> float:
    tensor = preprocessing(False)(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(torch.sigmoid(model(tensor).flatten())[0].cpu())
