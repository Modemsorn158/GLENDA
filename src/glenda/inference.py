"""Load a final checkpoint and apply the evaluation preprocessing."""
from pathlib import Path
from collections.abc import Mapping
from numbers import Real
import math

import torch
from PIL import Image

from glenda.dataset import inference_contract, preprocessing
from glenda.models import build_model


def load_final(checkpoint: Path, device: torch.device, *, allow_legacy=False):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("Invalid checkpoint structure")
    config = payload["config"]
    if config.get("architecture") not in {"efficientnet_b0", "resnet50"} or config.get("fine_tuning") not in {"frozen", "partial", "full"}:
        raise ValueError("Invalid checkpoint architecture or fine-tuning mode")
    contract = config.get("inference_contract")
    if contract is None and not allow_legacy:
        raise ValueError("Legacy checkpoint has no inference contract; explicit acknowledgement required")
    if contract is not None and contract != inference_contract():
        raise ValueError("Checkpoint inference contract is incompatible with current preprocessing")
    threshold = config.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold in final checkpoint")
    model = build_model(config["architecture"], config["fine_tuning"], pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, config


def predict_image(model, image: Image.Image, device: torch.device) -> float:
    tensor = preprocessing(False)(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        if not isinstance(logits, torch.Tensor) or logits.numel() != 1 or not torch.isfinite(logits).all():
            raise ValueError("Model must return one finite logit per image")
        return float(torch.sigmoid(logits.flatten())[0].cpu())
