"""Local research-only image demonstration for a selected final checkpoint."""
import argparse
import math
import os
import tempfile
from pathlib import Path

import torch

from glenda.inference import load_final, predict_image

RESEARCH_NOTICE = (
    "Research demonstration only. Not clinically validated and not for diagnosis or treatment decisions. "
    "A negative model result does not rule out pathology."
)


def classify(model, config, image, device):
    if image is None:
        return "Please provide an image.", "", ""
    score = predict_image(model, image, device)
    threshold = float(config["threshold"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        return "Model result unavailable; pathology is not ruled out", "", f"{threshold:.8f}"
    label = "Model result: above threshold" if score >= threshold else "Model result: below threshold; pathology is not ruled out"
    return label, f"{score:.8f}", f"{threshold:.8f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    a = p.parse_args()
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    # A separate directory per session prevents normal cleanup from removing other files.
    with tempfile.TemporaryDirectory(prefix="gradio_uploads_", dir=a.checkpoint.resolve().parent) as temp_dir:
        os.environ["GRADIO_TEMP_DIR"] = temp_dir
        import gradio as gr
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, config = load_final(a.checkpoint, device)
        app = gr.Interface(
            fn=lambda image: classify(model, config, image, device),
            inputs=gr.Image(type="pil", label="Laparoscopic image"),
            outputs=[gr.Textbox(label="Model result"), gr.Textbox(label="Model score (not calibrated probability)"), gr.Textbox(label="Decision threshold")],
            title="GLENDA research demonstration",
            description=RESEARCH_NOTICE,
        )
        app.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
