"""Local research-only image demonstration for a selected final checkpoint."""
import argparse
import os
from pathlib import Path

import torch

from glenda.inference import load_final, predict_image


def classify(model, config, image, device):
    if image is None:
        return "Please provide an image.", "", ""
    score = predict_image(model, image, device)
    threshold = float(config["threshold"])
    label = "Pathology-positive" if score >= threshold else "No visible pathology predicted"
    return label, f"{score:.8f}", f"{threshold:.8f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    a = p.parse_args()
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_TEMP_DIR"] = str(a.checkpoint.resolve().parent / "gradio_tmp")
    import gradio as gr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_final(a.checkpoint, device)
    app = gr.Interface(
        fn=lambda image: classify(model, config, image, device),
        inputs=gr.Image(type="pil", label="Laparoscopic image"),
        outputs=[gr.Textbox(label="Predicted class"), gr.Textbox(label="Model score (not calibrated probability)"), gr.Textbox(label="Decision threshold")],
        title="GLENDA research demonstration",
        description="Research demonstration only. Not clinically validated and not for diagnosis or treatment decisions.",
    )
    app.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
