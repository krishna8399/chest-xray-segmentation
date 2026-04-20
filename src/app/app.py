"""Gradio web app for chest X-ray lung segmentation.

Usage: python src/app/app.py
"""

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import apply_clahe
from src.models.unet import UNet
from src.models.deeplabv3 import DeepLabV3Segmenter


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    name = config["model"]["name"]

    if name == "unet":
        model = UNet(in_channels=config["model"].get("in_channels", 1),
                     out_channels=config["model"].get("out_channels", 1),
                     features=config["model"].get("features", [64, 128, 256, 512]))
    elif name == "deeplabv3":
        model = DeepLabV3Segmenter(pretrained=False, out_channels=1)
    else:
        raise ValueError(f"Unknown model: {name}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model.to(device)
    return model, config


device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
image_size = 256

for name in ["best_unet.pt", "best_deeplabv3.pt"]:
    ckpt = Path(f"models/checkpoints/{name}")
    if ckpt.exists():
        model, cfg = load_model(str(ckpt), device)
        image_size = cfg["data"]["image_size"]
        print(f"Loaded {name} on {device}")
        break


def predict(input_image):
    if model is None:
        return None, None, "❌ No model loaded. Train a model first."

    if len(input_image.shape) == 3:
        gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = input_image

    enhanced = apply_clahe(gray)
    resized = cv2.resize(enhanced, (image_size, image_size))
    normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()

    binary_mask = (mask > 0.5).astype(np.uint8)

    h, w = gray.shape[:2]
    mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    prob_resized = cv2.resize(mask, (w, h))

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay[mask_resized == 1] = (overlay[mask_resized == 1] * 0.5 +
                                   np.array([0, 120, 255]) * 0.5)

    heatmap = cv2.applyColorMap((prob_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    lung_pct = mask_resized.sum() / mask_resized.size * 100
    stats = f"Lung area: {lung_pct:.1f}% of image"

    return overlay.astype(np.uint8), heatmap, stats


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Chest X-ray"),
    outputs=[
        gr.Image(label="Segmentation overlay"),
        gr.Image(label="Probability heatmap"),
        gr.Textbox(label="Statistics"),
    ],
    title="🫁 Chest X-Ray Lung Segmentation",
    description="Upload a chest X-ray to segment lung regions using U-Net.",
)

if __name__ == "__main__":
    demo.launch(share=True)
