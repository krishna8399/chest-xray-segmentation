"""Full evaluation with per-image metrics and visualizations.

Usage:
    python src/evaluation/evaluate.py --model unet --checkpoint models/checkpoints/best_unet.pt
    python src/evaluation/evaluate.py --model deeplabv3 --checkpoint models/checkpoints/best_deeplabv3.pt
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import create_dataloaders
from src.evaluation.metrics import compute_batch_metrics
from src.models.deeplabv3 import DeepLabV3Segmenter
from src.models.unet import UNet


def load_model(model_name: str, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]["model"]

    if model_name == "unet":
        model = UNet(
            in_channels=config.get("in_channels", 1),
            out_channels=config.get("out_channels", 1),
            features=config.get("features", [64, 128, 256, 512]),
        )
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = DeepLabV3Segmenter(pretrained=False, out_channels=config.get("out_channels", 1))
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.to(device).eval()
    print(f"Loaded {model_name} from epoch {ckpt['epoch']} (val Dice {ckpt['val_dice']:.4f})")
    return model


def evaluate(model, test_loader, device):
    all_metrics = {"dice": [], "iou": [], "sensitivity": [], "specificity": [], "pixel_accuracy": []}
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            m = compute_batch_metrics(outputs, masks)
            for k in all_metrics:
                all_metrics[k].append(m[k])

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


def save_visualizations(model, test_loader, device, out_dir: Path, n_samples: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    with torch.no_grad():
        for images, masks in test_loader:
            if saved >= n_samples:
                break
            images, masks = images.to(device), masks.to(device)
            outputs = torch.sigmoid(model(images))

            for i in range(min(images.size(0), n_samples - saved)):
                img = images[i, 0].cpu().numpy()
                gt = masks[i, 0].cpu().numpy()
                pred = outputs[i, 0].cpu().numpy()
                pred_bin = (pred > 0.5).astype(np.float32)

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(img, cmap="gray"); axes[0].set_title("Input")
                axes[1].imshow(gt, cmap="gray"); axes[1].set_title("Ground Truth")
                axes[2].imshow(pred_bin, cmap="gray"); axes[2].set_title("Prediction")

                overlay = np.stack([img, img, img], axis=-1)
                overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
                overlay[..., 0] = np.clip(overlay[..., 0] + pred_bin * 0.4, 0, 1)
                axes[3].imshow(overlay); axes[3].set_title("Overlay")

                for ax in axes:
                    ax.axis("off")

                fig.tight_layout()
                fig.savefig(out_dir / f"sample_{saved:03d}.png", dpi=100, bbox_inches="tight")
                plt.close(fig)
                saved += 1

    print(f"Saved {saved} visualizations to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model")
    parser.add_argument("--model", choices=["unet", "deeplabv3"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/chest_xray")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--visualize", action="store_true", help="Save prediction visualizations")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of visualizations to save")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader = create_dataloaders(args.data_dir, batch_size=args.batch_size)
    model = load_model(args.model, args.checkpoint, device)

    metrics = evaluate(model, test_loader, device)

    print(f"\nTest Set Results ({args.model})")
    print(f"  Dice Score   : {metrics['dice']:.4f}")
    print(f"  IoU          : {metrics['iou']:.4f}")
    print(f"  Sensitivity  : {metrics['sensitivity']:.4f}")
    print(f"  Specificity  : {metrics['specificity']:.4f}")
    print(f"  Pixel Acc.   : {metrics['pixel_accuracy']:.4f}")

    if args.visualize:
        out_dir = Path("assets") / args.model
        save_visualizations(model, test_loader, device, out_dir, args.n_samples)


if __name__ == "__main__":
    main()
