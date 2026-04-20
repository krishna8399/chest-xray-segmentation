"""Training loop with MLflow experiment tracking.

Usage:
    python src/models/train.py --config configs/unet.yaml
    python src/models/train.py --config configs/deeplabv3.yaml
"""

import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import create_dataloaders
from src.models.unet import UNet
from src.models.deeplabv3 import DeepLabV3Segmenter
from src.models.losses import get_loss_function
from src.evaluation.metrics import compute_batch_metrics


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(config):
    name = config["model"]["name"]
    if name == "unet":
        return UNet(
            in_channels=config["model"].get("in_channels", 1),
            out_channels=config["model"].get("out_channels", 1),
            features=config["model"].get("features", [64, 128, 256, 512]),
        )
    elif name == "deeplabv3":
        return DeepLabV3Segmenter(
            pretrained=config["model"].get("pretrained", True),
            out_channels=config["model"].get("out_channels", 1),
        )
    raise ValueError(f"Unknown model: {name}")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_dice, all_iou = [], []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        metrics = compute_batch_metrics(outputs, masks)
        all_dice.append(metrics["dice"])
        all_iou.append(metrics["iou"])
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{metrics['dice']:.3f}")

    n = len(dataloader.dataset)
    return {"train_loss": running_loss / n, "train_dice": np.mean(all_dice),
            "train_iou": np.mean(all_iou)}


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_metrics = {"dice": [], "iou": [], "sensitivity": [], "specificity": []}

    for images, masks in tqdm(dataloader, desc="Validating", leave=False):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item() * images.size(0)
        metrics = compute_batch_metrics(outputs, masks)
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    n = len(dataloader.dataset)
    return {
        "val_loss": running_loss / n,
        "val_dice": np.mean(all_metrics["dice"]),
        "val_iou": np.mean(all_metrics["iou"]),
        "val_sensitivity": np.mean(all_metrics["sensitivity"]),
        "val_specificity": np.mean(all_metrics["specificity"]),
    }


def train(config_path):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        mlflow.log_params({
            "model": config["model"]["name"],
            "image_size": config["data"]["image_size"],
            "batch_size": config["data"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "loss": config["training"]["loss"],
            "optimizer": config["training"]["optimizer"],
        })
        for key, val in config["mlflow"].get("tags", {}).items():
            mlflow.set_tag(key, val)

        train_loader, val_loader, _ = create_dataloaders(
            config["data"]["data_dir"], config["data"]["image_size"],
            config["data"]["batch_size"], config["data"]["num_workers"])

        model = create_model(config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📦 Model: {config['model']['name']} ({total_params:,} params)")
        mlflow.log_param("total_params", total_params)

        freeze_epochs = config["model"].get("freeze_backbone_epochs", 0)
        if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
            model.freeze_backbone()

        criterion = get_loss_function(config["training"]["loss"])
        lr = config["training"]["learning_rate"]
        wd = config["training"]["weight_decay"]

        if config["training"]["optimizer"] == "adamw":
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

        scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"],
                                       eta_min=lr * 0.01)

        es = config["training"]["early_stopping"]
        best_dice = 0.0
        patience_counter = 0

        ckpt_dir = Path("models/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, config["training"]["epochs"] + 1):
            print(f"\n{'='*60}\nEpoch {epoch}/{config['training']['epochs']}")

            if epoch == freeze_epochs + 1 and hasattr(model, "unfreeze_backbone"):
                model.unfreeze_backbone()

            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = validate(model, val_loader, criterion, device)
            scheduler.step()

            mlflow.log_metrics({**train_metrics, **val_metrics,
                                "lr": optimizer.param_groups[0]["lr"]}, step=epoch)

            print(f"  Train — Loss: {train_metrics['train_loss']:.4f} | Dice: {train_metrics['train_dice']:.4f}")
            print(f"  Val   — Loss: {val_metrics['val_loss']:.4f} | Dice: {val_metrics['val_dice']:.4f}")

            if val_metrics["val_dice"] > best_dice:
                best_dice = val_metrics["val_dice"]
                patience_counter = 0
                save_path = ckpt_dir / f"best_{config['model']['name']}.pt"
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_dice": best_dice, "config": config}, save_path)
                mlflow.pytorch.log_model(model, "best_model")
                mlflow.log_metric("best_val_dice", best_dice)
                print(f"  💾 Best model saved (Dice: {best_dice:.4f})")
            else:
                patience_counter += 1
                print(f"  ⏳ No improvement ({patience_counter}/{es['patience']})")

            if patience_counter >= es["patience"]:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                break

        print(f"\n✅ Training complete! Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)
