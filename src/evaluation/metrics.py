"""Medical image segmentation metrics: Dice, IoU, sensitivity, specificity."""

import numpy as np
import torch
from typing import Dict


def dice_score(prediction, target, smooth=1e-6):
    pred = prediction.flatten()
    tgt = target.flatten()
    intersection = (pred * tgt).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + tgt.sum() + smooth))


def iou_score(prediction, target, smooth=1e-6):
    pred = prediction.flatten()
    tgt = target.flatten()
    intersection = (pred * tgt).sum()
    union = pred.sum() + tgt.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def sensitivity(prediction, target, smooth=1e-6):
    pred = prediction.flatten()
    tgt = target.flatten()
    tp = (pred * tgt).sum()
    fn = (tgt * (1 - pred)).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def specificity(prediction, target, smooth=1e-6):
    pred = prediction.flatten()
    tgt = target.flatten()
    tn = ((1 - pred) * (1 - tgt)).sum()
    fp = (pred * (1 - tgt)).sum()
    return float((tn + smooth) / (tn + fp + smooth))


def compute_all_metrics(prediction, target, threshold=0.5):
    pred_binary = (prediction > threshold).astype(np.float32)
    return {
        "dice": dice_score(pred_binary, target),
        "iou": iou_score(pred_binary, target),
        "sensitivity": sensitivity(pred_binary, target),
        "specificity": specificity(pred_binary, target),
        "pixel_accuracy": float((pred_binary == target).mean()),
    }


@torch.no_grad()
def compute_batch_metrics(predictions, targets, threshold=0.5):
    preds = torch.sigmoid(predictions).cpu().numpy()
    tgts = targets.cpu().numpy()

    batch_metrics = {"dice": [], "iou": [], "sensitivity": [],
                     "specificity": [], "pixel_accuracy": []}

    for pred, tgt in zip(preds, tgts):
        metrics = compute_all_metrics(pred.squeeze(), tgt.squeeze(), threshold)
        for key, value in metrics.items():
            batch_metrics[key].append(value)

    return {k: float(np.mean(v)) for k, v in batch_metrics.items()}
