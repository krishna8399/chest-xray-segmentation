"""Medical image preprocessing: CLAHE, normalization, mask processing."""

import cv2
import numpy as np
from typing import Tuple


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def normalize_image(image):
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def process_mask(mask, threshold=0.5):
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return (mask > threshold).astype(np.float32)


def combine_lung_masks(left_mask, right_mask):
    left = process_mask(left_mask)
    right = process_mask(right_mask)
    return np.clip(left + right, 0, 1)
