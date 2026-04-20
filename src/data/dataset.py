"""PyTorch Dataset for chest X-ray segmentation with medical augmentations."""

import os
from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import apply_clahe, process_mask


def get_train_transforms(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.1, scale=(0.85, 1.15), rotate=(-15, 15),
                 mode=cv2.BORDER_CONSTANT, p=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.GaussNoise(std_range=(0.02, 0.11), p=0.2),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])


class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, use_clahe=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.use_clahe = use_clahe

        image_files = sorted(os.listdir(self.image_dir))
        mask_files = sorted(os.listdir(self.mask_dir))

        image_stems = {Path(f).stem: f for f in image_files}
        mask_stems = {Path(f).stem: f for f in mask_files}
        common = sorted(set(image_stems.keys()) & set(mask_stems.keys()))

        self.image_paths = [self.image_dir / image_stems[s] for s in common]
        self.mask_paths = [self.mask_dir / mask_stems[s] for s in common]
        print(f"📁 Loaded {len(self.image_paths)} image-mask pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if self.use_clahe:
            image = apply_clahe(image)

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = process_mask(mask)

        image = np.expand_dims(image, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(0).float()
        elif isinstance(mask, np.ndarray) and mask.ndim == 2:
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


def create_dataloaders(data_dir, image_size=256, batch_size=8, num_workers=4):
    if os.name == "nt":
        num_workers = 0
    pin = torch.cuda.is_available()

    train_ds = ChestXrayDataset(
        f"{data_dir}/train/images", f"{data_dir}/train/masks",
        transform=get_train_transforms(image_size))
    val_ds = ChestXrayDataset(
        f"{data_dir}/val/images", f"{data_dir}/val/masks",
        transform=get_val_transforms(image_size))
    test_ds = ChestXrayDataset(
        f"{data_dir}/test/images", f"{data_dir}/test/masks",
        transform=get_val_transforms(image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader
