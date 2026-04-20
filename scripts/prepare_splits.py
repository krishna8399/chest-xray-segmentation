"""Organize the Kaggle chest X-ray dataset into train/val/test splits.

Actual dataset layout (nikhilpandey360/chest-xray-masks-and-labels):

  <data_root>/Lung Segmentation/
    CXR_png/          — all images (CHNCXR_* Shenzhen + MCUCXR_* Montgomery)
    masks/            — CHNCXR masks named <stem>_mask.png
                        MCUCXR masks named <stem>.png  (already combined)
    test/             — 96 Shenzhen images with no ground-truth masks (skipped)

Usage:
    python scripts/prepare_splits.py
    python scripts/prepare_splits.py --data_root data/chest_xray --out_root data/chest_xray
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.preprocessing import process_mask


SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42

# Sub-path inside data_root where the Kaggle zip was extracted
LUNG_SEG_SUBDIR = "Lung Segmentation"


def find_pairs(data_root: Path):
    """Match every image in CXR_png/ to its mask in masks/.

    Shenzhen (CHNCXR_*): mask filename is <stem>_mask.png
    Montgomery (MCUCXR_*): mask filename is <stem>.png
    Images with no mask (Kaggle test set) are silently skipped.
    """
    img_dir = data_root / LUNG_SEG_SUBDIR / "CXR_png"
    mask_dir = data_root / LUNG_SEG_SUBDIR / "masks"

    if not img_dir.exists():
        return []

    pairs = []
    skipped = 0
    for img_path in sorted(img_dir.glob("*.png")):
        stem = img_path.stem
        if stem.startswith("CHNCXR"):
            mask_path = mask_dir / f"{stem}_mask.png"
        else:  # MCUCXR — Montgomery, mask has same stem
            mask_path = mask_dir / f"{stem}.png"

        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            skipped += 1

    if skipped:
        print(f"  Skipped {skipped} images with no ground-truth mask (Kaggle test set)")
    return pairs


def split_indices(n: int, seed: int):
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    train_end = int(n * SPLITS["train"])
    val_end = train_end + int(n * SPLITS["val"])
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def make_split_dirs(out_root: Path):
    for split in SPLITS:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "masks").mkdir(parents=True, exist_ok=True)


def copy_and_binarize_mask(src: Path, dst: Path):
    """Read mask, binarize to {0,255} uint8, write to dst."""
    raw = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    binary = process_mask(raw)                          # float32 in {0.0, 1.0}
    cv2.imwrite(str(dst), (binary * 255).astype(np.uint8))


def run(data_root: Path, out_root: Path, seed: int):
    pairs = find_pairs(data_root)
    if not pairs:
        print(f"No image-mask pairs found under: {data_root / LUNG_SEG_SUBDIR}")
        print("Expected sub-folders: CXR_png/  masks/")
        sys.exit(1)

    print(f"Found {len(pairs)} image-mask pairs")
    idx_map = split_indices(len(pairs), seed)
    make_split_dirs(out_root)

    for split, indices in idx_map.items():
        img_out = out_root / split / "images"
        msk_out = out_root / split / "masks"
        for i in indices:
            img_path, mask_path = pairs[i]
            stem = img_path.stem
            shutil.copy2(img_path, img_out / f"{stem}.png")
            copy_and_binarize_mask(mask_path, msk_out / f"{stem}.png")
        print(f"  {split:5s}: {len(indices)} images")


def verify(out_root: Path):
    print("\nVerification:")
    all_ok = True
    for split in SPLITS:
        img_stems = {p.stem for p in (out_root / split / "images").glob("*.png")}
        msk_stems = {p.stem for p in (out_root / split / "masks").glob("*.png")}
        unmatched = img_stems.symmetric_difference(msk_stems)
        status = "OK" if not unmatched else f"MISMATCH — {len(unmatched)} unmatched"
        print(f"  {split:5s}: {len(img_stems)} images, {len(msk_stems)} masks — {status}")
        if unmatched:
            all_ok = False
            for name in sorted(unmatched)[:5]:
                in_imgs = name in img_stems
                print(f"    {name}  (image={in_imgs}, mask={not in_imgs})")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Prepare chest X-ray dataset splits")
    parser.add_argument(
        "--data_root", type=Path,
        default=Path("data/chest_xray"),
        help="Folder containing the extracted Kaggle dataset (default: data/chest_xray)",
    )
    parser.add_argument(
        "--out_root", type=Path,
        default=Path("data/chest_xray"),
        help="Output root for train/val/test folders (default: data/chest_xray)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()

    if not data_root.exists():
        print(f"Error: --data_root '{data_root}' does not exist.")
        sys.exit(1)

    print(f"Source : {data_root}")
    print(f"Output : {out_root}")
    print(f"Splits : train={SPLITS['train']:.0%}  val={SPLITS['val']:.0%}  test={SPLITS['test']:.0%}")
    print()

    run(data_root, out_root, args.seed)
    ok = verify(out_root)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
