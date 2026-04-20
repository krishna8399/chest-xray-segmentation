"""Download Montgomery County and Shenzhen chest X-ray datasets.

Usage: python scripts/download_data.py
"""

import subprocess
import sys
from pathlib import Path


def download_datasets():
    data_dir = Path("data/chest_xray")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Downloading chest X-ray datasets...")
    print()

    kaggle_available = False
    try:
        import kaggle
        kaggle_available = True
    except (ImportError, OSError):
        pass

    if kaggle_available:
        print("📦 Kaggle API detected. Downloading...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "nikhilpandey360/chest-xray-masks-and-labels",
            "-p", str(data_dir), "--unzip",
        ], check=True)
        print("✅ Download complete!")
    else:
        print("⚠️  Kaggle API not configured.")
        print("   Option 1: Set up Kaggle API (kaggle.com/settings → Create Token)")
        print("   Option 2: Download manually from:")
        print("     https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels")
        print(f"   Extract to: {data_dir}")

    print(f"\n📁 Data directory: {data_dir}")


if __name__ == "__main__":
    download_datasets()
