# 🫁 Chest X-Ray Lung Segmentation Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-blue.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A medical image segmentation pipeline for chest X-rays that segments lung regions using U-Net (built from scratch) and DeepLabV3+. Includes rigorous experiment tracking with MLflow, medical-specific evaluation metrics (Dice, IoU, sensitivity, specificity), and a Gradio demo app deployed on HuggingFace Spaces.

## 🏗️ Architecture

```
Chest X-ray Input (grayscale)
       │
       ▼
┌────────────────────┐
│  Preprocessing     │  CLAHE enhancement, resize to 256×256
│  + Augmentation    │  Elastic deform, rotation, flip
└────────┬───────────┘
         │
    ┌────▼────┐          ┌──────────────┐
    │  U-Net  │          │ DeepLabV3+   │
    │ (scratch)│          │ (pretrained) │
    └────┬────┘          └──────┬───────┘
         │                      │
         ▼                      ▼
┌────────────────────────────────────────┐
│  Evaluation Engine                     │
│  Dice · IoU · Sensitivity · Specificity│
└────────────────┬───────────────────────┘
                 │
         ┌───────▼──────────┐
         │  MLflow Tracking  │
         └───────┬──────────┘
                 │
         ┌───────▼──────────┐
         │  Gradio App      │
         └──────────────────┘
```

## 📊 Results

| Model | Dice Score | IoU | Sensitivity | Specificity | Params |
|-------|-----------|-----|-------------|-------------|--------|
| U-Net (scratch) | - | - | - | - | ~7.8M |
| DeepLabV3+ (ResNet-50) | - | - | - | - | ~26M |

## 🚀 Quick Start

```bash
git clone https://github.com/krishna8399/chest-xray-segmentation.git
cd chest-xray-segmentation

conda create -n xray-seg python=3.10 -y
conda activate xray-seg
pip install -r requirements.txt

python scripts/download_data.py
python src/models/train.py --config configs/unet.yaml
python src/models/train.py --config configs/deeplabv3.yaml
python src/app/app.py
```

## 📁 Project Structure

```
chest-xray-segmentation/
├── configs/
│   ├── unet.yaml                # U-Net from scratch config
│   └── deeplabv3.yaml           # DeepLabV3+ transfer learning config
├── src/
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset + medical augmentations
│   │   └── preprocessing.py     # CLAHE, normalization, mask processing
│   ├── models/
│   │   ├── unet.py              # U-Net built from scratch
│   │   ├── deeplabv3.py         # DeepLabV3+ wrapper (torchvision)
│   │   ├── losses.py            # Dice loss, BCE-Dice combo, Focal loss
│   │   └── train.py             # Training loop + MLflow logging
│   ├── evaluation/
│   │   └── metrics.py           # Dice, IoU, sensitivity, specificity
│   └── app/
│       └── app.py               # Gradio web interface
├── scripts/
│   └── download_data.py         # Dataset download
└── tests/
```

## 🔧 Tech Stack

- **Segmentation**: PyTorch, torchvision (DeepLabV3+)
- **Medical Imaging**: albumentations, OpenCV, CLAHE
- **Experiment Tracking**: MLflow
- **Demo**: Gradio (HuggingFace Spaces)
- **Deployment**: Docker

## 📈 Dataset

**Montgomery County + Shenzhen Hospital** chest X-ray datasets:
- 800 total images with pixel-level lung segmentation masks
- Split: 640 train / 80 validation / 80 test

## 🧠 What I Learned

-
-
-

## 📄 License

MIT License

## 👤 Author

**Krishna Singh** — MSc Artificial Intelligence @ IU Berlin
- GitHub: [@krishna8399](https://github.com/krishna8399)
- LinkedIn: [krishna839](https://linkedin.com/in/krishna839)
