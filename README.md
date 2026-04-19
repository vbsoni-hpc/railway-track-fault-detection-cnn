# Railway Track Fault Detection using Vision Transformer

**EE655: Computer Vision — Course Project, IIT Kanpur**

## Authors
- **Akash Tiwari** (Roll No: 241090403)
- **Vaibhav Soni** (Roll No: 241090419)

## Overview
A hybrid deep learning architecture combining Convolutional Neural Networks (ResNet-18 / EfficientNet-B0) with a custom Vision Transformer (ViT) implemented from scratch in PyTorch. The model performs binary classification of railway track images into **Defective** and **Non-defective** categories. The `cnn_baselines.py` script isolates and trains the CNNs to extract optimal initial weights, which are then integrated downstream into our hybrid CNN+ViT pipeline inside `hybrid.ipynb`.

## Architecture
| Component | Details |
|-----------|---------|
| Input Size | 224 × 224 × 3 (RGB) |
| Patch Size | 16 × 16 |
| Number of Patches | 196 |
| Embedding Dimension | 128 |
| Attention Heads | 4 |
| Transformer Blocks | 6 |
| MLP Hidden Dimension | 256 |
| Total Parameters | 919,170 |

## Dataset
Railway Track Fault Detection dataset with the following split:

| Split | Defective | Non-defective | Total |
|-------|-----------|---------------|-------|
| Train | 150 | 150 | 300 |
| Validation | 31 | 31 | 62 |
| Test | 11 | 11 | 22 |

## Results
- **Best Validation Accuracy:** 80.65% (Epoch 16)
- **Training Accuracy:** ~82% (Epoch 30)

## Data Augmentation
- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation ±0.2)
- Random Affine Translation (±10%)
- ImageNet Normalization

## Requirements
```
torch
torchvision
scikit-learn
numpy
```

## Usage
```bash
# 1. Place the dataset in the 'archive (1)/' directory
# 2. Train the CNN baselines to generate weight checkpoints (.pth files)
python cnn_baselines.py

# 3. Open and run the hybrid.ipynb notebook to execute the Hybrid CNN + ViT model
```

## Project Structure
```text
├── cnn_baselines.py        # Trains CNN (ResNet/EfficientNet) and exports weights
├── hybrid.ipynb            # Hybrid architecture execution (CNN features + ViT)
├── best_resnet18.pth       # Generated ResNet weights (After running baselines)
├── best_efficientnet_b0.pth# Generated EfficientNet weights (After running baselines)
├── archive (1)/            # Dataset directory
│   └── Railway Track fault Detection Updated/
│       ├── Train/
│       │   ├── Defective/
│       │   └── Non defective/
│       ├── Validation/
│       │   ├── Defective/
│       │   └── Non defective/
│       └── Test/
│           ├── Defective/
│           └── Non defective/
├── writeup/                # CVPR-format LaTeX write-up
│   └── main.tex
└── README.md
```

## License
MIT
