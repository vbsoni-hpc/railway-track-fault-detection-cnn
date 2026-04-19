# Railway Track Fault Detection

## Overview
A deep learning project utilizing custom convolutional neural network (CNN) architectures, specifically ResNet-18 and EfficientNet-B0, for binary classification of railway track images into Defective and Non-defective categories.

## Dataset
Railway Track Fault Detection dataset with a structured split for training, validation, and testing.

## Results
- Evaluates multiple model architectures (ResNet-18, EfficientNet-B0)
- Explores Ensemble strategies for improved performance (Soft Voting, Weighted Voting, Stacking)
- Automatically generated evaluation metrics including accuracy, precision, recall, F1 score, and AUC-ROC.
- Checkpoints saved for individual models during the training process.

## Data Augmentation
- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation ±0.2)
- Random Affine Translation (±10%)
- ImageNet Normalization

## Requirements
```
torch torchvision timm scikit-learn matplotlib seaborn numpy
```

## Usage
```
# Place the dataset in archive/ directory
python resnet_efficientnet_training.py
```

## Project Structure
```
├── resnet_efficientnet_training.py # Main training & evaluation script
├── railway_ensemble.py # Ensemble methodology script
├── best_resnet18.pth # Model checkpoint (generated after training)
├── best_efficientnet_b0.pth # Model checkpoint (generated after training)
├── archive/ # Dataset directory
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
├── writeup/ # CVPR-format LaTeX write-up
│   └── main.tex
└── README.md
```

## License
MIT

## About
Railway Track Fault Detection using ResNet-18 and EfficientNet-B0 | Course Project.
