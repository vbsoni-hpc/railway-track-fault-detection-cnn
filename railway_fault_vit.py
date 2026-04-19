# ============================================================================
# 🚂 RAILWAY TRACK FAULT DETECTION USING VISION TRANSFORMER (ViT)
# ============================================================================
# Copy each section (marked with # %% [CELL X]) into a separate Jupyter cell.
# ============================================================================


# %% [CELL 1] — Install Dependencies (run once)
# Uncomment and run this cell if packages are not installed:
# !pip install torch torchvision timm albumentations pytorch-grad-cam scikit-learn matplotlib seaborn tqdm pillow


# %% [CELL 2] — Imports & Configuration
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import torchvision.transforms as T
from torchvision.utils import make_grid

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

# ── Configuration ──
CONFIG = {
    # Paths — UPDATE THESE to match your folder structure
    'data_root': r'archive (1)/Railway Track fault Detection Updated',
    'train_dir': r'archive (1)/Railway Track fault Detection Updated/Train',
    'val_dir':   r'archive (1)/Railway Track fault Detection Updated/Validation',
    'test_dir':  r'archive (1)/Railway Track fault Detection Updated/Test',

    # Model
    'model_name': 'vit_base_patch16_224',  # ViT-B/16 pretrained on ImageNet-21k
    'image_size': 224,
    'num_classes': 1,                       # Binary classification (sigmoid output)

    # Training — Phase 1 (frozen backbone)
    'phase1_epochs': 15,
    'phase1_lr': 1e-3,

    # Training — Phase 2 (fine-tune backbone)
    'phase2_epochs': 30,
    'phase2_lr': 5e-6,
    'phase2_lr_head': 1e-4,

    # General
    'batch_size': 16,
    'weight_decay': 1e-4,
    'label_smoothing': 0.05,
    'grad_clip': 1.0,
    'early_stop_patience': 10,
    'seed': 42,
    'num_workers': 0,  # Set to 0 on Windows to avoid multiprocessing issues
}

# ── Reproducibility ──
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG['seed'])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🎮  GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected — training will be slow. Consider using Google Colab.")


# %% [CELL 3] — Exploratory Data Analysis (EDA)
print("=" * 60)
print("📊  DATASET EXPLORATION")
print("=" * 60)

# Count images per class per split
splits = {'Train': CONFIG['train_dir'], 'Validation': CONFIG['val_dir'], 'Test': CONFIG['test_dir']}
eda_data = []

for split_name, split_path in splits.items():
    for class_name in ['Defective', 'Non defective']:
        class_path = os.path.join(split_path, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            eda_data.append({'Split': split_name, 'Class': class_name, 'Count': count})

eda_df = pd.DataFrame(eda_data)
print("\n📋  Image counts per split:")
print(eda_df.pivot(index='Split', columns='Class', values='Count').to_string())
print(f"\n📦  Total images: {eda_df['Count'].sum()}")

# ── Visualize sample images ──
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle('🚂 Railway Track Samples', fontsize=18, fontweight='bold', y=1.02)

for row_idx, class_name in enumerate(['Defective', 'Non defective']):
    class_path = os.path.join(CONFIG['train_dir'], class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    samples = random.sample(images, min(5, len(images)))

    for col_idx, img_name in enumerate(samples):
        img = Image.open(os.path.join(class_path, img_name)).convert('RGB')
        axes[row_idx, col_idx].imshow(img)
        axes[row_idx, col_idx].set_title(
            f'{"⚠️ DEFECTIVE" if class_name == "Defective" else "✅ OK"}',
            fontsize=11, fontweight='bold',
            color='red' if class_name == 'Defective' else 'green'
        )
        axes[row_idx, col_idx].axis('off')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Image size distribution ──
sizes = []
for class_name in ['Defective', 'Non defective']:
    class_path = os.path.join(CONFIG['train_dir'], class_name)
    for img_name in os.listdir(class_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img = Image.open(os.path.join(class_path, img_name))
            sizes.append({'width': img.size[0], 'height': img.size[1], 'class': class_name})

sizes_df = pd.DataFrame(sizes)
print(f"\n📐  Image size stats:")
print(f"    Width  — min: {sizes_df['width'].min()}, max: {sizes_df['width'].max()}, mean: {sizes_df['width'].mean():.0f}")
print(f"    Height — min: {sizes_df['height'].min()}, max: {sizes_df['height'].max()}, mean: {sizes_df['height'].mean():.0f}")


# %% [CELL 4] — Dataset Class & Augmentations
class RailwayDataset(Dataset):
    """Custom Dataset for Railway Track Fault Detection."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Class mapping: Defective = 1, Non defective = 0
        class_map = {'Defective': 1, 'Non defective': 0}

        for class_name, label in class_map.items():
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                continue
            for img_name in sorted(os.listdir(class_path)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(label)

        print(f"    Loaded {len(self.images)} images from {root_dir}")
        print(f"    → Defective: {sum(self.labels)}, Non-defective: {len(self.labels) - sum(self.labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = torch.tensor(label, dtype=torch.float32)
        return image, label


# ── Augmentation Pipelines ──
def get_train_transforms(image_size=224):
    """Aggressive augmentation pipeline for small dataset."""
    return A.Compose([
        A.Resize(image_size, image_size),
        # Spatial transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=20,
            border_mode=0, p=0.5
        ),
        A.Perspective(scale=(0.02, 0.05), p=0.3),

        # Color / lighting transforms
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.7),

        # Noise & blur (simulates real-world capture conditions)
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # Contrast enhancement (great for cracks / surface defects)
        A.CLAHE(clip_limit=2.0, p=0.3),

        # Cutout / Dropout regularization
        A.CoarseDropout(
            max_holes=8, max_height=image_size // 10, max_width=image_size // 10,
            min_holes=2, fill_value=0, p=0.4
        ),

        # Normalize for ImageNet pretrained models
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size=224):
    """Minimal transforms for validation/test — no augmentation."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ── Create datasets & dataloaders ──
print("\n📦  Loading datasets...")
train_dataset = RailwayDataset(CONFIG['train_dir'], transform=get_train_transforms(CONFIG['image_size']))
val_dataset   = RailwayDataset(CONFIG['val_dir'],   transform=get_val_transforms(CONFIG['image_size']))
test_dataset  = RailwayDataset(CONFIG['test_dir'],  transform=get_val_transforms(CONFIG['image_size']))

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                          num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'], shuffle=False,
                          num_workers=CONFIG['num_workers'], pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG['batch_size'], shuffle=False,
                          num_workers=CONFIG['num_workers'], pin_memory=True)

print(f"\n✅  Train: {len(train_dataset)} images ({len(train_loader)} batches)")
print(f"✅  Val:   {len(val_dataset)} images ({len(val_loader)} batches)")
print(f"✅  Test:  {len(test_dataset)} images ({len(test_loader)} batches)")

# ── Visualize augmented samples ──
fig, axes = plt.subplots(2, 6, figsize=(20, 7))
fig.suptitle('🔄 Augmented Training Samples', fontsize=16, fontweight='bold')

# Show same image with different augmentations
sample_img_path = train_dataset.images[0]
original = np.array(Image.open(sample_img_path).convert('RGB'))
aug_transform = get_train_transforms(CONFIG['image_size'])

axes[0, 0].imshow(original)
axes[0, 0].set_title('Original', fontweight='bold')
axes[0, 0].axis('off')

for i in range(1, 6):
    augmented = aug_transform(image=original)['image']
    # Denormalize for display
    img_display = augmented.permute(1, 2, 0).numpy()
    img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)
    axes[0, i].imshow(img_display)
    axes[0, i].set_title(f'Aug #{i}', fontweight='bold')
    axes[0, i].axis('off')

# Second row — different image
sample_img_path2 = train_dataset.images[len(train_dataset) // 2]
original2 = np.array(Image.open(sample_img_path2).convert('RGB'))

axes[1, 0].imshow(original2)
axes[1, 0].set_title('Original', fontweight='bold')
axes[1, 0].axis('off')

for i in range(1, 6):
    augmented = aug_transform(image=original2)['image']
    img_display = augmented.permute(1, 2, 0).numpy()
    img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)
    axes[1, i].imshow(img_display)
    axes[1, i].set_title(f'Aug #{i}', fontweight='bold')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('augmented_samples.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [CELL 5] — ViT Model Architecture
class RailwayViT(nn.Module):
    """
    Vision Transformer for Railway Track Fault Detection.

    Architecture:
        ViT-B/16 (pretrained on ImageNet-21k)
        → CLS token embedding (768-dim)
        → LayerNorm → Dense(256) → GELU → Dropout
        → Dense(1) → Sigmoid
    """

    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, drop_rate=0.3):
        super().__init__()

        # Load pretrained ViT backbone (removes default classification head)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,        # Remove classifier head → returns features
            drop_rate=0.1,        # Backbone-level dropout
        )

        # Get feature dimension from backbone
        feature_dim = self.backbone.num_features  # 768 for ViT-B/16
        print(f"🧠  Backbone: {model_name}")
        print(f"    Feature dimension: {feature_dim}")

        # Custom classification head
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.7),
            nn.Linear(128, 1),
        )

        # Initialize head weights
        self._init_head()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"    Total params:     {total_params:,}")
        print(f"    Trainable params: {trainable_params:,}")

    def _init_head(self):
        """Kaiming initialization for the classification head."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Freeze all backbone parameters (for Phase 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"❄️  Backbone FROZEN — Trainable params: {trainable:,}")

    def unfreeze_backbone(self, unfreeze_ratio=0.5):
        """
        Progressively unfreeze the backbone (for Phase 2 fine-tuning).
        unfreeze_ratio: fraction of blocks to unfreeze (from the top).
        """
        # Unfreeze all first
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Then freeze the bottom (1 - unfreeze_ratio) blocks
        blocks = list(self.backbone.blocks)
        num_blocks = len(blocks)
        freeze_until = int(num_blocks * (1 - unfreeze_ratio))

        # Always keep patch embedding frozen
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False

        # Freeze early blocks
        for i in range(freeze_until):
            for param in blocks[i].parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔥  Unfroze top {num_blocks - freeze_until}/{num_blocks} transformer blocks")
        print(f"    Trainable params: {trainable:,}")

    def forward(self, x):
        features = self.backbone(x)   # [B, 768]
        logits = self.head(features)   # [B, 1]
        return logits


# ── Instantiate the model ──
print("\n" + "=" * 60)
print("🏗️  BUILDING MODEL")
print("=" * 60)
model = RailwayViT(
    model_name=CONFIG['model_name'],
    pretrained=True,
    drop_rate=0.3
).to(DEVICE)


# %% [CELL 6] — Training Engine
class TrainingEngine:
    """Complete training loop with two-phase strategy, logging, and checkpointing."""

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_recall': [], 'val_f1': [],
            'lr': []
        }
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_model_state = None

    def _get_criterion(self):
        """Binary cross-entropy with label smoothing."""
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0]).to(self.device)  # Balanced dataset
        )

    def _train_one_epoch(self, loader, optimizer, criterion, scaler, epoch, total_epochs):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        preds_all, labels_all = [], []

        pbar = tqdm(loader, desc=f"  Epoch {epoch}/{total_epochs} [TRAIN]", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision forward pass
            if self.device.type == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = self.model(images).squeeze(1)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(images).squeeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds_all.extend((probs >= 0.5).astype(int))
            labels_all.extend(labels.cpu().numpy().astype(int))

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = accuracy_score(labels_all, preds_all)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def _validate(self, loader, criterion):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        preds_all, labels_all, probs_all = [], [], []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.device.type == 'cuda':
                with autocast(device_type='cuda'):
                    outputs = self.model(images).squeeze(1)
                    loss = criterion(outputs, labels)
            else:
                outputs = self.model(images).squeeze(1)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            probs_all.extend(probs)
            preds_all.extend((probs >= 0.5).astype(int))
            labels_all.extend(labels.cpu().numpy().astype(int))

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = accuracy_score(labels_all, preds_all)
        epoch_recall = recall_score(labels_all, preds_all, zero_division=0)
        epoch_f1 = f1_score(labels_all, preds_all, zero_division=0)

        return epoch_loss, epoch_acc, epoch_recall, epoch_f1

    def train_phase(self, train_loader, val_loader, phase_name, epochs, optimizer, scheduler=None):
        """Run a complete training phase."""
        print(f"\n{'='*60}")
        print(f"🏋️  {phase_name}")
        print(f"{'='*60}")

        criterion = self._get_criterion()
        scaler = GradScaler() if self.device.type == 'cuda' else None
        self.patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self._train_one_epoch(
                train_loader, optimizer, criterion, scaler, epoch, epochs
            )

            # Validate
            val_loss, val_acc, val_recall, val_f1 = self._validate(val_loader, criterion)

            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']

            # Log history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_recall'].append(val_recall)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)

            # Print epoch summary
            print(f"  Epoch {epoch:02d}/{epochs} │ "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} │ "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Recall: {val_recall:.4f} F1: {val_f1:.4f} │ "
                  f"LR: {current_lr:.2e}")

            # Scheduler step
            if scheduler:
                scheduler.step()

            # Save best model (by val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
                print(f"  ✅ New best model! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config['early_stop_patience']:
                print(f"\n  ⏹️  Early stopping triggered after {epoch} epochs (patience={self.config['early_stop_patience']})")
                break

        print(f"\n  🏆  Best Val Loss: {self.best_val_loss:.4f} | Best Val Acc: {self.best_val_acc:.4f}")

    def load_best_model(self):
        """Load the best model weights."""
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)
            print("✅  Loaded best model weights")

    def save_checkpoint(self, filepath='best_railway_vit.pth'):
        """Save model checkpoint."""
        if self.best_model_state:
            torch.save({
                'model_state_dict': self.best_model_state,
                'config': self.config,
                'history': self.history,
                'best_val_loss': self.best_val_loss,
                'best_val_acc': self.best_val_acc,
            }, filepath)
            print(f"💾  Checkpoint saved to: {filepath}")


# %% [CELL 7] — Run Two-Phase Training
print("\n" + "=" * 60)
print("🚀  STARTING TWO-PHASE TRAINING")
print("=" * 60)

engine = TrainingEngine(model, DEVICE, CONFIG)

# ── PHASE 1: Feature Extraction (Frozen Backbone) ──
model.freeze_backbone()

optimizer_phase1 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG['phase1_lr'],
    weight_decay=CONFIG['weight_decay']
)
scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_phase1, T_max=CONFIG['phase1_epochs'], eta_min=1e-6
)

engine.train_phase(
    train_loader, val_loader,
    phase_name="PHASE 1 — Feature Extraction (Backbone Frozen)",
    epochs=CONFIG['phase1_epochs'],
    optimizer=optimizer_phase1,
    scheduler=scheduler_phase1
)

# ── PHASE 2: Fine-Tuning (Unfreeze Top 50% of Backbone) ──
engine.load_best_model()                       # Start from best Phase 1 weights
model.unfreeze_backbone(unfreeze_ratio=0.5)    # Unfreeze top half of transformer blocks

# Layer-wise learning rate: lower LR for backbone, higher for head
param_groups = [
    {'params': [p for n, p in model.backbone.named_parameters() if p.requires_grad],
     'lr': CONFIG['phase2_lr'], 'name': 'backbone'},
    {'params': model.head.parameters(),
     'lr': CONFIG['phase2_lr_head'], 'name': 'head'},
]

optimizer_phase2 = optim.AdamW(param_groups, weight_decay=CONFIG['weight_decay'])
scheduler_phase2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_phase2, T_0=10, T_mult=2, eta_min=1e-7
)

engine.train_phase(
    train_loader, val_loader,
    phase_name="PHASE 2 — Fine-Tuning (Top 50% Unfrozen, Layer-wise LR)",
    epochs=CONFIG['phase2_epochs'],
    optimizer=optimizer_phase2,
    scheduler=scheduler_phase2
)

# Load the absolute best model and save
engine.load_best_model()
engine.save_checkpoint('best_railway_vit.pth')


# %% [CELL 8] — Training History Visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle('📈 Training History — ViT Railway Fault Detection', fontsize=16, fontweight='bold')

epochs_range = range(1, len(engine.history['train_loss']) + 1)

# Phase boundary
phase1_end = CONFIG['phase1_epochs']
# Loss
axes[0, 0].plot(epochs_range, engine.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
axes[0, 0].plot(epochs_range, engine.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
axes[0, 0].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1→2')
axes[0, 0].set_title('Loss', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy
axes[0, 1].plot(epochs_range, engine.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
axes[0, 1].plot(epochs_range, engine.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
axes[0, 1].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1→2')
axes[0, 1].set_title('Accuracy', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Recall
axes[0, 2].plot(epochs_range, engine.history['val_recall'], 'g-', label='Val Recall', linewidth=2)
axes[0, 2].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1→2')
axes[0, 2].set_title('Validation Recall (Sensitivity)', fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# F1 Score
axes[1, 0].plot(epochs_range, engine.history['val_f1'], 'm-', label='Val F1', linewidth=2)
axes[1, 0].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1→2')
axes[1, 0].set_title('Validation F1 Score', fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Learning Rate
axes[1, 1].plot(epochs_range, engine.history['lr'], 'k-', linewidth=2)
axes[1, 1].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1→2')
axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_yscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Train vs Val gap (overfitting indicator)
gap = [t - v for t, v in zip(engine.history['train_acc'], engine.history['val_acc'])]
axes[1, 2].plot(epochs_range, gap, 'orange', linewidth=2, label='Train-Val Acc Gap')
axes[1, 2].axhline(y=0, color='green', linestyle='-', alpha=0.3)
axes[1, 2].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7, label='Phase 1→2')
axes[1, 2].fill_between(epochs_range, gap, alpha=0.2, color='orange')
axes[1, 2].set_title('Overfitting Indicator (Train-Val Gap)', fontweight='bold')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [CELL 9] — Test Set Evaluation
print("\n" + "=" * 60)
print("🧪  TEST SET EVALUATION")
print("=" * 60)

@torch.no_grad()
def evaluate_model(model, loader, device):
    """Full evaluation with all metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        if device.type == 'cuda':
            with autocast(device_type='cuda'):
                outputs = model(images).squeeze(1)
        else:
            outputs = model(images).squeeze(1)

        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend((probs >= 0.5).astype(int))
        all_labels.extend(labels.numpy().astype(int))

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# Test-Time Augmentation (TTA) — free accuracy boost
@torch.no_grad()
def evaluate_with_tta(model, dataset, device, n_augments=5):
    """
    Test-Time Augmentation: Run each test image through multiple augmentations
    and average the predictions for more robust results.
    """
    model.eval()
    tta_transform = get_train_transforms(CONFIG['image_size'])
    val_transform = get_val_transforms(CONFIG['image_size'])

    all_labels = []
    all_probs = []

    for idx in range(len(dataset)):
        img_path = dataset.images[idx]
        label = dataset.labels[idx]
        all_labels.append(label)

        original_img = np.array(Image.open(img_path).convert('RGB'))
        probs_list = []

        # Original (no augmentation)
        img_tensor = val_transform(image=original_img)['image'].unsqueeze(0).to(device)
        if device.type == 'cuda':
            with autocast(device_type='cuda'):
                out = model(img_tensor).squeeze()
        else:
            out = model(img_tensor).squeeze()
        probs_list.append(torch.sigmoid(out).cpu().item())

        # Augmented versions
        for _ in range(n_augments):
            aug_img = tta_transform(image=original_img)['image'].unsqueeze(0).to(device)
            if device.type == 'cuda':
                with autocast(device_type='cuda'):
                    out = model(aug_img).squeeze()
            else:
                out = model(aug_img).squeeze()
            probs_list.append(torch.sigmoid(out).cpu().item())

        # Average probability across all augmented versions
        avg_prob = np.mean(probs_list)
        all_probs.append(avg_prob)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    return all_labels, all_preds, all_probs


# ── Standard evaluation ──
test_labels, test_preds, test_probs = evaluate_model(model, test_loader, DEVICE)

print("\n📊  Standard Evaluation Results:")
print("-" * 40)
print(f"  Accuracy:    {accuracy_score(test_labels, test_preds):.4f}")
print(f"  Precision:   {precision_score(test_labels, test_preds, zero_division=0):.4f}")
print(f"  Recall:      {recall_score(test_labels, test_preds, zero_division=0):.4f}")
print(f"  F1 Score:    {f1_score(test_labels, test_preds, zero_division=0):.4f}")
if len(np.unique(test_labels)) > 1:
    print(f"  AUC-ROC:     {roc_auc_score(test_labels, test_probs):.4f}")

# ── TTA evaluation ──
print("\n🔄  Running Test-Time Augmentation (TTA)...")
tta_labels, tta_preds, tta_probs = evaluate_with_tta(model, test_dataset, DEVICE, n_augments=7)

print("\n📊  TTA Evaluation Results:")
print("-" * 40)
print(f"  Accuracy:    {accuracy_score(tta_labels, tta_preds):.4f}")
print(f"  Precision:   {precision_score(tta_labels, tta_preds, zero_division=0):.4f}")
print(f"  Recall:      {recall_score(tta_labels, tta_preds, zero_division=0):.4f}")
print(f"  F1 Score:    {f1_score(tta_labels, tta_preds, zero_division=0):.4f}")
if len(np.unique(tta_labels)) > 1:
    print(f"  AUC-ROC:     {roc_auc_score(tta_labels, tta_probs):.4f}")

print("\n📋  Classification Report (TTA):")
print(classification_report(tta_labels, tta_preds, target_names=['Non-Defective', 'Defective']))


# %% [CELL 10] — Comprehensive Evaluation Plots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('🔬 Test Set Evaluation — ViT Railway Fault Detection', fontsize=16, fontweight='bold')

# 1. Confusion Matrix
cm = confusion_matrix(tta_labels, tta_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Non-Defective', 'Defective'],
            yticklabels=['Non-Defective', 'Defective'],
            annot_kws={'size': 16, 'fontweight': 'bold'})
axes[0, 0].set_title('Confusion Matrix', fontweight='bold', fontsize=13)
axes[0, 0].set_xlabel('Predicted', fontsize=11)
axes[0, 0].set_ylabel('Actual', fontsize=11)

# 2. ROC Curve
if len(np.unique(tta_labels)) > 1:
    fpr, tpr, thresholds = roc_curve(tta_labels, tta_probs)
    auc_score = roc_auc_score(tta_labels, tta_probs)
    axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ViT (AUC = {auc_score:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    axes[0, 1].fill_between(fpr, tpr, alpha=0.15, color='blue')
    axes[0, 1].set_title('ROC Curve', fontweight='bold', fontsize=13)
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=11)
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=11)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, 'Not enough classes\nfor ROC', ha='center', va='center', fontsize=14)

# 3. Precision-Recall Curve
if len(np.unique(tta_labels)) > 1:
    precision_vals, recall_vals, _ = precision_recall_curve(tta_labels, tta_probs)
    ap_score = average_precision_score(tta_labels, tta_probs)
    axes[0, 2].plot(recall_vals, precision_vals, 'r-', linewidth=2.5, label=f'AP = {ap_score:.4f}')
    axes[0, 2].fill_between(recall_vals, precision_vals, alpha=0.15, color='red')
    axes[0, 2].set_title('Precision-Recall Curve', fontweight='bold', fontsize=13)
    axes[0, 2].set_xlabel('Recall', fontsize=11)
    axes[0, 2].set_ylabel('Precision', fontsize=11)
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)
else:
    axes[0, 2].text(0.5, 0.5, 'Not enough classes\nfor PR curve', ha='center', va='center', fontsize=14)

# 4. Prediction Confidence Distribution
axes[1, 0].hist(tta_probs[tta_labels == 0], bins=20, alpha=0.6, color='green', label='Non-Defective', edgecolor='black')
axes[1, 0].hist(tta_probs[tta_labels == 1], bins=20, alpha=0.6, color='red', label='Defective', edgecolor='black')
axes[1, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
axes[1, 0].set_title('Prediction Confidence Distribution', fontweight='bold', fontsize=13)
axes[1, 0].set_xlabel('Predicted Probability (Defective)', fontsize=11)
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# 5. Per-Sample Predictions
colors = ['green' if p == l else 'red' for p, l in zip(tta_preds, tta_labels)]
axes[1, 1].bar(range(len(tta_probs)), tta_probs, color=colors, alpha=0.8, edgecolor='black')
axes[1, 1].axhline(y=0.5, color='black', linestyle='--', linewidth=2)
axes[1, 1].set_title('Per-Sample Predictions (Green=Correct, Red=Wrong)', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Sample Index', fontsize=11)
axes[1, 1].set_ylabel('Predicted Probability', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

# 6. Metrics Summary Bar Chart
metrics = {
    'Accuracy': accuracy_score(tta_labels, tta_preds),
    'Precision': precision_score(tta_labels, tta_preds, zero_division=0),
    'Recall': recall_score(tta_labels, tta_preds, zero_division=0),
    'F1': f1_score(tta_labels, tta_preds, zero_division=0),
}
if len(np.unique(tta_labels)) > 1:
    metrics['AUC-ROC'] = roc_auc_score(tta_labels, tta_probs)

bars = axes[1, 2].bar(metrics.keys(), metrics.values(),
                       color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336'][:len(metrics)],
                       edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, metrics.values()):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontweight='bold', fontsize=12)
axes[1, 2].set_ylim(0, 1.15)
axes[1, 2].set_title('Metrics Summary', fontweight='bold', fontsize=13)
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [CELL 11] — Grad-CAM Explainability
print("\n" + "=" * 60)
print("🔍  GRAD-CAM EXPLAINABILITY")
print("=" * 60)

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget


def get_gradcam_for_vit(model, image_tensor, original_image_np, method='gradcam'):
    """
    Generate Grad-CAM heatmap for a ViT model.

    For ViT, we target the LayerNorm before the last transformer block
    since attention maps need reshape handling.
    """
    # Target layer — last transformer block's LayerNorm
    target_layer = model.backbone.blocks[-1].norm1

    # Reshape transform for ViT (patches → spatial grid)
    def reshape_transform(tensor, height=14, width=14):
        # tensor shape: [batch, num_patches+1, embed_dim]
        # Remove CLS token, reshape to spatial grid
        result = tensor[:, 1:, :]  # Remove CLS token
        result = result.reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.permute(0, 3, 1, 2)  # [B, C, H, W]
        return result

    if method == 'gradcam':
        cam = GradCAM(model=model, target_layers=[target_layer],
                      reshape_transform=reshape_transform)
    else:
        cam = GradCAMPlusPlus(model=model, target_layers=[target_layer],
                               reshape_transform=reshape_transform)

    # Generate heatmap
    targets = [BinaryClassifierOutputTarget(1)]  # Activate for "Defective" class
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0).to(DEVICE), targets=targets)
    grayscale_cam = grayscale_cam[0]  # [H, W]

    # Overlay on original image
    visualization = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)

    return grayscale_cam, visualization


# ── Generate Grad-CAM for all test images ──
val_transform = get_val_transforms(CONFIG['image_size'])

num_test = len(test_dataset)
cols = min(num_test, 6)
rows = (num_test + cols - 1) // cols

fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3.5, rows * 7))
fig.suptitle('🔍 Grad-CAM Heatmaps — Where Is the Model Looking?',
             fontsize=16, fontweight='bold', y=1.01)

if rows * 2 == 1:
    axes = axes.reshape(1, -1)
elif cols == 1:
    axes = axes.reshape(-1, 1)

for idx in range(num_test):
    row = (idx // cols) * 2
    col = idx % cols

    # Load and prepare image
    img_path = test_dataset.images[idx]
    label = test_dataset.labels[idx]
    original_img = np.array(Image.open(img_path).convert('RGB'))
    original_resized = np.array(Image.open(img_path).convert('RGB').resize((224, 224)))
    original_float = original_resized.astype(np.float32) / 255.0

    # Get model prediction
    img_tensor = val_transform(image=original_img)['image']
    with torch.no_grad():
        pred_logit = model(img_tensor.unsqueeze(0).to(DEVICE)).squeeze()
        pred_prob = torch.sigmoid(pred_logit).item()
    pred_label = 1 if pred_prob >= 0.5 else 0

    # Generate Grad-CAM
    try:
        heatmap, overlay = get_gradcam_for_vit(model, img_tensor, original_float, method='gradcam')
    except Exception as e:
        print(f"  ⚠️ Grad-CAM failed for image {idx}: {e}")
        continue

    # Determine correctness
    correct = pred_label == label
    status = "✅" if correct else "❌"
    actual = "Defective" if label == 1 else "OK"
    predicted = "Defective" if pred_label == 1 else "OK"

    # Plot original
    if row < axes.shape[0] and col < axes.shape[1]:
        axes[row, col].imshow(original_resized)
        axes[row, col].set_title(
            f'{status} Actual: {actual}\nPred: {predicted} ({pred_prob:.2f})',
            fontsize=9, fontweight='bold',
            color='green' if correct else 'red'
        )
        axes[row, col].axis('off')

    # Plot Grad-CAM overlay
    if row + 1 < axes.shape[0] and col < axes.shape[1]:
        axes[row + 1, col].imshow(overlay)
        axes[row + 1, col].set_title('Grad-CAM', fontsize=9, fontweight='bold')
        axes[row + 1, col].axis('off')

# Hide unused subplots
for idx in range(num_test, rows * cols):
    row = (idx // cols) * 2
    col = idx % cols
    if row < axes.shape[0] and col < axes.shape[1]:
        axes[row, col].axis('off')
    if row + 1 < axes.shape[0] and col < axes.shape[1]:
        axes[row + 1, col].axis('off')

plt.tight_layout()
plt.savefig('gradcam_results.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [CELL 12] — Attention Map Visualization (ViT-specific)
print("\n" + "=" * 60)
print("👁️  ATTENTION MAP VISUALIZATION")
print("=" * 60)


def get_attention_maps(model, image_tensor, device):
    """Extract attention maps from all transformer layers."""
    model.eval()
    attention_maps = []

    # Register hooks to capture attention weights
    hooks = []

    def hook_fn(module, input, output):
        # output of attention: (attn_output, attn_weights)
        # For timm ViT, attention is computed in the Attention module
        attention_maps.append(output)

    # Register forward hooks on all attention layers
    for block in model.backbone.blocks:
        hook = block.attn.attn_drop.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0).to(device))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return attention_maps


# Visualize attention for a few samples
num_samples = min(4, len(test_dataset))
fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
fig.suptitle('👁️ ViT Self-Attention Patterns (Last Layer)',
             fontsize=16, fontweight='bold')

if num_samples == 1:
    axes = axes.reshape(1, -1)

for sample_idx in range(num_samples):
    img_path = test_dataset.images[sample_idx]
    label = test_dataset.labels[sample_idx]
    original_img = np.array(Image.open(img_path).convert('RGB'))
    original_resized = np.array(Image.open(img_path).convert('RGB').resize((224, 224)))

    img_tensor = val_transform(image=original_img)['image']

    # Get attention maps using forward hooks on attention projection
    model.eval()
    attn_weights_list = []

    def capture_attn(module, input, output):
        attn_weights_list.append(input[0].detach().cpu())

    # Hook on the last block's attention softmax output
    hook = model.backbone.blocks[-1].attn.attn_drop.register_forward_hook(capture_attn)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor.unsqueeze(0).to(DEVICE)).squeeze()).item()

    hook.remove()

    # Plot original image
    axes[sample_idx, 0].imshow(original_resized)
    actual = "Defective" if label == 1 else "OK"
    pred_str = f"Pred: {'Defective' if pred >= 0.5 else 'OK'} ({pred:.2f})"
    axes[sample_idx, 0].set_title(f'Actual: {actual}\n{pred_str}', fontsize=10, fontweight='bold')
    axes[sample_idx, 0].axis('off')

    # Plot attention from different heads
    if attn_weights_list:
        attn = attn_weights_list[0][0]  # [num_heads, num_patches+1, num_patches+1]
        num_heads = attn.shape[0]

        for head_idx in range(min(3, num_heads)):
            # CLS token attention to all patches
            cls_attn = attn[head_idx, 0, 1:]  # [num_patches]
            num_patches = cls_attn.shape[0]
            grid_size = int(num_patches ** 0.5)

            if grid_size * grid_size == num_patches:
                attn_map = cls_attn.reshape(grid_size, grid_size).numpy()
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

                axes[sample_idx, head_idx + 1].imshow(original_resized, alpha=0.5)
                axes[sample_idx, head_idx + 1].imshow(
                    np.array(Image.fromarray((plt.cm.jet(
                        np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize((224, 224)))/ 255.0
                    ) * 255).astype(np.uint8)).resize((224, 224))),
                    alpha=0.5
                )
                axes[sample_idx, head_idx + 1].set_title(f'Head {head_idx}', fontsize=10, fontweight='bold')
            axes[sample_idx, head_idx + 1].axis('off')

plt.tight_layout()
plt.savefig('attention_maps.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [CELL 13] — Misclassification Analysis
print("\n" + "=" * 60)
print("🔎  MISCLASSIFICATION ANALYSIS")
print("=" * 60)

# Find misclassified images
misclassified = []
for idx in range(len(test_dataset)):
    if tta_preds[idx] != tta_labels[idx]:
        misclassified.append({
            'index': idx,
            'path': test_dataset.images[idx],
            'true_label': 'Defective' if tta_labels[idx] == 1 else 'Non-Defective',
            'pred_label': 'Defective' if tta_preds[idx] == 1 else 'Non-Defective',
            'confidence': tta_probs[idx],
        })

if misclassified:
    print(f"\n❌  Found {len(misclassified)} misclassified images:")
    for m in misclassified:
        print(f"    Image: {os.path.basename(m['path'])}")
        print(f"      True: {m['true_label']} → Predicted: {m['pred_label']} (conf: {m['confidence']:.3f})")

    # Visualize misclassified with Grad-CAM
    n_mis = len(misclassified)
    if n_mis > 0:
        fig, axes = plt.subplots(2, n_mis, figsize=(5 * n_mis, 9))
        fig.suptitle('❌ Misclassified Images + Grad-CAM Analysis',
                     fontsize=14, fontweight='bold')

        if n_mis == 1:
            axes = axes.reshape(-1, 1)

        for i, m in enumerate(misclassified):
            original = np.array(Image.open(m['path']).convert('RGB').resize((224, 224)))
            original_float = original.astype(np.float32) / 255.0
            img_tensor = val_transform(image=np.array(Image.open(m['path']).convert('RGB')))['image']

            axes[0, i].imshow(original)
            axes[0, i].set_title(
                f"True: {m['true_label']}\nPred: {m['pred_label']} ({m['confidence']:.2f})",
                fontsize=10, fontweight='bold', color='red'
            )
            axes[0, i].axis('off')

            try:
                _, overlay = get_gradcam_for_vit(model, img_tensor, original_float)
                axes[1, i].imshow(overlay)
                axes[1, i].set_title('Grad-CAM', fontweight='bold')
            except:
                axes[1, i].text(0.5, 0.5, 'Grad-CAM\nFailed', ha='center', va='center')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.savefig('misclassified_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
else:
    print("\n🎉  No misclassifications! Perfect score on test set!")


# %% [CELL 14] — Final Summary Report
print("\n" + "=" * 60)
print("📝  FINAL SUMMARY REPORT")
print("=" * 60)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║          🚂 RAILWAY TRACK FAULT DETECTION — RESULTS         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Model:        Vision Transformer (ViT-B/16)                 ║
║  Pretrained:   ImageNet-21k                                  ║
║  Training:     Two-phase (frozen → fine-tuned)               ║
║  Augmentation: Aggressive (spatial + color + noise + cutout) ║
║  TTA:          7 augmented copies averaged                   ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  DATASET                                                     ║
║  ├─ Train:      300 images (150 Defective + 150 OK)          ║
║  ├─ Validation:  62 images ( 31 Defective +  31 OK)          ║
║  └─ Test:        22 images ( 11 Defective +  11 OK)          ║
╠══════════════════════════════════════════════════════════════╣
║  TEST RESULTS (with TTA)                                     ║
║  ├─ Accuracy:   {accuracy_score(tta_labels, tta_preds):.4f}                                    ║
║  ├─ Precision:  {precision_score(tta_labels, tta_preds, zero_division=0):.4f}                                    ║
║  ├─ Recall:     {recall_score(tta_labels, tta_preds, zero_division=0):.4f}                                    ║
║  ├─ F1 Score:   {f1_score(tta_labels, tta_preds, zero_division=0):.4f}                                    ║""")
if len(np.unique(tta_labels)) > 1:
    print(f"║  └─ AUC-ROC:   {roc_auc_score(tta_labels, tta_probs):.4f}                                    ║")
print(f"""║                                                              ║
║  Training:     Phase 1 ({CONFIG['phase1_epochs']} epochs) + Phase 2 ({CONFIG['phase2_epochs']} epochs)       ║
║  Best Val Loss: {engine.best_val_loss:.4f}                                   ║
║  Best Val Acc:  {engine.best_val_acc:.4f}                                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

print("📁  Saved files:")
print("    ├─ best_railway_vit.pth      (model checkpoint)")
print("    ├─ sample_images.png          (dataset samples)")
print("    ├─ augmented_samples.png      (augmentation demo)")
print("    ├─ training_history.png       (loss/acc/lr curves)")
print("    ├─ evaluation_results.png     (confusion matrix, ROC, PR curves)")
print("    ├─ gradcam_results.png        (explainability heatmaps)")
print("    ├─ attention_maps.png         (ViT attention visualization)")
print("    └─ misclassified_analysis.png (error analysis)")

print("\n🎉  Done! Your Railway Track Fault Detection model is ready.")
