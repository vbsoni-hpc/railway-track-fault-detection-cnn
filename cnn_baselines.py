"""
This script:
   Trains CNN baselines (ResNet-18, EfficientNet-B0)
"""

# ════════════════════════════════════════════════════════════════
# CELL 1 — Imports & Configuration
# ════════════════════════════════════════════════════════════════
import os
import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import timm

# ── Reproducibility ──
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Hyperparameters ──
num_classes = 2
batch_size = 16
img_size = 224

# ── Data Paths ──
data_root = "archive (1)/Railway Track fault Detection Updated"

# ════════════════════════════════════════════════════════════════
# CELL 4 — CNN Baseline Models
# ════════════════════════════════════════════════════════════════

class CNNBaseline(nn.Module):
    """
    Wrapper that takes a pretrained CNN backbone from timm,
    replaces its classifier head, and supports freeze/unfreeze.
    """
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  {model_name}: feat_dim={feat_dim}, total params={total_params:,}")

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ════════════════════════════════════════════════════════════════
# CELL 5 — Training Utilities
# ════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc


def train_model(model, model_name, train_loader, val_loader,
                phase1_epochs=10, phase2_epochs=20, lr1=1e-3, lr2=5e-5):
    """
    Two-phase training: freeze backbone → unfreeze backbone.
    For ViT (no freeze/unfreeze), pass phase1_epochs=0.
    """
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # ── Phase 1: Frozen backbone ──
    if phase1_epochs > 0 and hasattr(model, 'freeze_backbone'):
        model.freeze_backbone()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr1, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs)
        print(f"\n  [{model_name}] Phase 1: Head-only training ({phase1_epochs} epochs)")

        for epoch in range(1, phase1_epochs + 1):
            t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            v_loss, v_acc = evaluate(model, val_loader, criterion)
            scheduler.step()

            history['train_loss'].append(t_loss)
            history['train_acc'].append(t_acc)
            history['val_loss'].append(v_loss)
            history['val_acc'].append(v_acc)

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_state = copy.deepcopy(model.state_dict())

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:02d}/{phase1_epochs} | "
                      f"Train Loss: {t_loss:.4f} Acc: {t_acc:.1f}% | "
                      f"Val Loss: {v_loss:.4f} Acc: {v_acc:.1f}%")

    # ── Phase 2: Fine-tune (or full training for ViT) ──
    if hasattr(model, 'unfreeze_backbone'):
        model.unfreeze_backbone()
        print(f"  [{model_name}] Phase 2: Fine-tuning ({phase2_epochs} epochs)")
    else:
        print(f"  [{model_name}] Full training ({phase2_epochs} epochs)")

    optimizer = optim.AdamW(model.parameters(), lr=lr2, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)
    patience, patience_limit = 0, 10

    for epoch in range(1, phase2_epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:02d}/{phase2_epochs} | "
                  f"Train Loss: {t_loss:.4f} Acc: {t_acc:.1f}% | "
                  f"Val Loss: {v_loss:.4f} Acc: {v_acc:.1f}% | "
                  f"Patience: {patience}/{patience_limit}")

        if patience >= patience_limit:
            print(f"    Early stopping at epoch {epoch}.")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    print(f"  [{model_name}] Best Val Acc: {best_val_acc:.2f}%")
    return history, best_val_acc




def main():
    # CELL 2 — Transforms & Data Loaders
    # ════════════════════════════════════════════════════════════════
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root=f"{data_root}/Train", transform=train_transform)
    val_dataset   = ImageFolder(root=f"{data_root}/Validation", transform=eval_transform)
    test_dataset  = ImageFolder(root=f"{data_root}/Test", transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ════════════════════════════════════════════════════════════════
    # ════════════════════════════════════════════════════════════════
    # CELL 6 — Train CNN Baselines
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TRAINING CNN BASELINES")
    print("=" * 60)

    cnn_configs = {
        'ResNet-18':       'resnet18',
        'EfficientNet-B0': 'efficientnet_b0',
    }

    trained_models = {}
    training_histories = {}
    val_accuracies = {}

    for display_name, timm_name in cnn_configs.items():
        print(f"\n{'─' * 50}")
        print(f"  Training: {display_name}")
        print(f"{'─' * 50}")

        model_cnn = CNNBaseline(timm_name, pretrained=True).to(device)
        hist, best_acc = train_model(
            model_cnn, display_name, train_loader, val_loader,
            phase1_epochs=10, phase2_epochs=20,
            lr1=1e-3, lr2=5e-5
        )
        trained_models[display_name] = model_cnn
        training_histories[display_name] = hist
        val_accuracies[display_name] = best_acc

        # Save checkpoint
        ckpt_path = f"best_{timm_name}.pth"
        torch.save(model_cnn.state_dict(), ckpt_path)
        print(f"  Saved: {ckpt_path}")


if __name__ == '__main__':
    main()

