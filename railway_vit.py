"""
Railway Track Fault Detection — Vision Transformer (ViT)
Minimal, clean version. Run cells one by one in your notebook.
"""

# ────────────────────────────────────────────────────────────
# CELL 1: Imports & Setup
# ────────────────────────────────────────────────────────────
import os, random, numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Seed everything
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ────────────────────────────────────────────────────────────
# CELL 2: Paths — UPDATE THESE IF NEEDED
# ────────────────────────────────────────────────────────────
TRAIN_DIR = r"archive (1)/Railway Track fault Detection Updated/Train"
VAL_DIR   = r"archive (1)/Railway Track fault Detection Updated/Validation"
TEST_DIR  = r"archive (1)/Railway Track fault Detection Updated/Test"

IMG_SIZE = 224
BATCH_SIZE = 16

# ────────────────────────────────────────────────────────────
# CELL 3: Dataset & Augmentations
# ────────────────────────────────────────────────────────────
train_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, border_mode=0, p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1),
    ], p=0.6),
    A.OneOf([
        A.GaussNoise(var_limit=(10, 50), p=1),
        A.GaussianBlur(blur_limit=(3, 5), p=1),
    ], p=0.3),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=IMG_SIZE//10, max_width=IMG_SIZE//10, min_holes=2, fill_value=0, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class RailwayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images, self.labels = [], []
        for class_name, label in [('Defective', 1), ('Non defective', 0)]:
            folder = os.path.join(root_dir, class_name)
            if not os.path.exists(folder):
                continue
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.images.append(os.path.join(folder, f))
                    self.labels.append(label)
        print(f"  Loaded {len(self.images)} images from {root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert('RGB'))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


print("Loading datasets...")
train_ds = RailwayDataset(TRAIN_DIR, train_transforms)
val_ds   = RailwayDataset(VAL_DIR,   val_transforms)
test_ds  = RailwayDataset(TEST_DIR,  val_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ────────────────────────────────────────────────────────────
# CELL 4: ViT Model
# ────────────────────────────────────────────────────────────
class RailwayViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features  # 768 for ViT-B/16

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        print(f"  ViT backbone: {model_name}, feature dim: {feat_dim}")
        print(f"  Total params: {sum(p.numel() for p in self.parameters()):,}")

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("  Backbone FROZEN")

    def unfreeze_backbone(self, ratio=0.5):
        for p in self.backbone.parameters():
            p.requires_grad = True
        # Keep patch embed frozen
        for p in self.backbone.patch_embed.parameters():
            p.requires_grad = False
        # Freeze bottom blocks
        blocks = list(self.backbone.blocks)
        freeze_n = int(len(blocks) * (1 - ratio))
        for b in blocks[:freeze_n]:
            for p in b.parameters():
                p.requires_grad = False
        print(f"  Unfroze top {len(blocks)-freeze_n}/{len(blocks)} blocks")

    def forward(self, x):
        return self.head(self.backbone(x))


model = RailwayViT(pretrained=True).to(DEVICE)

# ────────────────────────────────────────────────────────────
# CELL 5: Training Functions
# ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, preds, labels = 0, [], []
    for imgs, lbl in tqdm(loader, leave=False):
        imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs).squeeze(1)
        loss = criterion(out, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds.extend((torch.sigmoid(out).detach().cpu().numpy() >= 0.5).astype(int))
        labels.extend(lbl.cpu().numpy().astype(int))
    return total_loss / len(loader.dataset), accuracy_score(labels, preds)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, labels, probs = 0, [], [], []
    for imgs, lbl in loader:
        imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
        out = model(imgs).squeeze(1)
        loss = criterion(out, lbl)
        total_loss += loss.item() * imgs.size(0)
        p = torch.sigmoid(out).cpu().numpy()
        probs.extend(p)
        preds.extend((p >= 0.5).astype(int))
        labels.extend(lbl.cpu().numpy().astype(int))
    labels, preds, probs = np.array(labels), np.array(preds), np.array(probs)
    return (
        total_loss / len(loader.dataset),
        accuracy_score(labels, preds),
        recall_score(labels, preds, zero_division=0),
        f1_score(labels, preds, zero_division=0),
        probs
    )


def train_phase(model, train_loader, val_loader, optimizer, scheduler, epochs, phase_name):
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    best_state = None
    patience, patience_limit = 0, 10

    print(f"\n{'='*50}")
    print(f"  {phase_name}")
    print(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc, v_rec, v_f1, _ = evaluate(model, val_loader, criterion)
        lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"Train: loss={t_loss:.4f} acc={t_acc:.4f} | "
              f"Val: loss={v_loss:.4f} acc={v_acc:.4f} rec={v_rec:.4f} f1={v_f1:.4f} | "
              f"LR={lr:.1e}")

        if scheduler:
            scheduler.step()

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            print(f"    ✓ Best model saved (val_loss={v_loss:.4f})")
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"    Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    return best_loss

# ────────────────────────────────────────────────────────────
# CELL 6: Phase 1 — Frozen Backbone (train head only)
# ────────────────────────────────────────────────────────────
model.freeze_backbone()

optimizer1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=15, eta_min=1e-6)

train_phase(model, train_loader, val_loader, optimizer1, scheduler1,
            epochs=15, phase_name="PHASE 1: Feature Extraction (backbone frozen)")

# ────────────────────────────────────────────────────────────
# CELL 7: Phase 2 — Fine-tune top 50% of backbone
# ────────────────────────────────────────────────────────────
model.unfreeze_backbone(ratio=0.5)

optimizer2 = optim.AdamW([
    {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': 5e-6},
    {'params': model.head.parameters(), 'lr': 1e-4},
], weight_decay=1e-4)
scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=10, T_mult=2, eta_min=1e-7)

train_phase(model, train_loader, val_loader, optimizer2, scheduler2,
            epochs=30, phase_name="PHASE 2: Fine-tuning (top 50% unfrozen)")

# Save checkpoint
torch.save(model.state_dict(), 'best_railway_vit.pth')
print("\nModel saved to best_railway_vit.pth")

# ────────────────────────────────────────────────────────────
# CELL 8: Test Evaluation
# ────────────────────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
t_loss, t_acc, t_rec, t_f1, t_probs = evaluate(model, test_loader, criterion)
t_labels = np.array(test_ds.labels)
t_preds = (t_probs >= 0.5).astype(int)

print(f"\n{'='*50}")
print(f"  TEST RESULTS")
print(f"{'='*50}")
print(f"  Accuracy:   {t_acc:.4f}")
print(f"  Precision:  {precision_score(t_labels, t_preds, zero_division=0):.4f}")
print(f"  Recall:     {t_rec:.4f}")
print(f"  F1 Score:   {t_f1:.4f}")
if len(np.unique(t_labels)) > 1:
    print(f"  AUC-ROC:    {roc_auc_score(t_labels, t_probs):.4f}")
print(f"{'='*50}")
