"""
Railway Track Fault Detection — Ensemble Model + CNN Comparison
================================================================
This script:
  1. Trains CNN baselines (ResNet-18, EfficientNet-B0)
  2. Builds 3 ensemble strategies (soft voting, weighted voting, stacking)
  3. Generates comprehensive comparison plots for the project report

Run cell-by-cell in Jupyter or as a single script.
Requires: torch, torchvision, timm, scikit-learn, matplotlib, seaborn, numpy
"""

# ════════════════════════════════════════════════════════════════
# CELL 1 — Imports & Configuration
# ════════════════════════════════════════════════════════════════
import os
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as tv_models

import timm

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression

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
data_root = "archive/Railway Track fault Detection Updated"

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
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)  # [B, 2]
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_probs.extend(probs[:, 1].cpu().numpy())   # P(defective)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)


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
            v_loss, v_acc, _, _, _ = evaluate(model, val_loader, criterion)
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
        v_loss, v_acc, _, _, _ = evaluate(model, val_loader, criterion)
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

    cnn_configs = OrderedDict({
        'ResNet-18':       'resnet18',
        'EfficientNet-B0': 'efficientnet_b0',
    })

    trained_models = OrderedDict()
    training_histories = OrderedDict()
    val_accuracies = OrderedDict()

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


    # ════════════════════════════════════════════════════════════════
    # CELL 8 — Collect Predictions from All Models
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  COLLECTING PREDICTIONS")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()

    # Collect test predictions and probabilities
    test_results = OrderedDict()

    for name, model in trained_models.items():
        model.eval()
        _, acc, labels, preds, probs = evaluate(model, test_loader, criterion)
        test_results[name] = {'labels': labels, 'preds': preds, 'probs': probs, 'acc': acc}
        print(f"  {name:20s} → Test Acc: {acc:.2f}%")

    # Collect validation predictions (needed for stacking)
    val_results = OrderedDict()
    for name, model in trained_models.items():
        model.eval()
        _, acc, labels, preds, probs = evaluate(model, val_loader, criterion)
        val_results[name] = {'labels': labels, 'preds': preds, 'probs': probs}


    # ════════════════════════════════════════════════════════════════
    # CELL 9 — Ensemble Methods
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  BUILDING ENSEMBLES")
    print("=" * 60)

    # Ground truth (same for all models)
    test_labels = test_results[list(test_results.keys())[0]]['labels']
    val_labels = val_results[list(val_results.keys())[0]]['labels']

    # Stack probabilities: shape (N, num_models)
    model_names = list(trained_models.keys())
    test_prob_matrix = np.column_stack([test_results[n]['probs'] for n in model_names])
    val_prob_matrix  = np.column_stack([val_results[n]['probs']  for n in model_names])

    # ── 1. Soft Voting (average probabilities) ──
    ens_soft_probs = test_prob_matrix.mean(axis=1)
    ens_soft_preds = (ens_soft_probs >= 0.5).astype(int)
    ens_soft_acc = 100 * accuracy_score(test_labels, ens_soft_preds)
    print(f"  Ensemble (Soft Vote)     → Test Acc: {ens_soft_acc:.2f}%")

    test_results['Ensemble (Soft Vote)'] = {
        'labels': test_labels, 'preds': ens_soft_preds,
        'probs': ens_soft_probs, 'acc': ens_soft_acc
    }

    # ── 2. Weighted Voting (weight by validation accuracy) ──
    weights = np.array([val_accuracies.get(n, 50.0) for n in model_names])
    weights = weights / weights.sum()
    ens_weighted_probs = (test_prob_matrix * weights[np.newaxis, :]).sum(axis=1)
    ens_weighted_preds = (ens_weighted_probs >= 0.5).astype(int)
    ens_weighted_acc = 100 * accuracy_score(test_labels, ens_weighted_preds)
    print(f"  Ensemble (Weighted Vote) → Test Acc: {ens_weighted_acc:.2f}%")

    print(f"    Weights: {dict(zip(model_names, [f'{w:.3f}' for w in weights]))}")

    test_results['Ensemble (Weighted)'] = {
        'labels': test_labels, 'preds': ens_weighted_preds,
        'probs': ens_weighted_probs, 'acc': ens_weighted_acc
    }

    # ── 3. Stacking (Logistic Regression meta-learner) ──
    meta_clf = LogisticRegression(max_iter=1000, C=1.0)
    meta_clf.fit(val_prob_matrix, val_labels)
    ens_stack_probs = meta_clf.predict_proba(test_prob_matrix)[:, 1]
    ens_stack_preds = meta_clf.predict(test_prob_matrix)
    ens_stack_acc = 100 * accuracy_score(test_labels, ens_stack_preds)
    print(f"  Ensemble (Stacking)      → Test Acc: {ens_stack_acc:.2f}%")

    test_results['Ensemble (Stacking)'] = {
        'labels': test_labels, 'preds': ens_stack_preds,
        'probs': ens_stack_probs, 'acc': ens_stack_acc
    }


    # ════════════════════════════════════════════════════════════════
    # CELL 10 — Comprehensive Metrics Table
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  COMPREHENSIVE RESULTS")
    print("=" * 60)

    metrics_table = {}
    for name, res in test_results.items():
        labels = res['labels']
        preds = res['preds']
        probs = res['probs']

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        except:
            auc = 0.0

        metrics_table[name] = {
            'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1': f1, 'AUC-ROC': auc
        }

    # Print formatted table
    print(f"\n{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("─" * 65)
    for name, m in metrics_table.items():
        print(f"{name:<25} {m['Accuracy']:>7.4f} {m['Precision']:>7.4f} "
              f"{m['Recall']:>7.4f} {m['F1']:>7.4f} {m['AUC-ROC']:>7.4f}")

    # Count parameters
    param_counts = {}
    for name, model in trained_models.items():
        param_counts[name] = sum(p.numel() for p in model.parameters())
    print(f"\n{'Model':<25} {'Parameters':>15}")
    print("─" * 42)
    for name, count in param_counts.items():
        print(f"{name:<25} {count:>15,}")


    # ════════════════════════════════════════════════════════════════
    # CELL 11 — Publication-Quality Comparison Plots
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  GENERATING COMPARISON PLOTS")
    print("=" * 60)

    # Style setup
    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.dpi': 150,
    })

    COLORS = {
        'ResNet-18':               '#1f77b4',
        'EfficientNet-B0':         '#ff7f0e',
        'Ensemble (Soft Vote)':    '#9467bd',
        'Ensemble (Weighted)':     '#8c564b',
        'Ensemble (Stacking)':     '#e377c2',
    }

    # ────────────────────────────────────────
    # PLOT 1: Model Comparison Bar Chart
    # ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
    model_list = list(metrics_table.keys())
    n_models = len(model_list)
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    bar_width = 0.8 / n_models

    for i, name in enumerate(model_list):
        values = [metrics_table[name][m] for m in metric_names]
        bars = ax.bar(x + i * bar_width, values, bar_width,
                      label=name, color=COLORS.get(name, f'C{i}'),
                      edgecolor='white', linewidth=0.5)
        # Add value labels on top
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_names, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — All Metrics', fontweight='bold', fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=4, fontsize=8, frameon=True)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_metrics.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("  Saved: comparison_metrics.png")


    # ────────────────────────────────────────
    # PLOT 2: ROC Curves (All Models)
    # ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, res in test_results.items():
        labels = res['labels']
        probs = res['probs']
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            auc_val = roc_auc_score(labels, probs)
            ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc_val:.3f})',
                    color=COLORS.get(name, None))

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random Baseline')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models', fontweight='bold', fontsize=15)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig('comparison_roc.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("  Saved: comparison_roc.png")


    # ────────────────────────────────────────
    # PLOT 3: Confusion Matrix Grid
    # ────────────────────────────────────────
    n_methods = len(test_results)
    cols = 4
    rows = (n_methods + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, res) in enumerate(test_results.items()):
        r, c = idx // cols, idx % cols
        cm = confusion_matrix(res['labels'], res['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[r, c],
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 14, 'fontweight': 'bold'},
                    cbar=False)
        acc = accuracy_score(res['labels'], res['preds'])
        axes[r, c].set_title(f'{name}\nAcc: {acc:.1%}', fontweight='bold', fontsize=10)
        axes[r, c].set_xlabel('Predicted')
        axes[r, c].set_ylabel('Actual')

    # Hide unused subplots
    for idx in range(n_methods, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis('off')

    fig.suptitle('Confusion Matrices — All Models', fontweight='bold', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig('comparison_confusion.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("  Saved: comparison_confusion.png")


    # ────────────────────────────────────────
    # PLOT 4: Radar / Spider Chart
    # ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
    num_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for name in test_results:
        values = [metrics_table[name][m] for m in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name,
                color=COLORS.get(name, None), markersize=5)
        ax.fill(angles, values, alpha=0.08, color=COLORS.get(name, None))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontweight='bold', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, alpha=0.7)
    ax.set_title('Multi-Metric Radar Comparison', fontweight='bold',
                 fontsize=15, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=8)

    plt.tight_layout()
    plt.savefig('comparison_radar.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("  Saved: comparison_radar.png")


    # ────────────────────────────────────────
    # PLOT 5: Parameter Efficiency (Accuracy vs Model Size)
    # ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    for name in trained_models:
        params = param_counts[name]
        acc = metrics_table[name]['Accuracy'] * 100
        f1 = metrics_table[name]['F1']

        size = max(f1 * 600, 80)  # bubble size proportional to F1
        ax.scatter(params / 1e6, acc, s=size,
                   color=COLORS.get(name, 'gray'), alpha=0.8,
                   edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(name, (params / 1e6, acc),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight='bold',
                    color=COLORS.get(name, 'gray'))

    ax.set_xlabel('Parameters (Millions)', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_title('Parameter Efficiency — Accuracy vs. Model Size\n(bubble size ∝ F1 score)',
                 fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig('comparison_efficiency.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("  Saved: comparison_efficiency.png")


    # ────────────────────────────────────────
    # PLOT 6: Training Curves Comparison
    # ────────────────────────────────────────
    if training_histories:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for name, hist in training_histories.items():
            epochs_range = range(1, len(hist['train_acc']) + 1)
            axes[0].plot(epochs_range, hist['train_acc'], '-', label=f'{name} (train)',
                         color=COLORS.get(name, None), alpha=0.6)
            axes[0].plot(epochs_range, hist['val_acc'], '--', label=f'{name} (val)',
                         color=COLORS.get(name, None), linewidth=2)

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Training & Validation Accuracy', fontweight='bold')
        axes[0].legend(fontsize=7, ncol=2)
        axes[0].grid(True, alpha=0.3)

        for name, hist in training_histories.items():
            epochs_range = range(1, len(hist['train_loss']) + 1)
            axes[1].plot(epochs_range, hist['train_loss'], '-', label=f'{name} (train)',
                         color=COLORS.get(name, None), alpha=0.6)
            axes[1].plot(epochs_range, hist['val_loss'], '--', label=f'{name} (val)',
                         color=COLORS.get(name, None), linewidth=2)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training & Validation Loss', fontweight='bold')
        axes[1].legend(fontsize=7, ncol=2)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle('CNN Baseline Training Curves', fontweight='bold', fontsize=15, y=1.02)
        plt.tight_layout()
        plt.savefig('comparison_training_curves.png', dpi=200, bbox_inches='tight')
        plt.show()
        print("  Saved: comparison_training_curves.png")


    # ════════════════════════════════════════════════════════════════
    # CELL 12 — Final Summary Report
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON REPORT")
    print("=" * 70)

    # Find best individual and best ensemble
    individual_models = {k: v for k, v in metrics_table.items() if 'Ensemble' not in k}
    ensemble_models = {k: v for k, v in metrics_table.items() if 'Ensemble' in k}

    best_individual = max(individual_models, key=lambda k: individual_models[k]['F1'])
    best_ensemble = max(ensemble_models, key=lambda k: ensemble_models[k]['F1'])

    print(f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║       RAILWAY TRACK FAULT DETECTION — ENSEMBLE COMPARISON          ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  Dataset: 300 train / 62 val / 22 test images (balanced)             ║
    ║  Task:    Binary classification (Defective vs Non-defective)         ║
    ║                                                                      ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  INDIVIDUAL MODELS                                                   ║
    ║  ┌──────────────────────┬────────┬───────┬────────┬──────┬────────┐  ║
    ║  │ Model                │  Acc   │ Prec  │ Recall │  F1  │  AUC   │  ║
    ║  ├──────────────────────┼────────┼───────┼────────┼──────┼────────┤  ║""")

    for name, m in individual_models.items():
        marker = " ◄" if name == best_individual else "  "
        print(f"║  │ {name:<20s} │ {m['Accuracy']:.4f} │{m['Precision']:.4f} │ {m['Recall']:.4f} │{m['F1']:.4f}│ {m['AUC-ROC']:.4f} │{marker}║")

    print(f"""║  └──────────────────────┴────────┴───────┴────────┴──────┴────────┘  ║
    ║                                                                      ║
    ║  ENSEMBLE METHODS                                                    ║
    ║  ┌──────────────────────┬────────┬───────┬────────┬──────┬────────┐  ║
    ║  │ Method               │  Acc   │ Prec  │ Recall │  F1  │  AUC   │  ║
    ║  ├──────────────────────┼────────┼───────┼────────┼──────┼────────┤  ║""")

    for name, m in ensemble_models.items():
        marker = " ◄" if name == best_ensemble else "  "
        print(f"║  │ {name:<20s} │ {m['Accuracy']:.4f} │{m['Precision']:.4f} │ {m['Recall']:.4f} │{m['F1']:.4f}│ {m['AUC-ROC']:.4f} │{marker}║")

    print(f"""║  └──────────────────────┴────────┴───────┴────────┴──────┴────────┘  ║
    ║                                                                      ║
    ║  Best Individual: {best_individual:<45s}    ║
    ║  Best Ensemble:   {best_ensemble:<45s}    ║
    ║                                                                      ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  MODEL SIZES                                                         ║""")
    for name, count in param_counts.items():
        print(f"║    {name:<20s}: {count:>12,} params                           ║")

    print(f"""║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    print("Saved figures:")
    print("  ├─ comparison_metrics.png        (bar chart: all metrics)")
    print("  ├─ comparison_roc.png            (ROC curves overlay)")
    print("  ├─ comparison_confusion.png      (confusion matrix grid)")
    print("  ├─ comparison_radar.png          (radar/spider chart)")
    print("  ├─ comparison_efficiency.png     (accuracy vs model size)")
    print("  └─ comparison_training_curves.png (CNN training histories)")
    print("\nDone! Use these plots in your project report.")

if __name__ == '__main__':
    main()
