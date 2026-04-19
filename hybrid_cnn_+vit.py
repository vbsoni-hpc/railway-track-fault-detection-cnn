import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# =========================================================
# CONFIG
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_root = "archive (1)/Railway Track fault Detection Updated"

# Choose one:
BACKBONE_TYPE = "resnet18"       # "resnet18" or "efficientnet_b0"
WEIGHTS_PATH  = "best_resnet18.pth"   # or "efficientnet_model.pth"

num_classes = 2
batch_size = 16
img_size = 224
embedding_dim = 128
attention_heads = 4
transformer_blocks = 4
mlp_hidden_nodes = 256
learning_rate = 3e-4
epochs = 20
dropout_rate = 0.1

# =========================================================
# TRANSFORMS
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# =========================================================
# DATASETS
# =========================================================
train_dataset = ImageFolder(root=f"{data_root}/Train", transform=train_transform)
val_dataset   = ImageFolder(root=f"{data_root}/Validation", transform=eval_transform)
test_dataset  = ImageFolder(root=f"{data_root}/Test", transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class_names = train_dataset.classes
print(f"Classes: {class_names}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# =========================================================
# CHECKPOINT HELPERS
# =========================================================
def extract_state_dict(checkpoint):
    """
    Supports:
    - raw state_dict
    - {"state_dict": ...}
    - {"model_state_dict": ...}
    - {"model": ...}
    - {"net": ...}
    """
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "backbone_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint

def strip_prefixes(state_dict, prefixes=("module.", "backbone.", "model.", "net.")):
    """
    Removes common prefixes from saved checkpoints so they can load into torchvision models.
    Handles nested prefixes too.
    """
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    changed = True
        cleaned[new_k] = v
    return cleaned

# =========================================================
# BACKBONE LOADING
# =========================================================
def build_backbone(backbone_type, weight_path=None):
    if backbone_type == "resnet18":
        base = models.resnet18(weights=None)
        feature_channels = 512

        if weight_path is not None:
            ckpt = torch.load(weight_path, map_location=device)
            state_dict = extract_state_dict(ckpt)
            state_dict = strip_prefixes(state_dict)
            missing, unexpected = base.load_state_dict(state_dict, strict=False)
            print("\n[ResNet18] Loaded weights")
            print("Missing keys:", len(missing))
            print("Unexpected keys:", len(unexpected))

        feature_extractor = nn.Sequential(*list(base.children())[:-2])  # conv -> layer4
        return feature_extractor, feature_channels

    elif backbone_type == "efficientnet_b0":
        base = models.efficientnet_b0(weights=None)
        feature_channels = 1280

        if weight_path is not None:
            ckpt = torch.load(weight_path, map_location=device)
            state_dict = extract_state_dict(ckpt)
            state_dict = strip_prefixes(state_dict)
            missing, unexpected = base.load_state_dict(state_dict, strict=False)
            print("\n[EfficientNet-B0] Loaded weights")
            print("Missing keys:", len(missing))
            print("Unexpected keys:", len(unexpected))

        feature_extractor = base.features
        return feature_extractor, feature_channels

    else:
        raise ValueError("BACKBONE_TYPE must be 'resnet18' or 'efficientnet_b0'")

# =========================================================
# MODEL COMPONENTS
# =========================================================
class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads, mlp_hidden, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm attention block
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # Pre-norm MLP block
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x

class HybridCNNViT(nn.Module):
    def __init__(self, feature_extractor, feature_channels, img_size=224):
        super().__init__()
        self.backbone = feature_extractor

        # Freeze backbone for transfer learning
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(feature_channels, embedding_dim)

        # Infer token count dynamically using a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            dummy_feat = self.backbone(dummy)
            if dummy_feat.dim() != 4:
                raise RuntimeError(
                    f"Backbone must return a 4D feature map (B, C, H, W), got {dummy_feat.shape}"
                )
            _, c, h, w = dummy_feat.shape
            self.num_tokens = h * w

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_tokens + 1, embedding_dim) * 0.02
        )
        self.pos_drop = nn.Dropout(dropout_rate)

        self.transformer = nn.Sequential(
            *[
                TransformerEncoder(
                    dim=embedding_dim,
                    heads=attention_heads,
                    mlp_hidden=mlp_hidden_nodes,
                    dropout=dropout_rate
                )
                for _ in range(transformer_blocks)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # CNN feature map
        x = self.backbone(x)               # (B, C, H, W)

        # Turn spatial map into tokens
        x = x.flatten(2).transpose(1, 2)   # (B, tokens, C)

        # Project to transformer embedding size
        x = self.proj(x)                   # (B, tokens, embedding_dim)

        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)   # (B, tokens+1, embedding_dim)

        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.pos_drop(x)

        x = self.transformer(x)

        x = self.norm(x[:, 0])             # CLS token
        x = self.head(x)
        return x

# =========================================================
# BUILD MODEL
# =========================================================
backbone, feature_channels = build_backbone(BACKBONE_TYPE, WEIGHTS_PATH)
model = HybridCNNViT(backbone, feature_channels, img_size=img_size).to(device)

print(f"\nBackbone: {BACKBONE_TYPE}")
print(f"Feature channels: {feature_channels}")
print(f"Transformer tokens: {model.num_tokens + 1} (including CLS)")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# =========================================================
# TRAINING SETUP
# =========================================================
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
    weight_decay=1e-2
)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# =========================================================
# TRAIN / VALIDATE
# =========================================================
best_val_acc = 0.0

for epoch in range(epochs):
    # Ensure frozen backbone stays frozen in eval mode (important with BatchNorm)
    model.train()
    model.backbone.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100.0 * correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100.0 * val_correct / val_total
    scheduler.step()

    print(
        f"Epoch {epoch+1:02d}/{epochs} | "
        f"Train Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_hybrid_cnn_vit.pth")
        print(f"  >> Saved best model (Val Acc: {val_acc:.2f}%)")

# =========================================================
# TEST EVALUATION
# =========================================================
print("\n" + "=" * 60)
print("TESTING on held-out test set")
print("=" * 60)

model.load_state_dict(torch.load("best_hybrid_cnn_vit.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
print(f"\nTest Accuracy: {test_acc:.2f}%\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
