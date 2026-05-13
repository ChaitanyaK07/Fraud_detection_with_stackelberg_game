"""
=============================================================================
FILE 2: BASELINE MODEL -- Tabular Transformer Fraud Detector
=============================================================================
Deep Learning-Based Fraud Detection in Banking and UPI Transactions
Advisor: Aneesh Chivukula

Architecture: FT-Transformer (Feature Tokenization Transformer)
  - Each of the 18 features is independently projected to d_model=64 tokens
  - Learnable CLS token aggregates global representation
  - 2 Transformer blocks with 4-head attention
  - Classification head with dropout

Capacity is deliberately modest (64-dim, 2 layers) for 18 features
to prevent overfitting. Focal Loss handles class imbalance (~5:1).

Why Transformer over LSTM/GNN:
  - UPI transactions have no natural temporal sequence ordering
  - No explicit graph structure available
  - Self-attention captures arbitrary pairwise feature interactions
  - Matches SOTA on tabular fraud benchmarks (Gorishniy et al., 2021)
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
import os, pickle, warnings
warnings.filterwarnings('ignore')

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "processed_data"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters — tuned for 18 features, ~18K training samples
HP = dict(
    d_model      = 64,
    n_heads      = 4,
    n_layers     = 2,
    d_ffn        = 128,
    dropout      = 0.4,   # strong dropout for realistic convergence
    lr           = 3e-4,  # lower lr
    weight_decay = 1e-2,  # strong L2
    batch_size   = 256,
    epochs       = 60,
    patience     = 10,
    focal_gamma  = 2.0,
    focal_alpha  = 0.75,
)


# ─────────────────────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce    = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )
        p_t    = torch.exp(-bce)
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ─────────────────────────────────────────────────────────────
# TRANSFORMER BLOCK
# ─────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


# ─────────────────────────────────────────────────────────────
# TABULAR TRANSFORMER
# ─────────────────────────────────────────────────────────────
class TabularTransformer(nn.Module):
    """
    FT-Transformer for tabular fraud detection.
    Each feature is linearly projected to d_model, forming a token sequence.
    CLS token aggregates global representation for binary classification.
    """
    def __init__(self, n_features: int,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_ffn: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features

        # Each scalar feature -> d_model token
        self.feature_tokenizer = nn.Linear(1, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional embedding
        self.pos_emb = nn.Parameter(
            torch.randn(1, n_features + 1, d_model) * 0.02
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tokens = self.feature_tokenizer(x.unsqueeze(-1))          # (B, n, d)
        cls    = self.cls_token.expand(B, -1, -1)                  # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_emb   # (B, n+1, d)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0, :]).squeeze(-1)              # (B,)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_processed_data():
    X_train = np.load(f"{OUTPUT_DIR}/X_train.npy").astype(np.float32)
    X_val   = np.load(f"{OUTPUT_DIR}/X_val.npy").astype(np.float32)
    X_test  = np.load(f"{OUTPUT_DIR}/X_test.npy").astype(np.float32)
    y_train = np.load(f"{OUTPUT_DIR}/y_train.npy").astype(np.int64)
    y_val   = np.load(f"{OUTPUT_DIR}/y_val.npy").astype(np.int64)
    y_test  = np.load(f"{OUTPUT_DIR}/y_test.npy").astype(np.int64)

    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=HP['batch_size'],
                          shuffle=shuffle, num_workers=0)

    return (make_loader(X_train, y_train, True),
            make_loader(X_val,   y_val),
            make_loader(X_test,  y_test),
            X_train.shape[1])


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    logits_all, labels_all = [], []
    for Xb, yb in loader:
        logits_all.append(model(Xb.to(DEVICE)).cpu())
        labels_all.append(yb)
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= 0.5).astype(int)
    return dict(
        auc       = roc_auc_score(labels, probs),
        f1        = f1_score(labels, preds, zero_division=0),
        precision = precision_score(labels, preds, zero_division=0),
        recall    = recall_score(labels, preds, zero_division=0),
        cm        = confusion_matrix(labels, preds),
        probs     = probs,
        labels    = labels,
    )


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


def train_baseline():
    print("=" * 70)
    print("BASELINE MODEL -- Tabular Transformer Fraud Detector")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    train_loader, val_loader, test_loader, n_features = load_processed_data()
    print(f"[INFO] n_features={n_features}, train_size={len(train_loader.dataset)}")

    model = TabularTransformer(
        n_features=n_features,
        d_model=HP['d_model'], n_heads=HP['n_heads'],
        n_layers=HP['n_layers'], d_ffn=HP['d_ffn'],
        dropout=HP['dropout'],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {n_params:,}")

    criterion = FocalLoss(gamma=HP['focal_gamma'], alpha=HP['focal_alpha'])
    optimizer = optim.AdamW(model.parameters(),
                            lr=HP['lr'], weight_decay=HP['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=HP['epochs'], eta_min=1e-5
    )

    best_auc   = 0.0
    patience_c = 0
    history    = []

    for epoch in range(1, HP['epochs'] + 1):
        train_loss  = train_epoch(model, train_loader, optimizer, criterion)
        val_m       = evaluate(model, val_loader)
        scheduler.step()

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_auc': val_m['auc'], 'val_f1': val_m['f1'],
        })

        print(f"Epoch {epoch:3d}/{HP['epochs']} | "
              f"Loss: {train_loss:.4f} | "
              f"Val AUC: {val_m['auc']:.4f} | "
              f"Val F1: {val_m['f1']:.4f} | "
              f"Prec: {val_m['precision']:.4f} | "
              f"Rec: {val_m['recall']:.4f}")

        if val_m['auc'] > best_auc:
            best_auc   = val_m['auc']
            patience_c = 0
            torch.save(model.state_dict(), f"{MODEL_DIR}/baseline_best.pt")
        else:
            patience_c += 1
            if patience_c >= HP['patience']:
                print(f"\n[Early Stop] No AUC improvement for {HP['patience']} epochs.")
                break

    # Test evaluation
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION (best checkpoint)")
    print("=" * 70)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/baseline_best.pt",
                                      map_location=DEVICE))
    test_m = evaluate(model, test_loader)
    print(f"  AUC-ROC    : {test_m['auc']:.4f}")
    print(f"  F1-Score   : {test_m['f1']:.4f}")
    print(f"  Precision  : {test_m['precision']:.4f}")
    print(f"  Recall     : {test_m['recall']:.4f}")
    print(f"  Confusion Matrix:")
    cm = test_m['cm']
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

    np.save(f"{MODEL_DIR}/baseline_test_probs.npy",  test_m['probs'])
    np.save(f"{MODEL_DIR}/baseline_test_labels.npy", test_m['labels'])
    with open(f"{MODEL_DIR}/baseline_history.pkl", 'wb') as f:
        pickle.dump(history, f)

    print("\n[DONE] Baseline model saved.")
    return model, test_m, n_features


if __name__ == "__main__":
    train_baseline()
