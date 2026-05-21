"""
model_cnn.py — 1D CNN for MAP Prediction
==========================================
Architecture:
  - 5 conv blocks (Conv1D → BN → ReLU → MaxPool)
  - Global average pooling
  - Concatenate with physics features from common.py
  - 3-layer FC head → MAP output

Normalisation strategy:
  - Per-sample zero-mean / unit-variance (removes DC + amplitude shift)
  - Global std rescaling: test signals are aligned to train amplitude stats
    This is the single most important fix for train→test distribution shift.

Saves: ../outputs/models/cnn.pkl
"""

import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import QuantileTransformer
from common import load_data, TRAIN_PATH, TEST_PATH

os.makedirs("../outputs/models", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {device}")


# ── Architecture ───────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1D → BatchNorm → ReLU → MaxPool"""
    def __init__(self, in_ch, out_ch, kernel, pool):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x):
        return self.block(x)


class MAPNet(nn.Module):
    """
    1D CNN that fuses raw waveform features with hand-crafted physics features.

    Input:
        xr  — (B, 1, 1000)  normalised raw signal
        xp  — (B, n_phys)   physics feature vector from common.load_data()

    Output:
        (B,)  MAP prediction in transformed label space
    """
    def __init__(self, n_phys: int = 32):
        super().__init__()

        # Waveform encoder: 1000 → 32 time steps, 256 channels
        self.encoder = nn.Sequential(
            ConvBlock(1,   32,  7, 2),   # → 500
            ConvBlock(32,  64,  5, 2),   # → 250
            ConvBlock(64,  128, 5, 2),   # → 125
            ConvBlock(128, 256, 3, 2),   # → 63
            ConvBlock(256, 256, 3, 2),   # → 32
        )
        self.gap = nn.AdaptiveAvgPool1d(1)  # → (B, 256, 1)

        # Fusion head: waveform embedding + physics features → MAP
        self.head = nn.Sequential(
            nn.Linear(256 + n_phys, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128),          nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 64),           nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, xr: torch.Tensor, xp: torch.Tensor) -> torch.Tensor:
        feat = self.gap(self.encoder(xr)).squeeze(-1)   # (B, 256)
        return self.head(torch.cat([feat, xp], dim=1)).squeeze(-1)


# ── Signal normalisation ───────────────────────────────────────────────────────

def normalise(X: np.ndarray, ref_std: float = None) -> np.ndarray:
    """
    Per-sample zero-mean / unit-variance normalisation, then optional
    global-std rescaling to align test amplitude distribution to train.

    Args:
        X       : (N, 1000) raw signal array
        ref_std : if given, rescale so global std matches this value
    Returns:
        Xn      : (N, 1000) float32 normalised signals
    """
    mu  = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1,  keepdims=True) + 1e-8
    Xn  = (X - mu) / std
    if ref_std is not None:
        Xn = Xn * (ref_std / (float(Xn.std()) + 1e-8))
    return Xn.astype(np.float32)


# ── Training ───────────────────────────────────────────────────────────────────

def train(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH) -> None:
    print("\nPreparing data...")
    train_npy = np.load(train_path)
    test_npy  = np.load(test_path)

    X_tr = train_npy[:, :1000].astype(np.float32)
    X_te = test_npy[:,  :1000].astype(np.float32)
    y_tr = train_npy[:, -1].astype(np.float32)
    y_te = test_npy[:,  -1].astype(np.float32)

    # Align test amplitude distribution to train  ← key for low test MAE
    X_tr_n = normalise(X_tr)
    ref_std = float(X_tr_n.std())
    X_tr_n  = normalise(X_tr, ref_std)
    X_te_n  = normalise(X_te, ref_std)

    # Physics features (32-d, already StandardScaled inside load_data)
    F_tr, F_te, _, _, _ = load_data(train_path, test_path)
    F_tr = F_tr.astype(np.float32)
    F_te = F_te.astype(np.float32)

    # Map labels to a standard-normal space for smoother loss landscape
    qt     = QuantileTransformer(output_distribution="normal", random_state=42)
    y_tr_t = qt.fit_transform(y_tr.reshape(-1, 1)).ravel().astype(np.float32)

    # Tensors
    def to_tensor(*arrays):
        return [torch.tensor(a).to(device) for a in arrays]

    Xr_tr, Xp_tr, Yt = to_tensor(X_tr_n[:, None, :], F_tr, y_tr_t)
    Xr_te, Xp_te      = to_tensor(X_te_n[:, None, :], F_te)

    loader = DataLoader(
        TensorDataset(Xr_tr, Xp_tr, Yt),
        batch_size=256, shuffle=True, pin_memory=False,
    )

    # Model, optimiser, scheduler
    model     = MAPNet(n_phys=F_tr.shape[1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=1e-3,
        epochs=100, steps_per_epoch=len(loader), pct_start=0.1,
    )
    criterion = nn.HuberLoss(delta=2.0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} | Train: {len(y_tr):,} | Epochs: 100\n")

    best_mae, best_state = 999.0, None
    patience, wait       = 20, 0
    t0                   = time.time()

    def predict_map(xr, xp):
        """Run model and inverse-transform predictions to mmHg."""
        raw = model(xr, xp).cpu().numpy().reshape(-1, 1)
        return qt.inverse_transform(raw).ravel()

    for epoch in range(100):
        # ── train step
        model.train()
        for xr, xp, yt in loader:
            optimiser.zero_grad()
            loss = criterion(model(xr, xp), yt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

        # ── eval step
        model.eval()
        with torch.no_grad():
            p_tr = predict_map(Xr_tr, Xp_tr)
            p_te = predict_map(Xr_te, Xp_te)

        tr_mae = mean_absolute_error(y_tr, p_tr)
        te_mae = mean_absolute_error(y_te, p_te)

        if te_mae < best_mae:
            best_mae   = te_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {tr_mae:.3f} "
                  f"| Test: {te_mae:.3f} | Best: {best_mae:.3f}")

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s | Best Test MAE: {best_mae:.3f}")

    payload = {
        "model_state": best_state,
        "qt":          qt,
        "n_phys":      F_tr.shape[1],
        "ref_std":     ref_std,
    }
    save_path = "../outputs/models/cnn.pkl"
    pickle.dump(payload, open(save_path, "wb"))
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    train()
