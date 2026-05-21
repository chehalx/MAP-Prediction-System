"""
predict.py — MAP Prediction: load model, evaluate, plot.
=========================================================
Run from project root:
    python predict.py

Loads the saved CNN model, predicts MAP on the test set,
prints Train/Test MAE, and saves the trend plot via plot_2vectors().

The evaluator runs ONLY this file — no training required.
"""

import sys, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "notebooks")

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from common    import load_data
from model_cnn import MAPNet, normalise
from utils     import plot_2vectors

TRAIN_PATH = "data/dataset_train.npy"
TEST_PATH  = "data/dataset_test.npy"
MODEL_PATH = "outputs/models/cnn.pkl"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
F_tr, F_te, y_tr, y_te, _ = load_data(TRAIN_PATH, TEST_PATH)

train_raw = np.load(TRAIN_PATH)
test_raw  = np.load(TEST_PATH)
X_tr_raw  = train_raw[:, :1000].astype(np.float32)
X_te_raw  = test_raw[:,  :1000].astype(np.float32)

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model...")
payload = pickle.load(open(MODEL_PATH, "rb"))
model   = MAPNet(n_phys=payload["n_phys"])
model.load_state_dict(payload["model_state"])
model.eval()

qt      = payload["qt"]
ref_std = payload["ref_std"]

X_tr_n = normalise(X_tr_raw, ref_std)
X_te_n = normalise(X_te_raw, ref_std)

# ── Predict ────────────────────────────────────────────────────────────────────
def predict_map(X_norm: np.ndarray, F_phys: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xr  = torch.tensor(X_norm[:, None, :])
        xp  = torch.tensor(F_phys.astype(np.float32))
        raw = model(xr, xp).numpy().reshape(-1, 1)
    return qt.inverse_transform(raw).ravel()

print("Predicting...")
pred_tr = predict_map(X_tr_n, F_tr)
pred_te = predict_map(X_te_n, F_te)

# ── MAE ────────────────────────────────────────────────────────────────────────
train_mae = mean_absolute_error(y_tr, pred_tr)
test_mae  = mean_absolute_error(y_te, pred_te)
print(f"\n  Train MAE : {train_mae:.3f} mmHg")
print(f"  Test  MAE : {test_mae:.3f} mmHg")

# ── Plot via plot_2vectors(label, pred, name) ──────────────────────────────────
# name is used as the output filename: outputs/map_prediction
import os
os.makedirs("outputs/plots", exist_ok=True)
os.chdir("outputs/plots")   # plot_2vectors saves to {name}.png in cwd

plot_2vectors(y_te, pred_te, "map_prediction")
print("  outputs/plots/map_prediction.png")
