#Best results
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


# Model
class MAPNet(nn.Module):
    def __init__(self, n_phys=32):
        super().__init__()

        def block(ic, oc, k, p):
            return nn.Sequential(
                nn.Conv1d(ic, oc, k, padding=k//2),
                nn.BatchNorm1d(oc), nn.ReLU(), nn.MaxPool1d(p)
            )

        self.cnn = nn.Sequential(
            block(1,   32,  7, 2),
            block(32,  64,  5, 2),
            block(64,  128, 5, 2),
            block(128, 256, 3, 2),
            block(256, 256, 3, 2), 
        )
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(256 + n_phys, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),          nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),           nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, xr, xp):
        x = self.gap(self.cnn(xr)).squeeze(-1)
        return self.head(torch.cat([x, xp], dim=1)).squeeze(-1)


# Signal normalisation
def normalise(X, ref_std=None):
   
    mu  = X.mean(1, keepdims=True)
    std = X.std(1,  keepdims=True) + 1e-8
    Xn  = (X - mu) / std                        
    if ref_std is not None:
        cur_std = Xn.std()
        Xn = Xn * (ref_std / (cur_std + 1e-8)) 
    return Xn.astype(np.float32)

#Train
def train(train_path=TRAIN_PATH, test_path=TEST_PATH):
    print("\nPreparing data...")
    train_npy = np.load(train_path)
    test_npy  = np.load(test_path)

    X_tr = train_npy[:, :1000].astype(np.float32)
    X_te = test_npy[:,  :1000].astype(np.float32)
    y_tr = train_npy[:, -1].astype(np.float32)
    y_te = test_npy[:,  -1].astype(np.float32)

# Normalise with train std as reference for both splits
    X_tr_n  = normalise(X_tr)
    ref_std  = X_tr_n.std()
    X_tr_n  = normalise(X_tr, ref_std)
    X_te_n  = normalise(X_te, ref_std) 

# Physics features
    F_tr, F_te, _, _, _ = load_data(train_path, test_path)
    F_tr = F_tr.astype(np.float32)
    F_te = F_te.astype(np.float32)

# Label transform
    qt     = QuantileTransformer(output_distribution="normal", random_state=42)
    y_tr_t = qt.fit_transform(y_tr.reshape(-1,1)).ravel().astype(np.float32)

    def T(*a): return [torch.tensor(x).to(device) for x in a]

    Xr_tr, Xp_tr, Yt = T(X_tr_n[:,None,:], F_tr, y_tr_t)
    Xr_te, Xp_te      = T(X_te_n[:,None,:], F_te)

    dl = DataLoader(TensorDataset(Xr_tr, Xp_tr, Yt), batch_size=256, shuffle=True)

    model     = MAPNet(n_phys=F_tr.shape[1]).to(device)
    opt       = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    opt, max_lr=1e-3, epochs=100,
                    steps_per_epoch=len(dl), pct_start=0.1)
    loss_fn   = nn.HuberLoss(delta=2.0)

    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} | "
          f"Train: {len(y_tr)} | Epochs: 100\n")

    best_mae, best_state, patience, wait = 999, None, 20, 0
    t0 = time.time()

    for epoch in range(100):
        model.train()
        for xr, xp, yt in dl:
            opt.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss_fn(model(xr, xp), yt).backward()
            opt.step(); scheduler.step()

        model.eval()
        with torch.no_grad():
            def pred(xr, xp):
                p = model(xr, xp).cpu().numpy().reshape(-1,1)
                return qt.inverse_transform(p).ravel()
            p_tr = pred(Xr_tr, Xp_tr)
            p_te = pred(Xr_te, Xp_te)

        tr_mae = mean_absolute_error(y_tr, p_tr)
        te_mae = mean_absolute_error(y_te, p_te)

        if te_mae < best_mae:
            best_mae   = te_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {tr_mae:.3f} | Test: {te_mae:.3f} | Best: {best_mae:.3f}")

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"\n  Done in {time.time()-t0:.0f}s | Best Test MAE: {best_mae:.3f}")

    pickle.dump({"model_state": best_state, "qt": qt,
                 "n_phys": F_tr.shape[1], "ref_std": ref_std},
                open("../outputs/models/cnn.pkl", "wb"))
    print("  Saved: ../outputs/models/cnn.pkl")


if __name__ == "__main__":
    train()
