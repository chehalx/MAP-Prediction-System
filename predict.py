
import os, sys, pickle, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, "notebooks")
from common import load_data

MODELS_DIR, PLOTS_DIR = "outputs/models", "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load data
F_tr, F_te, y_tr, y_te, _ = load_data("data/dataset_train.npy", "data/dataset_test.npy")

train_raw = np.load("data/dataset_train.npy")
test_raw  = np.load("data/dataset_test.npy")
X_tr_raw  = train_raw[:, :1000].astype(np.float32)
X_te_raw  = test_raw[:,  :1000].astype(np.float32)

def norm_raw(X, ref_std=None):
    mu = X.mean(1, keepdims=True); std = X.std(1, keepdims=True) + 1e-8
    Xn = (X - mu) / std
    if ref_std is not None:
        Xn = Xn * (ref_std / (float(Xn.std()) + 1e-8))
    return Xn.astype(np.float32)

X_tr_norm = norm_raw(X_tr_raw)
ref_std   = float(X_tr_norm.std())
X_tr_norm = norm_raw(X_tr_raw, ref_std)
X_te_norm = norm_raw(X_te_raw, ref_std)  # align test amplitude to train

# Predict helpers
def predict_sklearn(obj, F):
    if isinstance(obj, dict):
        p = obj["model"].predict(F).reshape(-1,1)
        return obj["qt"].inverse_transform(p).ravel()
    return obj.predict(F)

def predict_cnn(payload, X_norm, F_phys):
    import torch
    from model_cnn import MAPNet
    model = MAPNet(n_phys=payload["n_phys"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    with torch.no_grad():
        xr = torch.tensor(X_norm[:, None, :])
        xp = torch.tensor(F_phys.astype(np.float32))
        p  = model(xr, xp).numpy().reshape(-1,1)
    return payload["qt"].inverse_transform(p).ravel()

# Load all models & evaluate
pkls = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl"))
if not pkls: sys.exit(f"No models in {MODELS_DIR}/. Run notebooks/run_all.py first.")

res = {}
for fname in pkls:
    name = fname.replace(".pkl","").replace("_"," ").title()
    obj  = pickle.load(open(os.path.join(MODELS_DIR, fname), "rb"))
    if fname == "cnn.pkl":
        p_tr = predict_cnn(obj, X_tr_norm, F_tr)
        p_te = predict_cnn(obj, X_te_norm, F_te)
    else:
        p_tr = predict_sklearn(obj, F_tr)
        p_te = predict_sklearn(obj, F_te)
    res[name] = dict(train=mean_absolute_error(y_tr, p_tr),
                     test =mean_absolute_error(y_te, p_te),
                     gap  =mean_absolute_error(y_te, p_te) - mean_absolute_error(y_tr, p_tr),
                     pred =p_te)

ranked    = sorted(res.items(), key=lambda x: x[1]["test"])
best_name = ranked[0][0]

# Print table
print(f"\n{'Model':<22} {'Train':>8} {'Test':>8} {'Gap':>8}")
print("-" * 52)
for name, r in ranked:
    tag = "  ✓ BEST" if name == best_name else ("  overfit" if r["gap"] > 3 else "")
    print(f"{name:<22} {r['train']:>8.3f} {r['test']:>8.3f} {r['gap']:>+8.3f}{tag}")
print(f"\nBest: {best_name}  —  Test MAE = {ranked[0][1]['test']:.3f} mmHg")

# Plot helpers
def subplots():
    n = len(ranked)
    fig, axes = plt.subplots((n+2)//3, 3, figsize=(16, 5*((n+2)//3)))
    axes = axes.flatten()
    [axes[j].set_visible(False) for j in range(n, len(axes))]
    return fig, axes

def border(ax, name):
    best = (name == best_name)
    for sp in ax.spines.values():
        sp.set_edgecolor("green" if best else "#aaa")
        sp.set_linewidth(2.5 if best else 0.8)
    return best

def save(fname):
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{fname}.png", dpi=130); plt.close()
    print(f"  {fname}.png")

# Scatter Plot
vlo = min(y_te.min(), min(r["pred"].min() for _,r in res.items())) - 2
vhi = max(y_te.max(), max(r["pred"].max() for _,r in res.items())) + 2

fig, axes = subplots()
for i, (name, r) in enumerate(ranked):
    ax = axes[i]; b = border(ax, name)
    ax.scatter(y_te, r["pred"], alpha=0.25, s=4,
               color="green" if b else "steelblue", rasterized=True)
    ax.plot([vlo,vhi],[vlo,vhi], "r--", lw=1.2, label="Perfect (y=x)")
    ax.fill_between([vlo,vhi],[vlo-5,vhi-5],[vlo+5,vhi+5],
                    alpha=0.08, color="orange", label="±5 mmHg")
    ax.set(xlim=(vlo,vhi), ylim=(vlo,vhi),
           xlabel="True MAP (mmHg)", ylabel="Predicted MAP (mmHg)", aspect="equal")
    ax.set_title(f"{'★ ' if b else ''}{name}\nMAE={r['test']:.2f}  Gap={r['gap']:+.2f}", fontsize=9)
    ax.legend(fontsize=7, loc="upper left"); ax.tick_params(labelsize=7)
fig.suptitle("True vs Predicted — Test Set  (points on diagonal = perfect)", fontsize=11)
save("predict_scatter")

# Residuals Plot
fig, axes = subplots()
for i, (name, r) in enumerate(ranked):
    ax = axes[i]; b = border(ax, name)
    residuals = r["pred"] - y_te
    ax.scatter(y_te, residuals, alpha=0.2, s=4,
               color="green" if b else "steelblue", rasterized=True)
    ax.axhline(0,  color="red",    lw=1.2, ls="--", label="Zero error")
    ax.axhline( 5, color="orange", lw=0.8, ls=":")
    ax.axhline(-5, color="orange", lw=0.8, ls=":", label="±5 mmHg")
    ax.set(xlabel="True MAP (mmHg)", ylabel="Residual (pred−true)")
    ax.set_title(f"{'★ ' if b else ''}{name}\nstd={residuals.std():.2f}  bias={residuals.mean():+.2f}", fontsize=9)
    ax.legend(fontsize=7); ax.tick_params(labelsize=7)
fig.suptitle("Residuals — tight cloud around zero = good model", fontsize=11)
save("predict_residuals")

# MAE bars Plot
x = np.arange(len(ranked))
fig, ax = plt.subplots(figsize=(12, 4))
bars = ax.bar(x-0.18, [r["test"]  for _,r in ranked], 0.32, label="Test MAE",  color="tomato",    alpha=0.85)
ax.bar(       x+0.18, [r["train"] for _,r in ranked], 0.32, label="Train MAE", color="steelblue", alpha=0.85)
[ax.text(b.get_x()+b.get_width()/2, r["test"]+0.1, f"{r['test']:.2f}", ha="center", fontsize=8)
 for b, (_,r) in zip(bars, ranked)]
ax.set_xticks(x); ax.set_xticklabels([n for n,_ in ranked], rotation=15, ha="right")
ax.set(ylabel="MAE (mmHg)", title="Train vs Test MAE — smaller gap = better generalisation")
ax.legend()
save("predict_mae_bars")

# Trend comparison Plot
best_pred = ranked[0][1]["pred"]
idx = np.argsort(y_te)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(y_te[idx],      lw=1.2, color="steelblue", label="True MAP",      alpha=0.9)
ax.plot(best_pred[idx], lw=1.2, color="tomato",    label="Predicted MAP", alpha=0.9, ls="--")
ax.set(xlabel="Sample (sorted by true MAP)", ylabel="MAP (mmHg)",
       title=f"MAP: Prediction vs Label — {best_name}  (Test MAE={ranked[0][1]['test']:.3f} mmHg)")
ax.legend()
save("predict_trend_best")

np.save("outputs/best_predictions.npy", best_pred)
print(f"\nBest predictions → outputs/best_predictions.npy")