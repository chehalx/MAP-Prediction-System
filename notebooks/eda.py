"""
eda.py — Exploratory Data Analysis for MAP Sensor Data
=======================================================
Run from notebooks/:  python eda.py

Produces 7 diagnostic plots in ../outputs/plots/:
  eda_1_distributions.png   — MAP & HR histograms + HR vs MAP scatter
  eda_2_signals.png         — Raw waveforms at low / mid / high MAP
  eda_3_decomposition.png   — Signal → cardiac → motion band decomposition
  eda_4_features.png        — Feature–MAP correlations + train→test shift
  eda_5_psd.png             — Average PSD train vs test
  eda_6_std_skew_shift.png  — STD & skew distribution shift (root cause check)
  eda_7_std_vs_map.png      — STD vs MAP correlation (key for normalisation)
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats
from scipy.signal import welch

from common import load_data, bandpass, extract_hr, FS, TRAIN_PATH, TEST_PATH

PLOT_DIR = "../outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}.png", dpi=130)
    plt.close()
    print(f"  {name}.png")


# ── Load ───────────────────────────────────────────────────────────────────────

print("Loading data...")
train_npy = np.load(TRAIN_PATH)
test_npy  = np.load(TEST_PATH)

X_tr = train_npy[:, :1000].astype(np.float32)
X_te = test_npy[:,  :1000].astype(np.float32)
y_tr = train_npy[:, -1].astype(np.float32)
y_te = test_npy[:,  -1].astype(np.float32)

Xc_tr = bandpass(X_tr, 0.5, 8)
Xm_tr = bandpass(X_tr, 8,  20)

hr_tr = extract_hr(X_tr)
hr_te = extract_hr(X_te)

# load_data returns: (F_tr, F_te, y_tr, y_te, scaler)
F_tr, F_te, _, _, _ = load_data(TRAIN_PATH, TEST_PATH)

t = np.arange(1000) / FS

print(f"  Train MAP: {y_tr.mean():.1f}±{y_tr.std():.1f}  |  "
      f"Test MAP: {y_te.mean():.1f}±{y_te.std():.1f}")


# ── Plot 1 — MAP & HR distributions ───────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(y_tr, bins=40, color="steelblue", alpha=0.7, label="Train")
axes[0].hist(y_te, bins=40, color="tomato",    alpha=0.7, label="Test")
axes[0].axvline(y_tr.mean(), color="blue", ls="--", label=f"Train μ={y_tr.mean():.1f}")
axes[0].axvline(y_te.mean(), color="red",  ls="--", label=f"Test  μ={y_te.mean():.1f}")
axes[0].set(title=f"MAP Distribution (shift={abs(y_tr.mean()-y_te.mean()):.1f} mmHg)",
            xlabel="MAP (mmHg)")
axes[0].legend(fontsize=7)

axes[1].hist(hr_tr, bins=40, color="steelblue", alpha=0.7, label="Train")
axes[1].hist(hr_te, bins=40, color="tomato",    alpha=0.7, label="Test")
axes[1].set(title="HR Distribution", xlabel="HR (bpm)")
axes[1].legend(fontsize=7)

axes[2].scatter(hr_tr, y_tr, s=5, alpha=0.3, color="steelblue", label="Train")
axes[2].scatter(hr_te, y_te, s=5, alpha=0.3, color="tomato",    label="Test")
axes[2].set(title="HR vs MAP", xlabel="HR (bpm)", ylabel="MAP (mmHg)")
axes[2].legend(fontsize=7)

fig.suptitle("Label & HR Distributions", fontweight="bold")
savefig("eda_1_distributions")


# ── Plot 2 — Raw signals at low / mid / high MAP ───────────────────────────────

lo = np.where(y_tr < np.percentile(y_tr, 15))[0][:2]
mi = np.where((y_tr > y_tr.mean() - 4) & (y_tr < y_tr.mean() + 4))[0][:2]
hi = np.where(y_tr > np.percentile(y_tr, 85))[0][:2]

fig, axes = plt.subplots(2, 3, figsize=(15, 6))
for col, (idxs, label, color) in enumerate([
        (lo, "Low",  "steelblue"),
        (mi, "Mid",  "seagreen"),
        (hi, "High", "tomato"),
]):
    for row, i in enumerate(idxs):
        axes[row][col].plot(t, X_tr[i], color=color, lw=0.8)
        axes[row][col].set_title(f"{label} MAP={y_tr[i]:.1f}", fontsize=8)
        axes[row][col].tick_params(labelsize=7)

fig.suptitle("Raw PPG Signals — Low / Mid / High MAP", fontweight="bold")
savefig("eda_2_signals")


# ── Plot 3 — Signal decomposition ─────────────────────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(15, 7))
for row, i in enumerate([lo[0], mi[0], hi[0]]):
    for ax, sig, title, color in zip(
        axes[row],
        [X_tr[i], Xc_tr[i], Xm_tr[i]],
        [f"Raw  MAP={y_tr[i]:.1f}", "Cardiac (0.5–8 Hz)", "Motion (8–20 Hz)"],
        ["gray", "steelblue", "tomato"],
    ):
        ax.plot(t, sig, color=color, lw=0.8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)

fig.suptitle("Signal Decomposition: Raw → Cardiac → Motion", fontweight="bold")
savefig("eda_3_decomposition")


# ── Plot 4 — Feature correlations & distribution shift ────────────────────────

corr  = np.array([np.corrcoef(F_tr[:, i], y_tr)[0, 1] for i in range(F_tr.shape[1])])
top   = np.argsort(np.abs(corr))[-15:][::-1]
shift = np.abs(F_te.mean(0) - F_tr.mean(0)) / (np.abs(F_tr.mean(0)) + 1e-8)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(
    [f"F{i}" for i in top], corr[top],
    color=["steelblue" if v > 0 else "tomato" for v in corr[top]],
)
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set(title="Top 15 Feature Correlations with MAP", xlabel="Pearson r")

axes[1].bar(range(len(shift)), shift, color="tomato", alpha=0.8)
axes[1].axhline(shift.mean(), color="navy", ls="--",
                label=f"Mean shift = {shift.mean():.2f}")
axes[1].set(title="Feature Distribution Shift (Train→Test)",
            xlabel="Feature Index", ylabel="|μ_te − μ_tr| / |μ_tr|")
axes[1].legend()

fig.suptitle("Feature Analysis", fontweight="bold")
savefig("eda_4_features")


# ── Plot 5 — PSD train vs test ────────────────────────────────────────────────

freqs   = welch(X_tr[0], fs=FS, nperseg=256)[0]
psd_tr  = np.mean([welch(s, fs=FS, nperseg=256)[1] for s in X_tr[:200]], axis=0)
psd_te  = np.mean([welch(s, fs=FS, nperseg=256)[1] for s in X_te[:200]], axis=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(freqs, psd_tr, color="steelblue", lw=1.5, label="Train")
ax.semilogy(freqs, psd_te, color="tomato",    lw=1.5, ls="--", label="Test")
ax.axvspan(0.5,  8, alpha=0.12, color="green",  label="Cardiac band (0.5–8 Hz)")
ax.axvspan(8,   20, alpha=0.12, color="orange", label="Motion band  (8–20 Hz)")
ax.set(title="Average PSD — Train vs Test", xlabel="Frequency (Hz)",
       ylabel="PSD (log scale)", xlim=(0, 25))
ax.legend()
savefig("eda_5_psd")


# ── Plot 6 — STD & skew shift (root cause of train→test gap) ──────────────────

tr_std  = X_tr.std(axis=1)
te_std  = X_te.std(axis=1)
tr_skew = scipy.stats.skew(X_tr, axis=1).astype(np.float32)
te_skew = scipy.stats.skew(X_te, axis=1).astype(np.float32)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(tr_std,  bins=50, alpha=0.65, color="steelblue", label="Train")
axes[0].hist(te_std,  bins=50, alpha=0.65, color="tomato",    label="Test")
axes[0].set(title="Signal STD Distribution Shift", xlabel="Per-sample STD")
axes[0].legend()

axes[1].hist(tr_skew, bins=50, alpha=0.65, color="steelblue", label="Train")
axes[1].hist(te_skew, bins=50, alpha=0.65, color="tomato",    label="Test")
axes[1].set(title="Signal Skew Distribution Shift  ← root cause of island",
            xlabel="Per-sample Skewness")
axes[1].legend()

fig.suptitle("Distribution Shift Diagnostics (STD & Skew)", fontweight="bold")
savefig("eda_6_std_skew_shift")

# Print key numbers
print(f"\n  STD  train: {tr_std.mean():.4f} ± {tr_std.std():.4f}")
print(f"  STD  test:  {te_std.mean():.4f} ± {te_std.std():.4f}")
print(f"  Skew train: {tr_skew.mean():.4f} ± {tr_skew.std():.4f}")
print(f"  Skew test:  {te_skew.mean():.4f} ± {te_skew.std():.4f}")


# ── Plot 7 — STD vs MAP (shows amplitude encodes blood pressure) ───────────────

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(tr_std[:3000], y_tr[:3000], s=3, alpha=0.25,
           color="steelblue", label="Train", rasterized=True)
ax.scatter(te_std,         y_te,        s=3, alpha=0.25,
           color="tomato",    label="Test",  rasterized=True)
r_tr = np.corrcoef(tr_std[:3000], y_tr[:3000])[0, 1]
r_te = np.corrcoef(te_std,        y_te)[0, 1]
ax.set(title=f"Signal STD vs MAP  (r_train={r_tr:+.3f}, r_test={r_te:+.3f})",
       xlabel="Per-sample STD", ylabel="MAP (mmHg)")
ax.legend()
savefig("eda_7_std_vs_map")

print(f"\n  STD–MAP correlation  train: r={r_tr:+.3f}")
print(f"  STD–MAP correlation  test:  r={r_te:+.3f}")
print("\nEDA complete — 7 plots saved to ../outputs/plots/")
