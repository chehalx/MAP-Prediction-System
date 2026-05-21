#Better than plain linear but still poor generalisation under distribution shift.

import os, sys, pickle, time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(__file__))
from common import load_data, TRAIN_PATH, TEST_PATH

MODEL_PATH = "../outputs/models/ridge_rbf.pkl"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def train(train_path=TRAIN_PATH, test_path=TEST_PATH):
    F_tr_s, _, y_tr, _, _ = load_data(train_path, test_path)

    print("\nTraining: Ridge + RBF Kernel...")
    t0    = time.time()
    model = make_pipeline(
        RBFSampler(gamma=0.05, n_components=2000, random_state=42),
        Ridge(alpha=10.0)
    )
    model.fit(F_tr_s, y_tr)

    train_mae = mean_absolute_error(y_tr, model.predict(F_tr_s))
    print(f"  Train MAE : {train_mae:.4f} mmHg")
    print(f"  Time      : {time.time()-t0:.1f}s")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved     : {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()

def run():
    import os
    import joblib
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from common import load_data, extract_features

    os.makedirs("../outputs/models", exist_ok=True)

    X_train, y_train, _, _, _, _ = load_data(
        "../data/dataset_train.npy",
        "../data/dataset_test.npy"
    )

    sc = StandardScaler()
    X_tr = sc.fit_transform(extract_features(X_train))

    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_train)

    joblib.dump(model, "../outputs/models/ridge.pkl")
    joblib.dump(sc, "../outputs/models/ridge_scaler.pkl")

    print("Ridge done")
