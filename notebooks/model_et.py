#More randomised than Random Forest — lower variance but still hurt by feature distribution shift between train and test.

import os, sys, pickle, time
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(__file__))
from common import load_data, TRAIN_PATH, TEST_PATH

MODEL_PATH = "../outputs/models/extra_trees.pkl"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def train(train_path=TRAIN_PATH, test_path=TEST_PATH):
    F_tr_s, _, y_tr, _, _ = load_data(train_path, test_path)

    print("\nTraining: Extra Trees...")
    t0    = time.time()
    model = ExtraTreesRegressor(
        n_estimators=300, max_depth=None,
        min_samples_leaf=5, n_jobs=-1, random_state=42
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
