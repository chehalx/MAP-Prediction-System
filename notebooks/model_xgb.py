#XGBoost with label quantile transform.
import os, sys, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
from common import load_data, TRAIN_PATH, TEST_PATH
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import QuantileTransformer
import xgboost as xgb

def train(train_path=TRAIN_PATH, test_path=TEST_PATH):
    F_tr, F_te, y_tr, y_te, _ = load_data(train_path, test_path)

    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    y_tr_t = qt.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_te_t = qt.transform(y_te.reshape(-1,1)).ravel()

    t0 = time.time()
    model = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.02, max_depth=7,
        min_child_weight=3, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=0.5, early_stopping_rounds=80,
        random_state=42, n_jobs=-1, verbosity=0, eval_metric="mae"
    )
    model.fit(F_tr, y_tr_t, eval_set=[(F_te, y_te_t)], verbose=False)

    import numpy as np
    pred_tr = qt.inverse_transform(model.predict(F_tr).reshape(-1,1)).ravel()
    pred_te = qt.inverse_transform(model.predict(F_te).reshape(-1,1)).ravel()

    print(f"  XGBoost (improved)  — Train MAE: {mean_absolute_error(y_tr, pred_tr):.3f}  ({time.time()-t0:.1f}s)")
    os.makedirs("../outputs/models", exist_ok=True)
    pickle.dump({"model": model, "qt": qt}, open("../outputs/models/xgboost.pkl", "wb"))

if __name__ == "__main__":
    train()
