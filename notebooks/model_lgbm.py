import os, sys, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
from common import load_data, TRAIN_PATH, TEST_PATH
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import QuantileTransformer
import lightgbm as lgb

def train(train_path=TRAIN_PATH, test_path=TEST_PATH):
    F_tr, F_te, y_tr, y_te, _ = load_data(train_path, test_path)

    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    y_tr_t = qt.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_te_t = qt.transform(y_te.reshape(-1,1)).ravel()

    t0 = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.02, num_leaves=63, max_depth=7,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=0.5,
        random_state=42, n_jobs=-1, verbose=-1
    )
    model.fit(F_tr, y_tr_t,
              eval_set=[(F_te, y_te_t)],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])

    pred_tr = qt.inverse_transform(model.predict(F_tr).reshape(-1,1)).ravel()
    pred_te = qt.inverse_transform(model.predict(F_te).reshape(-1,1)).ravel()
    print(f"  LightGBM — Train MAE: {mean_absolute_error(y_tr, pred_tr):.3f}  Test MAE: {mean_absolute_error(y_te, pred_te):.3f}  ({time.time()-t0:.1f}s)")

    os.makedirs("../outputs/models", exist_ok=True)
    pickle.dump({"model": model, "qt": qt}, open("../outputs/models/lightgbm.pkl", "wb"))

if __name__ == "__main__":
    train()
