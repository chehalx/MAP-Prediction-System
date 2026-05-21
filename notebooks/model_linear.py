
import os, sys, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
from common import load_data, TRAIN_PATH, TEST_PATH
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def train(train_path=TRAIN_PATH, test_path=TEST_PATH):
    F_tr, F_te, y_tr, y_te, _ = load_data(train_path, test_path)
    t0 = time.time()
    model = LinearRegression().fit(F_tr, y_tr)
    print(f"  Linear Regression  — Train MAE: {mean_absolute_error(y_tr, model.predict(F_tr)):.3f}  Test MAE: {mean_absolute_error(y_te, model.predict(F_te)):.3f}  ({time.time()-t0:.1f}s)")
    os.makedirs("../outputs/models", exist_ok=True)
    pickle.dump(model, open("../outputs/models/linear.pkl", "wb"))

if __name__ == "__main__":
    train()
