def run():
    import joblib
    import numpy as np
    from sklearn.metrics import mean_absolute_error

    test = np.load("../data/dataset_test.npy")
    y_test = test[:, -1]

    pred = np.load("../outputs/models/final_pred.npy")

    mae = mean_absolute_error(y_test, pred)
    print("Final MAE:", mae)