
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

FS         = 100
TRAIN_PATH = "../data/dataset_train.npy"
TEST_PATH  = "../data/dataset_test.npy"

def bandpass(X, lo, hi, order=3):
    b, a = butter(order, [lo/(FS/2), hi/(FS/2)], btype="band")
    return filtfilt(b, a, X, axis=1).astype(np.float32)

def extract_hr(X):
    lo, hi = int(FS*60/200), int(FS*60/30)
    hrs = []
    for s in X:
        s  = s - s.mean()
        ac = np.correlate(s, s, mode="full")[len(s):]
        ac /= (ac[0] + 1e-8)
        hrs.append(60.0 * FS / (np.argmax(ac[lo:hi]) + lo))
    return np.array(hrs, dtype=np.float32)

def make_features(X, Xc, Xm, hr):
    ac, am = Xc.max(1)-Xc.min(1), Xm.max(1)-Xm.min(1)
    h  = Xc.shape[1] // 2
    mu = X.mean(1, keepdims=True)
    Xn = (X - mu) / (np.sqrt(((X-mu)**2).mean(1, keepdims=True)) + 1e-8)

    feats = [
        # Raw
        X.std(1), X.mean(1), X.max(1)-X.min(1),
        # Cardiac
        Xc.std(1), ac, np.sqrt((Xc**2).mean(1)),
        np.percentile(Xc,10,1), np.percentile(Xc,90,1),
        skew(Xc,1).astype(np.float32), kurtosis(Xc,1).astype(np.float32),
        (Xc[:,:h].max(1)-Xc[:,:h].min(1)) / (Xc[:,h:].max(1)-Xc[:,h:].min(1)+1e-8),
        # Motion
        Xm.std(1), am, np.sqrt((Xm**2).mean(1)),
        skew(Xm,1).astype(np.float32), kurtosis(Xm,1).astype(np.float32),
        # HR-derived
        hr, hr**2, np.log1p(hr),
        ac/(hr+1e-8), ac*(60./(hr+1e-8)), hr*ac, hr*Xc.std(1), Xm.std(1)*ac,
        # Stationarity
        *[(srms := np.column_stack([np.sqrt((s**2).mean(1)) for s in np.split(Xn,5,1)]))[:,i]/(srms[:,0]+1e-8) for i in range(1,5)],
        np.sqrt((Xn[:,:200]**2).mean(1)) / (np.sqrt((Xn[:,800:]**2).mean(1))+1e-8),
        # Spectral
        *(lambda pc,pm: [pc.mean(1), pm.mean(1), pm.mean(1)/(pc.mean(1)+1e-8)])(
            np.abs(np.fft.rfft(Xc,axis=1))**2, np.abs(np.fft.rfft(Xm,axis=1))**2),
    ]
    return np.nan_to_num(np.column_stack(feats).astype(np.float32))

def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train, test = np.load(train_path), np.load(test_path)
    X_tr, y_tr  = train[:,:1000].astype(np.float32), train[:,-1].astype(np.float32)
    X_te, y_te  = test[:, :1000].astype(np.float32), test[:, -1].astype(np.float32)
    print(f"  Train {X_tr.shape} | Test {X_te.shape} | MAP {y_tr.mean():.1f}±{y_tr.std():.1f}")
    F_tr = make_features(X_tr, bandpass(X_tr,0.5,8), bandpass(X_tr,8,20), extract_hr(X_tr))
    F_te = make_features(X_te, bandpass(X_te,0.5,8), bandpass(X_te,8,20), extract_hr(X_te))
    sc   = StandardScaler()
    return sc.fit_transform(F_tr), sc.transform(F_te), y_tr, y_te, sc
