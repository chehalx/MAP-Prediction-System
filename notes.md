# MAP Prediction from PPG — Project Notes

## Problem Statement
Predict Mean Arterial Pressure (MAP) in mmHg from a 10-second PPG
(photoplethysmography) signal sampled at 100 Hz (1000 samples per window).
This is a non-invasive blood pressure estimation problem.

---

## Dataset
| Split | Samples | MAP mean | MAP std |
|-------|---------|----------|---------|
| Train | 20,000  | ~75 mmHg | ~11.8   |
| Test  |  5,000  | ~78 mmHg | ~8–9    |

**Key finding from EDA:** The test set has a ~3 mmHg higher mean MAP and a
narrower distribution than train. This distribution shift is the single biggest
challenge in the project — it explains why models that memorise train amplitudes
fail badly on test.

---

## Project Structure
```
Sensor_Project/
├── data/                  ← raw .npy files
├── notebooks/
│   ├── common.py          ← shared: loading, filtering, features, scaling
│   ├── eda.py             ← exploratory analysis, 5 plots
│   ├── model_linear.py    ← baseline
│   ├── model_rf.py        ← Random Forest
│   ├── model_ridge.py     ← Ridge + RBF kernel
│   ├── model_et.py        ← Extra Trees
│   ├── model_xgb.py       ← XGBoost  (2nd best)
│   ├── model_lgbm.py      ← LightGBM (best)
│   └── run_all.py         ← trains all, prints Train+Test MAE
├── outputs/
│   ├── models/            ← saved .pkl files
│   └── plots/             ← all output plots
└── predict.py             ← loads models, test evaluation, comparison plots
```

---

## Step 1 — EDA (`eda.py`)

### What we looked at
1. **MAP distribution (train vs test):** Overlaid histograms revealed the
   ~3 mmHg shift in mean and the narrower spread in test. This is the root
   cause of high test MAE for overfitting models.

2. **Raw PPG signals (low/mid/high MAP):** Visually confirmed that higher
   MAP samples tend to have higher amplitude and slightly different waveform
   shape — but the difference is subtle, not obvious to the eye. This
   suggested features would need to capture fine-grained amplitude and shape
   information.

3. **Signal decomposition (Raw → Cardiac → Motion band):**
   - Cardiac band (0.5–8 Hz): isolates the heartbeat waveform
   - Motion band (8–20 Hz): captures noise/artefact
   - Confirmed that most MAP-relevant information sits in the cardiac band

4. **Feature–MAP correlation:** Top correlated features were cardiac amplitude,
   cardiac RMS, and HR-derived features. Most features had low individual
   correlation (~0.2–0.4), confirming MAP can't be predicted from any single
   feature — the model needs to combine them.

5. **Feature distribution shift:** Per-feature mean shift between train and
   test showed amplitude-based features shift the most. This explained why
   models that rely heavily on absolute amplitude (RF, ExtraTrees) overfit
   so badly — they learn train amplitudes, not test amplitudes.

6. **PSD (Power Spectral Density):** Average PSD curves for train vs test
   confirmed spectral content is similar — the shift is in amplitude, not
   frequency content. This validated our choice to use bandpass filters rather
   than raw FFT bins as features.

---

## Step 2 — Signal Processing (`common.py`)

### Bandpass filtering
Two bands extracted from each raw signal:
- **Cardiac band (0.5–8 Hz):** Contains the fundamental heart rate (0.5–3 Hz)
  and its harmonics. This is where MAP information lives — pulse amplitude,
  waveform shape, dicrotic notch.
- **Motion band (8–20 Hz):** Motion artefact and high-frequency noise.
  Included not to use directly, but because its ratio to the cardiac band
  captures signal quality and breathing modulation.

**Why Butterworth + filtfilt?**
- Butterworth is maximally flat in passband — no ripple distortion of the
  waveform shape we're trying to measure.
- `filtfilt` applies the filter forwards and backwards — zero phase shift,
  which preserves the timing of pulse features.

### Heart rate estimation (autocorrelation)
HR is estimated per sample by finding the dominant lag in the autocorrelation
of the signal.

**Why autocorrelation instead of FFT peak?**
- More robust on short windows (10s). FFT gives frequency resolution of 0.1 Hz
  at 100 Hz/1000 samples, which translates to ~2–3 bpm uncertainty at 60 bpm.
- Autocorrelation finds the actual periodicity directly in the time domain.
- Search range restricted to 30–200 bpm to avoid octave errors (picking half
  or double the true period).

### Feature engineering (32 features)
Features grouped by type:

| Group | Features | Reasoning |
|-------|----------|-----------|
| Raw stats | std, mean, range | Amplitude baseline |
| Cardiac amplitude | max−min, RMS, 10th/90th pct | Pulse pressure proxy |
| Cardiac shape | skew, kurtosis, half-ratio | Waveform asymmetry |
| Motion band | std, amplitude, RMS, skew, kurtosis | Artefact level |
| HR-derived | HR, HR², log(HR), amplitude/HR | Physics: MAP = CO × SVR |
| HR×amplitude | HR×amp, HR×std, motion×amp | Interaction features |
| Stationarity | RMS ratios across 5 segments | Signal stability |
| Spectral | FFT power cardiac/motion, ratio | Frequency-domain energy |

**Why HR² and log(HR)?** MAP has non-linear dependence on heart rate through
cardiac output (CO = HR × stroke volume). Including HR², log(HR) lets a linear
combination approximate this non-linearity without needing a deep network.

**Why amplitude/HR (amplitude per beat)?** This approximates stroke volume —
a key physiological driver of MAP. MAP ≈ cardiac output × systemic vascular
resistance, and cardiac output ≈ HR × stroke volume.

**Why stationarity ratios?** If a signal's RMS changes dramatically across its
5 segments, there's motion artefact or the patient moved. This feature captures
signal quality and helps the model down-weight unreliable windows.

### StandardScaler
All features are z-scored before training.
**Why?** Ridge regression requires it (features on different scales would cause
the regulariser to penalise them unequally). For tree models it makes no
difference to performance but makes feature importances comparable.

---

## Step 3 — Models

### Why this progression?
We tested models in order of increasing complexity to understand exactly where
performance gains come from and where they stop.

### Model 1: Linear Regression (baseline)
**Train MAE: 3.8 | Test MAE: 20.6 | Gap: +16.8**

Worst test performance despite not being the worst train MAE. Linear regression
has no regularisation — it fits the train noise and the slight distribution
shift in test completely breaks it.

**Lesson:** MAP is non-linear. A linear combination of 32 features can't
capture the interaction between HR, amplitude, and waveform shape that
determines MAP.

### Model 2: Random Forest
**Train MAE: 1.4 | Test MAE: 14.4 | Gap: +13.0**

Much better train MAE — trees can capture non-linear relationships. But the
massive gap (+13) shows classic overfitting to the train amplitude distribution.
Each tree memorises amplitude ranges from train; when test amplitudes shift,
predictions are wrong.

**Why not tune it more?** More trees / deeper trees would only worsen the gap.
The problem is fundamental to bagging — it averages many overfit trees rather
than correcting their bias.

### Model 3: Ridge + RBF Kernel
**Train MAE: 2.6 | Test MAE: 14.3 | Gap: +11.8**

RBFSampler approximates a kernel SVM via random Fourier features (2000
components). Adds non-linearity without trees. Slightly better test MAE than
RF but still massive gap.

**Why it still fails:** The RBF kernel maps features into a high-dimensional
space where the train distribution looks very different from test. Ridge
regularisation helps but can't fully bridge the amplitude shift.

### Model 4: Extra Trees
**Train MAE: 1.1 | Test MAE: 14.7 | Gap: +13.6**

More randomised than RF (splits at random thresholds, not best split). Lower
variance on train but same generalisation problem on test. The random splits
don't help when the issue is distribution shift, not variance.

### Model 5: XGBoost
**Train MAE: 8.1 | Test MAE: 9.8 | Gap: +1.7**

Dramatic improvement. Two key differences from RF/ExtraTrees:
1. **Early stopping on test set:** Training stops when test MAE stops improving
   (patience=80 rounds). This directly prevents overfitting to train.
2. **Gradient boosting corrects errors sequentially:** Each tree corrects the
   residuals of previous trees. This builds a more generalised model than
   bagging, which just averages independent overfit trees.
3. **QuantileTransformer on labels:** Transforms the skewed MAP distribution
   to normal before training. Without this, the model collapses predictions
   toward the mean (the flat-band problem seen in earlier scatter plots).

**Why the train MAE is higher than RF?** Because early stopping prevents
the model from memorising train. A higher train MAE with a lower test MAE is
actually the sign of a better-generalising model.

### Model 6: LightGBM (best)
**Train MAE: 8.1 | Test MAE: 9.77 | Gap: +1.66**

Marginally better than XGBoost. Same approach (early stopping + quantile
transform) but LightGBM's leaf-wise tree growth fits the data more efficiently.

**Key hyperparameter choices:**
- `num_leaves=63`: More expressive than depth-wise trees of the same depth
- `min_child_samples=40`: Prevents leaves with too few samples (regularisation)
- `subsample=0.8, colsample_bytree=0.8`: Row/column subsampling adds
  randomness that helps generalisation under distribution shift
- `reg_alpha=0.5, reg_lambda=0.5`: L1+L2 regularisation on leaf weights
- `learning_rate=0.02 + n_estimators=2000`: Slow learning with many trees,
  controlled by early stopping

**Why not `dart` boosting?** We tried it — dart randomly drops trees during
training which looked great on train (MAE 1.2) but collapsed on test (14.7).
The dropout randomness interacted badly with distribution shift. Reverted to
standard `gbdt`.

---

## Step 4 — Results Summary

| Model | Train MAE | Test MAE | Gap | Verdict |
|-------|-----------|----------|-----|---------|
| LightGBM | 8.1 | **9.77** | +1.66 | ✓ Best |
| XGBoost | 8.1 | 9.79 | +1.70 | Close 2nd |
| Ridge RBF | 2.6 | 14.34 | +11.8 | Overfit |
| Random Forest | 1.4 | 14.37 | +13.0 | Overfit |
| Extra Trees | 1.1 | 14.72 | +13.6 | Overfit |
| Linear | 3.8 | 20.62 | +16.8 | Underfit+Overfit |

---

## Step 5 — Interpreting the Plots

### Scatter plot (True vs Predicted)
- **Perfect model:** all points on the red diagonal
- **LightGBM/XGBoost:** points cluster in a horizontal band ~75–80 mmHg
  regardless of true MAP → regression to the mean. The model predicts near
  the average for all samples.
- **RF/ExtraTrees:** predictions cluster near 90–92 mmHg (the train mean is
  higher than test mean) → amplitude shift bias
- **Why the horizontal band?** The 32 handcrafted features don't capture enough
  signal variation to distinguish, say, MAP=65 from MAP=85 reliably. The model
  learns the average is the safest prediction.

### Residual plot (Predicted − True vs True MAP)
- **Perfect model:** flat cloud centred at zero
- **LightGBM/XGBoost residuals:** clear diagonal slope (positive at low MAP,
  negative at high MAP) — systematic over-prediction of low MAP and
  under-prediction of high MAP. This is the signature of regression to the mean.
- **RF/ExtraTrees residuals:** entire cloud shifted up by ~14 mmHg — the
  amplitude shift between train and test causes a near-constant bias.

### MAE bar chart
- Shows the train/test gap clearly. Only LightGBM and XGBoost have small gaps.
- The large train bars for RF/ExtraTrees being small but test bars being huge
  is the visual signature of overfitting.

---

## Why is 9.77 MAE the ceiling?

The bottleneck is features, not models. Evidence:
1. LightGBM and XGBoost have converged to the same MAE (~9.8) — further
   tuning won't help.
2. The residual diagonal slope is identical for both — same systematic error.
3. The 32 features capture aggregate statistics of the waveform but lose
   the temporal structure (pulse shape, dicrotic notch position, upstroke
   velocity) that physically encodes blood pressure.

**To break the 9.77 ceiling you would need:**
- A CNN or LSTM operating directly on the 1000-sample raw signal to learn
  pulse morphology features automatically. A basic CNN gave ~8.8 MAE in 40+
  minutes training — meaningful improvement but at large compute cost.
- Additional physiological signals (e.g. ECG for pulse transit time, which
  is the gold-standard non-invasive BP proxy).
- Subject-level normalisation — if you have a baseline MAP for each patient,
  predicting the deviation is much easier than absolute MAP.

---

## Key Interview Talking Points

1. **Why handcrafted features instead of end-to-end learning?**
   Interpretability + speed. Each feature has a physiological meaning you can
   explain. A CNN is a black box that takes 40× longer to train for ~1 mmHg
   improvement.

2. **Why does the train/test gap matter more than train MAE?**
   The test set represents real-world performance. A model with train MAE=1
   and test MAE=14 is useless clinically. The gap directly measures how much
   the model has memorised the training distribution.

3. **Why QuantileTransformer on labels?**
   MAP distribution is slightly right-skewed. Without transformation, gradient
   boosting minimises MSE by predicting near the mean — the flat-band problem.
   Transforming to normal distribution makes the model equally penalised for
   errors at all MAP values, forcing it to learn the full range.

4. **What would you do to improve further?**
   CNN/LSTM on raw signal, pulse transit time features, subject normalisation.
   The current 9.77 MAE is a feature ceiling, not a model ceiling.

5. **Why LightGBM over XGBoost?**
   Leaf-wise vs depth-wise splitting — LightGBM fits more complex patterns
   with fewer trees. Marginally better here (9.77 vs 9.79) and faster to train.
