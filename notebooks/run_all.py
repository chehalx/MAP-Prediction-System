
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_linear, model_rf, model_ridge, model_et, model_xgb, model_lgbm, model_cnn

print("=" * 55)
print("  Training all models")
print("=" * 55)
for m in [model_linear, model_rf, model_ridge, model_et, model_xgb, model_lgbm, model_cnn]:
    m.train()
print("=" * 55)
print("  Done. Run predict.py for test evaluation.")
print("=" * 55)
