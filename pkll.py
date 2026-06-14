"""
Train 3 models for Smart Ride Demand Prediction:
  1. nyc_model.pkl -> trained on NYC data
  2. blr_model.pkl -> trained on Bangalore data
  3. adapted_model.pkl -> NYC model fine-tuned on Bangalore (transfer learning)

Run: python train_models.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_COLS = ["hour", "dayofweek", "is_weekend", "lag_1", "lag_2", "city"]
TARGET = "demand"


def evaluate(name, model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f" [{name}] MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")
    return mae, rmse, r2


def main():
    print("=" * 60)
    print("Smart Ride Demand — Training Pipeline")
    print("=" * 60)

    # ---------- Load data ----------
    print("\n[1/4] Loading datasets...")
    nyc = pd.read_csv("final_features_nyc.csv")
    blr = pd.read_csv("final_features_blr.csv")
    print(f" NYC rows: {len(nyc):,}")
    print(f" BLR rows: {len(blr):,}")

    X_nyc, y_nyc = nyc[FEATURE_COLS], nyc[TARGET]
    X_blr, y_blr = blr[FEATURE_COLS], blr[TARGET]

    # Train/test splits
    Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
        X_nyc, y_nyc, test_size=0.2, random_state=42
    )
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
        X_blr, y_blr, test_size=0.2, random_state=42
    )

    # ---------- Model 1: NYC ----------
    print("\n[2/4] Training NYC model...")
    nyc_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    nyc_model.fit(Xn_tr, yn_tr)
    evaluate("NYC on NYC", nyc_model, Xn_te, yn_te)
    evaluate("NYC on BLR (transfer baseline)", nyc_model, Xb_te, yb_te)
    joblib.dump(nyc_model, "nyc_model.pkl")
    print(" Saved -> nyc_model.pkl")

    # ---------- Model 2: BLR (from scratch) ----------
    print("\n[3/4] Training BLR model (from scratch)...")
    blr_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    blr_model.fit(Xb_tr, yb_tr)
    evaluate("BLR on BLR", blr_model, Xb_te, yb_te)
    joblib.dump(blr_model, "blr_model.pkl")
    print(" Saved -> blr_model.pkl")

    # ---------- Model 3: Adapted (Transfer Learning) ----------
    print("\n[4/4] Training Adapted model (NYC -> BLR transfer)...")
    # Strategy: warm-start the NYC model with additional boosting rounds on BLR data.
    # This simulates fine-tuning: we keep NYC's learned trees and add BLR-specific trees.
    adapted_model = GradientBoostingRegressor(
        n_estimators=200, # original NYC trees
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        warm_start=True,
    )
    adapted_model.fit(Xn_tr, yn_tr) # learn NYC patterns first
    adapted_model.n_estimators = 350 # add 150 new trees
    adapted_model.fit(Xb_tr, yb_tr) # fine-tune on BLR
    evaluate("Adapted on BLR", adapted_model, Xb_te, yb_te)
    joblib.dump(adapted_model, "adapted_model.pkl")
    print(" Saved -> adapted_model.pkl")

    print("\n" + "=" * 60)
    print("Done! Three .pkl files created in current folder:")
    print(" - nyc_model.pkl")
    print(" - blr_model.pkl")
    print(" - adapted_model.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()