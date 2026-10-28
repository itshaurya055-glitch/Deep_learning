import os
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.config import XGB_PARAMS, MODELS_DIR

class XGBoostDetector:
    def __init__(self, params: dict = None):
        self.params = params if params else XGB_PARAMS.copy()
        self.model = None
        self.threshold = 0.5

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_pos_weight = num_neg / max(1, num_pos)
        
        print(f"Training XGBoostDetector with scale_pos_weight = {scale_pos_weight:.2f}...")
        self.params["scale_pos_weight"] = scale_pos_weight
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X_train, y_train)

        # Calibrate optimal decision threshold on training set
        train_probs = self.predict_proba(X_train)
        best_f1, best_thresh = 0.0, 0.5
        for th in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(y_train, (train_probs >= th).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, th
        self.threshold = float(best_thresh)
        print(f"Calibrated XGBoost decision threshold to {self.threshold:.3f} (Train F1: {best_f1:.4f})")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        probs = self.model.predict_proba(X)[:, 1]
        return probs

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(int)

    def predict_step(self, step_features: pd.DataFrame, payer_account: str = None):
        prob = float(self.predict_proba(step_features)[0])
        is_blocked = int(prob >= self.threshold)
        return prob, is_blocked

    def save(self, filename: str = "xgboost_detector.json"):
        path = os.path.join(MODELS_DIR, filename)
        self.model.save_model(path)
        print(f"Saved XGBoostDetector to {path}")

    def load(self, filename: str = "xgboost_detector.json"):
        path = os.path.join(MODELS_DIR, filename)
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        print(f"Loaded XGBoostDetector from {path}")
