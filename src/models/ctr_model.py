"""
CTR Prediction Model
Lightweight and fast CTR estimator for advertiser-side valuation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


class CTRModel:
    """
    Predicts pCTR = P(click = 1 | context) using a calibrated linear/logistic model.
    Designed for sub-millisecond inference during evolutionary policy evaluations.
    """

    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        max_iter: int = 300,
        random_state: int = 42,
    ):
        if categorical_features is None:
            self.categorical_features = [
                "ad_exchange",
                "ad_slot_visibility",
                "ad_slot_format",
                "user_tag",
                "time_bucket",
            ]
        else:
            self.categorical_features = categorical_features

        if numerical_features is None:
            self.numerical_features = ["ad_slot_floor_price", "hour"]
        else:
            self.numerical_features = numerical_features

        self.max_iter = max_iter
        self.random_state = random_state

        # Preprocessing pipeline with max_categories for high-cardinality safety
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(
                        max_categories=30,
                        handle_unknown="infrequent_if_exist",
                        sparse_output=False,
                    ),
                    self.categorical_features,
                ),
                (
                    "num",
                    StandardScaler(),
                    self.numerical_features,
                ),
            ]
        )

        self.model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            C=1.0,
        )
        self.pipeline = Pipeline([("pre", self.preprocessor), ("clf", self.model)])
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist in the DataFrame."""
        features_needed = self.categorical_features + self.numerical_features
        available_cols = [c for c in features_needed if c in df.columns]
        missing_cols = set(features_needed) - set(available_cols)
        
        feat_df = df[available_cols].copy()
        for col in missing_cols:
            if col in self.categorical_features:
                feat_df[col] = "unknown"
            else:
                feat_df[col] = 0.0
        return feat_df

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "CTRModel":
        """
        Train the CTR model on historical context features and click labels.
        If dataset exceeds 150k rows, samples uniformly to preserve natural base rate
        for calibrated posterior click probability estimation.
        """
        if len(df) > 150000:
            df_copy = df.copy()
            df_copy["_y_target"] = y
            sampled_df = df_copy.sample(n=150000, random_state=self.random_state)
            fit_df = sampled_df.drop(columns=["_y_target"])
            fit_y = sampled_df["_y_target"].values
        else:
            fit_df = df
            fit_y = y

        X = self._prepare_features(fit_df)
        self.pipeline.fit(X, fit_y)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of click for a batch of auction opportunities.
        Returns 1D array of floats in range [0, 1].
        """
        if not self.is_fitted:
            raise RuntimeError("CTRModel must be fitted before calling predict_proba.")
        X = self._prepare_features(df)
        # Class 1 probability
        probs = self.pipeline.predict_proba(X)[:, 1]
        # Safety clipping
        return np.clip(probs, 1e-4, 0.999)

    def evaluate(self, df: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate CTR model performance against ground-truth click outcomes.
        """
        preds = self.predict_proba(df)
        
        # Check if single class
        if len(np.unique(y)) > 1:
            auc = float(roc_auc_score(y, preds))
        else:
            auc = 0.5

        loss = float(log_loss(y, preds))
        brier = float(brier_score_loss(y, preds))
        mean_pred = float(np.mean(preds))
        actual_ctr = float(np.mean(y))

        return {
            "auc": auc,
            "log_loss": loss,
            "brier_score": brier,
            "mean_pctr": mean_pred,
            "actual_ctr": actual_ctr,
            "calibration_ratio": mean_pred / (actual_ctr + 1e-8),
        }
