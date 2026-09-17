"""
Unit Tests for CTR Prediction Model
"""

import numpy as np
import pandas as pd
import pytest
from src.data.synthetic import generate_synthetic_rtb_data
from src.models.ctr_model import CTRModel


def test_ctr_model_train_and_predict():
    df = generate_synthetic_rtb_data(n_samples=2000, n_days=7, random_seed=42)
    train_df = df[df["day"] <= 5]
    test_df = df[df["day"] > 5]

    ctr_model = CTRModel()
    ctr_model.fit(train_df, train_df["click"].values)

    preds = ctr_model.predict_proba(test_df)
    assert len(preds) == len(test_df)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()
    # Predictions should have reasonable variance (not constant)
    assert np.std(preds) > 1e-4

    metrics = ctr_model.evaluate(test_df, test_df["click"].values)
    assert "auc" in metrics
    assert "log_loss" in metrics
    # AUC should be better than random guess (> 0.55)
    assert metrics["auc"] >= 0.55
