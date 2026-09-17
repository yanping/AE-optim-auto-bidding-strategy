"""
Evaluation Metrics for RTB Models
Includes calibration, survival analysis goodness-of-fit, and bidding accuracy.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import brier_score_loss, roc_auc_score


def compute_win_rate_mse(estimated_win_prob: np.ndarray, actual_win: np.ndarray) -> float:
    """Mean Squared Error between estimated win probability and actual binary win."""
    return float(np.mean((estimated_win_prob - actual_win) ** 2))


def compute_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Standard Brier Score for probability calibration."""
    return float(brier_score_loss(labels, probabilities))


def compute_auc_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC score for binary outcomes."""
    if len(np.unique(labels)) <= 1:
        return 0.5
    return float(roc_auc_score(labels, probabilities))


def evaluate_calibration_curve(
    predicted_probs: np.ndarray,
    ground_truth_labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes calibration curve: binned predicted vs empirical observed rates.
    Returns:
        prob_pred: Mean predicted probability per bin.
        prob_true: Empirical fraction of positive outcomes per bin.
        ece: Expected Calibration Error.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(predicted_probs, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    prob_pred = []
    prob_true = []
    bin_weights = []

    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) > 0:
            prob_pred.append(np.mean(predicted_probs[mask]))
            prob_true.append(np.mean(ground_truth_labels[mask]))
            bin_weights.append(np.sum(mask) / len(predicted_probs))

    prob_pred_arr = np.array(prob_pred)
    prob_true_arr = np.array(prob_true)
    weights_arr = np.array(bin_weights)

    # Expected Calibration Error
    ece = float(np.sum(weights_arr * np.abs(prob_pred_arr - prob_true_arr)))
    return prob_pred_arr, prob_true_arr, ece
