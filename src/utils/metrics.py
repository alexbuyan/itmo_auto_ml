"""Evaluation metrics for regression models."""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_log_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE on log-transformed values."""
    y_pred = np.maximum(y_pred, 0)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    return float(np.mean(np.abs(log_true - log_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics: mae, log_mae, rmse, mae_pct."""
    y_pred = np.maximum(y_pred, 0)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "log_mae": compute_log_mae(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae_pct": mean_absolute_error(y_true, y_pred) / np.mean(y_true) * 100,
    }
