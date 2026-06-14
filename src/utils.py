"""Utility functions for the ML pipeline."""

import numpy as np
from typing import Tuple


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        SMAPE score as percentage
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary with MAE, MSE, RMSE, R², and SMAPE
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    smape_score = smape(y_true, y_pred)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'smape': smape_score
    }
