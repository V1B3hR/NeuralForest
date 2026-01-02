"""
Evaluation metrics for NeuralForest.
Provides MSE, MAE, R², accuracy, and other performance metrics.
"""

import torch
import numpy as np
from typing import Union


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Mean Squared Error.

    Args:
        y_pred: Predicted values [B, ...]
        y_true: True values [B, ...]

    Returns:
        MSE value as float
    """
    return float(((y_pred - y_true) ** 2).mean().item())


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Mean Absolute Error.

    Args:
        y_pred: Predicted values [B, ...]
        y_true: True values [B, ...]

    Returns:
        MAE value as float
    """
    return float((y_pred - y_true).abs().mean().item())


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_pred: Predicted values [B, ...]
        y_true: True values [B, ...]

    Returns:
        RMSE value as float
    """
    return float(torch.sqrt(((y_pred - y_true) ** 2).mean()).item())


def r_squared(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    R² (coefficient of determination) score.
    R² = 1 - (SS_res / SS_tot)

    Args:
        y_pred: Predicted values [B, ...]
        y_true: True values [B, ...]

    Returns:
        R² value as float (1.0 = perfect fit, 0.0 = mean baseline, <0 = worse than mean)
    """
    # Residual sum of squares
    ss_res = ((y_true - y_pred) ** 2).sum()

    # Total sum of squares
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()

    # Prevent division by zero
    if ss_tot == 0:
        return 0.0

    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2.item())


def accuracy(
    y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Classification accuracy.
    For regression, uses threshold-based classification.
    For logits, applies sigmoid then threshold.

    Args:
        y_pred: Predicted values [B, ...] or logits
        y_true: True labels [B, ...] (0 or 1 for binary, integers for multi-class)
        threshold: Decision threshold (default: 0.5)

    Returns:
        Accuracy as float (0.0 to 1.0)
    """
    # If predictions are continuous, apply threshold
    if y_pred.dtype in [torch.float32, torch.float64]:
        # Apply sigmoid if values seem like logits (outside [0,1])
        if y_pred.min() < 0 or y_pred.max() > 1:
            y_pred = torch.sigmoid(y_pred)
        y_pred_binary = (y_pred >= threshold).float()
    else:
        y_pred_binary = y_pred.float()

    # Convert true labels to same type
    y_true_binary = y_true.float()

    # Compute accuracy
    correct = (y_pred_binary == y_true_binary).float().sum()
    total = float(y_true.numel())

    return float((correct / total).item())


def evaluate_all(
    y_pred: torch.Tensor, y_true: torch.Tensor, task_type: str = "regression"
) -> dict:
    """
    Compute all relevant metrics for a given task.

    Args:
        y_pred: Predicted values [B, ...]
        y_true: True values [B, ...]
        task_type: "regression" or "classification"

    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {}

    if task_type == "regression":
        metrics["mse"] = mse(y_pred, y_true)
        metrics["mae"] = mae(y_pred, y_true)
        metrics["rmse"] = rmse(y_pred, y_true)
        metrics["r2"] = r_squared(y_pred, y_true)
    elif task_type == "classification":
        metrics["accuracy"] = accuracy(y_pred, y_true)
        # Also include MAE and MSE for completeness
        metrics["mae"] = mae(y_pred, y_true)
        metrics["mse"] = mse(y_pred, y_true)
    else:
        raise ValueError(
            f"Unknown task_type: {task_type}. Use 'regression' or 'classification'."
        )

    return metrics


class MetricsTracker:
    """
    Tracks metrics over training/evaluation steps.
    """

    def __init__(self):
        self.history = {}

    def update(self, metrics: dict, step: int = None):
        """
        Add metrics for current step.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
        """
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []

            if step is not None:
                self.history[name].append((step, value))
            else:
                self.history[name].append(value)

    def get_latest(self, metric_name: str) -> Union[float, None]:
        """Get most recent value for a metric."""
        if metric_name not in self.history or len(self.history[metric_name]) == 0:
            return None
        return self.history[metric_name][-1]

    def get_mean(self, metric_name: str, last_n: int = None) -> Union[float, None]:
        """
        Get mean of a metric over history.

        Args:
            metric_name: Name of metric
            last_n: If provided, compute mean over last N values only
        """
        if metric_name not in self.history or len(self.history[metric_name]) == 0:
            return None

        values = self.history[metric_name]
        if last_n is not None:
            values = values[-last_n:]

        # Handle both (step, value) tuples and plain values
        if isinstance(values[0], tuple):
            values = [v[1] for v in values]

        return float(np.mean(values))

    def get_history(self, metric_name: str) -> list:
        """Get full history for a metric."""
        return self.history.get(metric_name, [])

    def reset(self):
        """Clear all metrics history."""
        self.history = {}

    def summary(self) -> dict:
        """
        Get summary statistics for all tracked metrics.

        Returns:
            Dictionary with summary stats for each metric
        """
        summary = {}
        for name, values in self.history.items():
            # Handle both (step, value) tuples and plain values
            if values and isinstance(values[0], tuple):
                vals = [v[1] for v in values]
            else:
                vals = values

            if vals:
                summary[name] = {
                    "latest": vals[-1],
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "count": len(vals),
                }

        return summary
