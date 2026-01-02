"""
Tests for metrics.py evaluation functions.
"""

import torch
import sys
import os

# Add parent directory to path so we can import metrics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import mse, mae, rmse, r_squared, accuracy, evaluate_all, MetricsTracker


def test_mse():
    """Test Mean Squared Error calculation."""
    y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    # Perfect prediction should give MSE = 0
    assert abs(mse(y_pred, y_true) - 0.0) < 1e-6
    
    # Test with errors
    y_pred2 = torch.tensor([2.0, 3.0, 4.0, 5.0])
    result = mse(y_pred2, y_true)
    expected = 1.0  # ((1)^2 + (1)^2 + (1)^2 + (1)^2) / 4 = 1.0
    assert abs(result - expected) < 1e-6


def test_mae():
    """Test Mean Absolute Error calculation."""
    y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    # Perfect prediction
    assert abs(mae(y_pred, y_true) - 0.0) < 1e-6
    
    # Test with errors
    y_pred2 = torch.tensor([2.0, 3.0, 5.0, 6.0])
    result = mae(y_pred2, y_true)
    expected = 1.5  # (1 + 1 + 2 + 2) / 4 = 1.5
    assert abs(result - expected) < 1e-6


def test_rmse():
    """Test Root Mean Squared Error calculation."""
    y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    # Perfect prediction
    assert abs(rmse(y_pred, y_true) - 0.0) < 1e-6
    
    # Test with errors
    y_pred2 = torch.tensor([2.0, 3.0, 4.0, 5.0])
    result = rmse(y_pred2, y_true)
    expected = 1.0  # sqrt(1.0) = 1.0
    assert abs(result - expected) < 1e-6


def test_r_squared():
    """Test R² score calculation."""
    # Perfect prediction should give R² = 1.0
    y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    result = r_squared(y_pred, y_true)
    assert abs(result - 1.0) < 1e-6
    
    # Prediction equal to mean should give R² = 0.0
    y_pred2 = torch.tensor([2.5, 2.5, 2.5, 2.5])
    result2 = r_squared(y_pred2, y_true)
    assert abs(result2 - 0.0) < 1e-6


def test_accuracy():
    """Test accuracy calculation."""
    # Binary classification with threshold
    y_pred = torch.tensor([0.1, 0.6, 0.8, 0.3])
    y_true = torch.tensor([0.0, 1.0, 1.0, 0.0])
    
    result = accuracy(y_pred, y_true, threshold=0.5)
    expected = 1.0  # All predictions correct
    assert abs(result - expected) < 1e-6
    
    # Test with some errors
    y_pred2 = torch.tensor([0.6, 0.4, 0.8, 0.7])
    result2 = accuracy(y_pred2, y_true, threshold=0.5)
    expected2 = 0.25  # 1 out of 4 correct: [1,0,1,1] vs [0,1,1,0]
    assert abs(result2 - expected2) < 1e-6


def test_evaluate_all_regression():
    """Test evaluate_all for regression task."""
    y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    metrics = evaluate_all(y_pred, y_true, task_type="regression")
    
    assert "mse" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    
    # Perfect prediction
    assert abs(metrics["mse"] - 0.0) < 1e-6
    assert abs(metrics["mae"] - 0.0) < 1e-6
    assert abs(metrics["r2"] - 1.0) < 1e-6


def test_evaluate_all_classification():
    """Test evaluate_all for classification task."""
    y_pred = torch.tensor([0.1, 0.6, 0.8, 0.3])
    y_true = torch.tensor([0.0, 1.0, 1.0, 0.0])
    
    metrics = evaluate_all(y_pred, y_true, task_type="classification")
    
    assert "accuracy" in metrics
    assert "mae" in metrics
    assert "mse" in metrics
    
    # Should have perfect accuracy
    assert abs(metrics["accuracy"] - 1.0) < 1e-6


def test_metrics_tracker():
    """Test MetricsTracker functionality."""
    tracker = MetricsTracker()
    
    # Add some metrics
    tracker.update({"loss": 0.5, "accuracy": 0.8}, step=0)
    tracker.update({"loss": 0.4, "accuracy": 0.85}, step=1)
    tracker.update({"loss": 0.3, "accuracy": 0.9}, step=2)
    
    # Test get_latest
    latest = tracker.get_latest("loss")
    if isinstance(latest, tuple):
        assert abs(latest[1] - 0.3) < 1e-6
    else:
        assert abs(latest - 0.3) < 1e-6
    
    # Test get_mean
    mean_loss = tracker.get_mean("loss")
    expected_mean = (0.5 + 0.4 + 0.3) / 3
    assert abs(mean_loss - expected_mean) < 1e-6
    
    # Test get_mean with last_n
    mean_loss_last2 = tracker.get_mean("loss", last_n=2)
    expected_mean_last2 = (0.4 + 0.3) / 2
    assert abs(mean_loss_last2 - expected_mean_last2) < 1e-6
    
    # Test summary
    summary = tracker.summary()
    assert "loss" in summary
    assert "accuracy" in summary
    assert summary["loss"]["count"] == 3
    
    # Test reset
    tracker.reset()
    assert len(tracker.history) == 0


if __name__ == "__main__":
    # Run tests
    test_mse()
    test_mae()
    test_rmse()
    test_r_squared()
    test_accuracy()
    test_evaluate_all_regression()
    test_evaluate_all_classification()
    test_metrics_tracker()
    print("✅ All metrics tests passed!")
