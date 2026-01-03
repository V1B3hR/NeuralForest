"""
Phase 4: Automated Learning, Testing, and Benchmarking

This module implements the AutoML forest, continuous generalization tests,
automated regression testing, and alert/metric checking systems.
"""

from __future__ import annotations

import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result from an automated test."""
    
    test_name: str
    passed: bool
    score: float
    threshold: float
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    
    benchmark_name: str
    metric_name: str
    value: float
    baseline: Optional[float]
    improvement: Optional[float]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ContinuousGeneralizationTester:
    """
    Performs continuous generalization testing on out-of-sample data.
    
    Features:
    - Periodic evaluation on held-out data
    - Unseen data simulation
    - Generalization metrics tracking
    """
    
    def __init__(
        self,
        test_frequency: int = 100,
        ood_data_generators: Optional[List[Callable]] = None
    ):
        """
        Initialize generalization tester.
        
        Args:
            test_frequency: How often to run tests (in steps)
            ood_data_generators: Functions to generate out-of-distribution data
        """
        self.test_frequency = test_frequency
        self.ood_data_generators = ood_data_generators or []
        self.test_history = []
        self.step_counter = 0
        
    def should_test(self) -> bool:
        """Check if it's time to run tests."""
        return self.step_counter % self.test_frequency == 0
    
    def test_generalization(
        self,
        forest,
        test_data: List[Tuple[torch.Tensor, torch.Tensor]],
        test_name: str = "generalization"
    ) -> TestResult:
        """
        Test forest generalization on held-out data.
        
        Args:
            forest: ForestEcosystem instance
            test_data: List of (input, target) tuples
            test_name: Name of the test
            
        Returns:
            TestResult
        """
        forest.eval()
        
        total_loss = 0.0
        total_samples = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for x, y in test_data:
                x = x.to(forest.device) if hasattr(forest, 'device') else x
                y = y.to(forest.device) if hasattr(forest, 'device') else y
                
                pred, _, _ = forest.forward_forest(x)
                loss = criterion(pred, y)
                
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        # Lower loss is better, threshold could be dynamic
        threshold = 1.0
        passed = avg_loss < threshold
        
        result = TestResult(
            test_name=test_name,
            passed=passed,
            score=avg_loss,
            threshold=threshold,
            timestamp=time.time(),
            details={
                "num_samples": total_samples,
                "num_batches": len(test_data),
            }
        )
        
        self.test_history.append(result)
        forest.train()
        
        return result
    
    def test_ood_robustness(
        self,
        forest,
        input_shape: Tuple[int, ...],
        num_samples: int = 100
    ) -> List[TestResult]:
        """
        Test forest on out-of-distribution data.
        
        Args:
            forest: ForestEcosystem instance
            input_shape: Shape of input data
            num_samples: Number of OOD samples to test
            
        Returns:
            List of TestResults
        """
        results = []
        
        for i, generator in enumerate(self.ood_data_generators):
            try:
                # Generate OOD data
                ood_data = generator(input_shape, num_samples)
                
                # Test on OOD data
                result = self.test_generalization(
                    forest,
                    ood_data,
                    test_name=f"ood_test_{i}"
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"OOD test {i} failed: {e}")
        
        return results
    
    def step(self):
        """Increment step counter."""
        self.step_counter += 1
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        if not self.test_history:
            return {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
            }
        
        passed = sum(1 for r in self.test_history if r.passed)
        failed = len(self.test_history) - passed
        
        recent_scores = [r.score for r in self.test_history[-10:]]
        
        return {
            "total_tests": len(self.test_history),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.test_history),
            "recent_avg_score": sum(recent_scores) / len(recent_scores) if recent_scores else 0,
            "recent_tests": [r.to_dict() for r in self.test_history[-5:]],
        }


class RegressionValidator:
    """
    Detects performance regressions and validates improvements.
    
    Features:
    - Baseline tracking
    - Regression detection
    - Validation checkpoints
    """
    
    def __init__(
        self,
        regression_threshold: float = 0.1,
        checkpoint_frequency: int = 1000
    ):
        """
        Initialize regression validator.
        
        Args:
            regression_threshold: Acceptable performance drop (fraction)
            checkpoint_frequency: How often to create checkpoints
        """
        self.regression_threshold = regression_threshold
        self.checkpoint_frequency = checkpoint_frequency
        self.baselines: Dict[str, float] = {}
        self.checkpoints: List[Dict[str, Any]] = []
        self.regressions_detected = []
        self.step_counter = 0
        
    def set_baseline(self, metric_name: str, value: float):
        """
        Set a baseline value for a metric.
        
        Args:
            metric_name: Name of the metric
            value: Baseline value
        """
        self.baselines[metric_name] = value
        logger.info(f"Baseline set for {metric_name}: {value:.4f}")
    
    def check_regression(
        self,
        metric_name: str,
        current_value: float,
        lower_is_better: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if there's a regression for a metric.
        
        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            lower_is_better: Whether lower values are better
            
        Returns:
            (is_regression, message)
        """
        if metric_name not in self.baselines:
            # No baseline yet, set it
            self.set_baseline(metric_name, current_value)
            return False, None
        
        baseline = self.baselines[metric_name]
        
        if lower_is_better:
            # For metrics like loss, lower is better
            change = (current_value - baseline) / baseline if baseline != 0 else 0
            is_regression = change > self.regression_threshold
        else:
            # For metrics like accuracy, higher is better
            change = (baseline - current_value) / baseline if baseline != 0 else 0
            is_regression = change > self.regression_threshold
        
        if is_regression:
            message = f"⚠️ Regression detected in {metric_name}: {baseline:.4f} → {current_value:.4f} ({change*100:.1f}% worse)"
            self.regressions_detected.append({
                "metric": metric_name,
                "baseline": baseline,
                "current": current_value,
                "change": change,
                "timestamp": time.time(),
            })
            return True, message
        
        # Update baseline if improved significantly
        if lower_is_better and current_value < baseline * 0.95:
            self.set_baseline(metric_name, current_value)
        elif not lower_is_better and current_value > baseline * 1.05:
            self.set_baseline(metric_name, current_value)
        
        return False, None
    
    def create_checkpoint(self, state: Dict[str, Any]):
        """
        Create a validation checkpoint.
        
        Args:
            state: State to checkpoint
        """
        checkpoint = {
            "step": self.step_counter,
            "timestamp": time.time(),
            "metrics": state.get("metrics", {}),
            "baselines": self.baselines.copy(),
        }
        
        self.checkpoints.append(checkpoint)
        logger.info(f"Checkpoint created at step {self.step_counter}")
    
    def should_checkpoint(self) -> bool:
        """Check if it's time to create a checkpoint."""
        return self.step_counter % self.checkpoint_frequency == 0
    
    def step(self):
        """Increment step counter."""
        self.step_counter += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of regression validation."""
        return {
            "total_checkpoints": len(self.checkpoints),
            "total_regressions": len(self.regressions_detected),
            "current_baselines": self.baselines.copy(),
            "recent_regressions": self.regressions_detected[-5:],
        }


class MetricAlerter:
    """
    Monitors metrics and triggers alerts based on thresholds.
    
    Features:
    - Threshold-based alerts
    - Custom alert rules
    - Alert callbacks
    """
    
    def __init__(self, alert_rules: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize metric alerter.
        
        Args:
            alert_rules: Dictionary of metric_name -> rule_config
        """
        self.alert_rules = alert_rules or {}
        self.alerts: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable[[str, Dict], None]] = []
        
        # Default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        defaults = {
            "diversity": {
                "type": "threshold",
                "operator": "less_than",
                "threshold": 0.1,
                "message": "Diversity critically low",
                "severity": "critical",
            },
            "alive_trees": {
                "type": "threshold",
                "operator": "less_than",
                "threshold": 3,
                "message": "Population too small",
                "severity": "critical",
            },
            "avg_fitness": {
                "type": "stagnation",
                "window": 10,
                "threshold": 0.01,
                "message": "Fitness stagnating",
                "severity": "warning",
            },
        }
        
        for metric, rule in defaults.items():
            if metric not in self.alert_rules:
                self.alert_rules[metric] = rule
    
    def check_metrics(self, metrics: Dict[str, float]):
        """
        Check metrics against alert rules.
        
        Args:
            metrics: Dictionary of metric values
        """
        for metric_name, value in metrics.items():
            if metric_name not in self.alert_rules:
                continue
            
            rule = self.alert_rules[metric_name]
            triggered = self._check_rule(metric_name, value, rule)
            
            if triggered:
                self._trigger_alert(metric_name, value, rule)
    
    def _check_rule(self, metric_name: str, value: float, rule: Dict[str, Any]) -> bool:
        """Check if a rule is triggered."""
        rule_type = rule.get("type", "threshold")
        
        if rule_type == "threshold":
            operator = rule.get("operator", "less_than")
            threshold = rule["threshold"]
            
            if operator == "less_than":
                return value < threshold
            elif operator == "greater_than":
                return value > threshold
            elif operator == "equals":
                return abs(value - threshold) < 1e-6
        
        # Can extend with more rule types
        return False
    
    def _trigger_alert(self, metric_name: str, value: float, rule: Dict[str, Any]):
        """Trigger an alert."""
        alert = {
            "metric": metric_name,
            "value": value,
            "message": rule.get("message", f"Alert for {metric_name}"),
            "severity": rule.get("severity", "warning"),
            "timestamp": time.time(),
        }
        
        self.alerts.append(alert)
        logger.warning(f"🚨 Alert: {alert['message']} (value: {value})")
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(metric_name, alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def register_callback(self, callback: Callable[[str, Dict], None]):
        """Register an alert callback."""
        self.alert_callbacks.append(callback)
    
    def add_rule(self, metric_name: str, rule: Dict[str, Any]):
        """Add a custom alert rule."""
        self.alert_rules[metric_name] = rule
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get alerts, optionally filtered by severity.
        
        Args:
            severity: Filter by severity level
            
        Returns:
            List of alerts
        """
        if severity:
            return [a for a in self.alerts if a["severity"] == severity]
        return self.alerts.copy()
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()


class AutoMLOrchestrator:
    """
    Orchestrates automated machine learning experiments.
    
    Combines:
    - Architecture search
    - Hyperparameter optimization
    - Continuous testing
    - Regression validation
    - Alert monitoring
    """
    
    def __init__(
        self,
        generalization_tester: Optional[ContinuousGeneralizationTester] = None,
        regression_validator: Optional[RegressionValidator] = None,
        metric_alerter: Optional[MetricAlerter] = None,
    ):
        """
        Initialize AutoML orchestrator.
        
        Args:
            generalization_tester: Generalization testing component
            regression_validator: Regression validation component
            metric_alerter: Metric alerting component
        """
        self.generalization_tester = generalization_tester or ContinuousGeneralizationTester()
        self.regression_validator = regression_validator or RegressionValidator()
        self.metric_alerter = metric_alerter or MetricAlerter()
        
        self.experiment_history = []
        self.current_experiment = None
    
    def step(
        self,
        forest,
        metrics: Dict[str, float],
        test_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ):
        """
        Perform one step of AutoML orchestration.
        
        Args:
            forest: ForestEcosystem instance
            metrics: Current metrics
            test_data: Optional test data for generalization testing
        """
        # Check for regressions
        for metric_name, value in metrics.items():
            is_regression, message = self.regression_validator.check_regression(
                metric_name, value, lower_is_better=True
            )
            if is_regression:
                logger.warning(message)
        
        # Check metrics for alerts
        self.metric_alerter.check_metrics(metrics)
        
        # Run generalization tests if needed
        if self.generalization_tester.should_test() and test_data:
            result = self.generalization_tester.test_generalization(
                forest, test_data
            )
            logger.info(f"Generalization test: {result.test_name} - Score: {result.score:.4f}")
        
        # Create checkpoint if needed
        if self.regression_validator.should_checkpoint():
            self.regression_validator.create_checkpoint({"metrics": metrics})
        
        # Increment counters
        self.generalization_tester.step()
        self.regression_validator.step()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current AutoML status."""
        return {
            "generalization": self.generalization_tester.get_test_summary(),
            "regression": self.regression_validator.get_summary(),
            "alerts": {
                "total": len(self.metric_alerter.alerts),
                "critical": len(self.metric_alerter.get_alerts("critical")),
                "warning": len(self.metric_alerter.get_alerts("warning")),
            },
        }
    
    def print_status(self):
        """Print status to console."""
        status = self.get_status()
        
        print("\n" + "="*60)
        print("🤖 AutoML Orchestrator Status")
        print("="*60)
        
        print("\n📊 Generalization Testing:")
        gen = status["generalization"]
        print(f"   Tests run: {gen['total_tests']}")
        print(f"   Pass rate: {gen['pass_rate']*100:.1f}%")
        print(f"   Recent avg score: {gen.get('recent_avg_score', 0):.4f}")
        
        print("\n🔍 Regression Validation:")
        reg = status["regression"]
        print(f"   Checkpoints: {reg['total_checkpoints']}")
        print(f"   Regressions detected: {reg['total_regressions']}")
        
        print("\n🚨 Alerts:")
        alerts = status["alerts"]
        print(f"   Total: {alerts['total']}")
        print(f"   Critical: {alerts['critical']}")
        print(f"   Warning: {alerts['warning']}")
        
        print("\n" + "="*60 + "\n")
