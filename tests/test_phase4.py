"""
Basic tests for Phase 4 components.

Tests the core functionality of:
- Seasonal evolution integration
- Genealogy tracking
- Monitoring system
- AutoML components
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
from pathlib import Path
import tempfile

from evolution import (
    SeasonalEvolution,
    GenealogyTracker,
    EvolutionMonitor,
    AutoMLOrchestrator,
    ContinuousGeneralizationTester,
    RegressionValidator,
    MetricAlerter,
)
from seasons import SeasonalCycle


def test_seasonal_evolution():
    """Test seasonal evolution parameter adaptation."""
    seasonal_evo = SeasonalEvolution(base_mutation_rate=0.1, base_crossover_prob=0.3)
    
    # Test all seasons
    for season in ["spring", "summer", "autumn", "winter"]:
        params = seasonal_evo.get_evolutionary_params(season)
        assert "mutation_rate" in params
        assert "crossover_prob" in params
        assert "selection_pressure" in params
        assert params["season"] == season
        
        nas_params = seasonal_evo.get_nas_parameters(season)
        assert "population_size" in nas_params
        assert "num_generations" in nas_params
    
    # Test recommendations
    forest_state = {"num_trees": 5, "avg_fitness": 5.0, "diversity": 0.5}
    recommendations = seasonal_evo.get_recommendations("spring", forest_state)
    assert isinstance(recommendations, list)


def test_genealogy_tracker():
    """Test genealogy tracking functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GenealogyTracker(save_dir=Path(tmpdir))
        
        # Register trees
        lineage1 = tracker.register_tree(
            tree_id=0,
            generation=0,
            creation_method="random",
            birth_fitness=5.0
        )
        assert lineage1.tree_id == 0
        
        lineage2 = tracker.register_tree(
            tree_id=1,
            generation=1,
            parent_ids=[0],
            creation_method="mutation",
            birth_fitness=6.0
        )
        assert 1 in tracker.lineages[0].children_ids
        
        # Update fitness
        tracker.update_fitness(0, 7.5)
        assert tracker.lineages[0].peak_fitness == 7.5
        
        # Mark eliminated
        tracker.mark_eliminated(1, age=50, reason="low_fitness")
        assert not tracker.lineages[1].is_alive
        
        # Get ancestors
        ancestors = tracker.get_ancestors(1)
        assert 0 in ancestors
        
        # Get statistics
        stats = tracker.get_lineage_statistics()
        assert stats["total_trees"] == 2
        assert stats["alive_trees"] == 1
        
        # Save and load
        tracker.save()
        save_path = Path(tmpdir) / "genealogy.json"
        assert save_path.exists()


def test_evolution_monitor():
    """Test evolution monitoring system."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monitor = EvolutionMonitor(window_size=10, save_dir=Path(tmpdir))
        
        # Record snapshots
        for step in range(15):
            forest_state = {
                "num_trees": 5,
                "alive_trees": 5,
                "dead_trees": 0,
                "avg_fitness": 5.0 + step * 0.1,
                "max_fitness": 7.0 + step * 0.1,
                "min_fitness": 3.0,
                "fitness_std": 1.0,
                "architecture_diversity": 0.5,
                "fitness_diversity": 0.4,
            }
            
            monitor.record_snapshot(
                generation=step // 5,
                step=step,
                forest_state=forest_state,
                season="spring"
            )
        
        # Check window size
        assert len(monitor.snapshots) == 10  # Limited by window_size
        
        # Get statistics
        stats = monitor.get_statistics()
        assert stats["total_snapshots"] == 15
        assert stats["current_generation"] == 2
        
        # Get trend
        fitness_trend = monitor.get_trend("avg_fitness")
        assert len(fitness_trend) == 10
        
        # Export metrics
        export_path = Path(tmpdir) / "metrics.json"
        monitor.export_metrics(export_path)
        assert export_path.exists()


def test_continuous_generalization_tester():
    """Test continuous generalization testing."""
    tester = ContinuousGeneralizationTester(test_frequency=10)
    
    # First step should trigger (step 0)
    assert tester.should_test()
    tester.step()
    
    # Now should not trigger until step 10
    assert not tester.should_test()
    for _ in range(9):
        tester.step()
    assert tester.should_test()
    
    # Get summary
    summary = tester.get_test_summary()
    assert summary["total_tests"] == 0


def test_regression_validator():
    """Test regression validation."""
    validator = RegressionValidator(regression_threshold=0.1, checkpoint_frequency=20)
    
    # Set baseline
    validator.set_baseline("loss", 1.0)
    assert "loss" in validator.baselines
    
    # Check for regression
    is_regression, message = validator.check_regression("loss", 1.2, lower_is_better=True)
    assert is_regression
    
    # Check for improvement
    is_regression, message = validator.check_regression("loss", 0.8, lower_is_better=True)
    assert not is_regression
    
    # Get summary
    summary = validator.get_summary()
    assert "total_regressions" in summary


def test_metric_alerter():
    """Test metric alerting system."""
    alerter = MetricAlerter()
    
    # Add custom rule
    alerter.add_rule("custom_metric", {
        "type": "threshold",
        "operator": "less_than",
        "threshold": 0.5,
        "message": "Custom metric too low",
        "severity": "warning"
    })
    
    # Check metrics (should trigger alert)
    metrics = {"custom_metric": 0.3}
    alerter.check_metrics(metrics)
    
    # Check alerts
    alerts = alerter.get_alerts()
    assert len(alerts) > 0
    assert alerts[0]["metric"] == "custom_metric"


def test_automl_orchestrator():
    """Test AutoML orchestrator integration."""
    orchestrator = AutoMLOrchestrator()
    
    # Get status
    status = orchestrator.get_status()
    assert "generalization" in status
    assert "regression" in status
    assert "alerts" in status


def test_seasonal_cycle_integration():
    """Test integration between seasonal cycle and evolution."""
    cycle = SeasonalCycle(steps_per_season=10)
    seasonal_evo = SeasonalEvolution()
    
    # Step through seasons
    for _ in range(45):
        season = cycle.current_season
        params = seasonal_evo.get_evolutionary_params(season)
        
        # Verify parameters adapt to season
        assert params["season"] == season
        
        cycle.step()


if __name__ == "__main__":
    # Run tests
    print("Running Phase 4 component tests...")
    
    test_seasonal_evolution()
    print("✓ Seasonal evolution tests passed")
    
    test_genealogy_tracker()
    print("✓ Genealogy tracker tests passed")
    
    test_evolution_monitor()
    print("✓ Evolution monitor tests passed")
    
    test_continuous_generalization_tester()
    print("✓ Generalization tester tests passed")
    
    test_regression_validator()
    print("✓ Regression validator tests passed")
    
    test_metric_alerter()
    print("✓ Metric alerter tests passed")
    
    test_automl_orchestrator()
    print("✓ AutoML orchestrator tests passed")
    
    test_seasonal_cycle_integration()
    print("✓ Seasonal cycle integration tests passed")
    
    print("\n✅ All Phase 4 tests passed!")
