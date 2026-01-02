"""
Tests for Phase 2: Forest Ecosystem Simulation (roadmap2.md)

Tests competition, robustness, selection, and statistics.
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NeuralForest import ForestEcosystem, DEVICE
from ecosystem_simulation import (
    CompetitionSystem,
    RobustnessTester,
    EcosystemSimulator,
    EcosystemStats,
)


class TestCompetitionSystem:
    """Test resource competition functionality."""
    
    def test_competition_system_creation(self):
        """Test creating a competition system."""
        comp = CompetitionSystem(fairness_factor=0.3)
        assert comp.fairness_factor == 0.3
        assert len(comp.allocation_history) == 0
    
    def test_data_allocation(self):
        """Test allocating data to trees based on fitness."""
        forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=10).to(DEVICE)
        
        # Plant trees with different fitness
        for _ in range(3):
            forest._plant_tree()
        
        for i, tree in enumerate(forest.trees):
            tree.fitness = float(i + 1) * 2.0
        
        comp = CompetitionSystem(fairness_factor=0.2)
        
        batch_x = torch.randn(100, 2).to(DEVICE)
        batch_y = torch.randn(100, 1).to(DEVICE)
        
        allocations = comp.allocate_data(forest, batch_x, batch_y)
        
        # Check all trees got some data
        assert len(allocations) == forest.num_trees()
        
        # Check total allocated equals batch size
        total = sum(v[0].shape[0] for v in allocations.values())
        assert total == 100
        
        # Check higher fitness trees got more data
        tree_ids = [t.id for t in sorted(forest.trees, key=lambda t: t.fitness)]
        sizes = [allocations[tid][0].shape[0] for tid in tree_ids]
        
        # Generally, sizes should increase with fitness
        assert sizes[-1] >= sizes[0]
    
    def test_allocation_history(self):
        """Test that allocation history is tracked."""
        forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=5).to(DEVICE)
        
        for _ in range(2):
            forest._plant_tree()
        
        comp = CompetitionSystem()
        
        # Allocate multiple times
        for _ in range(5):
            batch_x = torch.randn(50, 2).to(DEVICE)
            batch_y = torch.randn(50, 1).to(DEVICE)
            comp.allocate_data(forest, batch_x, batch_y)
        
        assert len(comp.allocation_history) == 5
        
        stats = comp.get_allocation_stats()
        assert stats['total_allocations'] == 5


class TestRobustnessTester:
    """Test robustness disruption functionality."""
    
    def test_drought_reduces_data(self):
        """Test drought reduces batch size."""
        batch_x = torch.randn(100, 2)
        batch_y = torch.randn(100, 1)
        
        drought_x, drought_y = RobustnessTester.apply_drought(
            batch_x, batch_y, severity=0.5
        )
        
        # Should have roughly 50% of data
        assert drought_x.shape[0] < batch_x.shape[0]
        assert drought_x.shape[0] >= 1  # At least 1 sample
        assert drought_y.shape[0] == drought_x.shape[0]
    
    def test_drought_severity_levels(self):
        """Test different drought severity levels."""
        batch_x = torch.randn(100, 2)
        batch_y = torch.randn(100, 1)
        
        # Test mild drought
        mild_x, _ = RobustnessTester.apply_drought(batch_x, batch_y, severity=0.2)
        assert mild_x.shape[0] > 70
        
        # Test severe drought
        severe_x, _ = RobustnessTester.apply_drought(batch_x, batch_y, severity=0.9)
        assert severe_x.shape[0] < 20
    
    def test_flood_adds_noise(self):
        """Test flood adds noise to data."""
        batch_x = torch.randn(50, 2)
        batch_y = torch.randn(50, 1)
        
        flood_x, flood_y = RobustnessTester.apply_flood(
            batch_x, batch_y, severity=0.5
        )
        
        # Shape should remain the same
        assert flood_x.shape == batch_x.shape
        assert flood_y.shape == batch_y.shape
        
        # Data should be different (noisy)
        assert not torch.allclose(flood_x, batch_x, atol=0.1)
    
    def test_flood_severity_levels(self):
        """Test different flood severity levels."""
        batch_x = torch.randn(50, 2)
        batch_y = torch.randn(50, 1)
        
        # Test mild flood
        mild_x, _ = RobustnessTester.apply_flood(batch_x, batch_y, severity=0.1)
        mild_noise = (mild_x - batch_x).abs().mean()
        
        # Test severe flood
        severe_x, _ = RobustnessTester.apply_flood(batch_x, batch_y, severity=0.9)
        severe_noise = (severe_x - batch_x).abs().mean()
        
        # Severe should have more noise
        assert severe_noise > mild_noise
    
    def test_disruption_dispatcher(self):
        """Test the disruption dispatcher."""
        batch_x = torch.randn(50, 2)
        batch_y = torch.randn(50, 1)
        
        # Test drought
        drought_x, _ = RobustnessTester.apply_disruption(
            batch_x, batch_y, "drought", 0.5
        )
        assert drought_x.shape[0] < batch_x.shape[0]
        
        # Test flood
        flood_x, _ = RobustnessTester.apply_disruption(
            batch_x, batch_y, "flood", 0.5
        )
        assert flood_x.shape == batch_x.shape
        
        # Test no disruption
        normal_x, _ = RobustnessTester.apply_disruption(
            batch_x, batch_y, "none", 0.0
        )
        assert torch.equal(normal_x, batch_x)


class TestEcosystemStats:
    """Test ecosystem statistics tracking."""
    
    def test_stats_creation(self):
        """Test creating ecosystem stats."""
        stats = EcosystemStats()
        
        assert stats.generation == 0
        assert stats.num_trees == 0
        assert stats.avg_fitness == 0.0
    
    def test_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = EcosystemStats(
            generation=5,
            num_trees=10,
            avg_fitness=7.5,
            max_fitness=12.0,
            min_fitness=3.0
        )
        
        stats_dict = stats.to_dict()
        
        assert stats_dict['generation'] == 5
        assert stats_dict['num_trees'] == 10
        assert stats_dict['avg_fitness'] == 7.5
        assert 'max_fitness' in stats_dict
        assert 'unique_architectures' in stats_dict


class TestEcosystemSimulator:
    """Test full ecosystem simulation."""
    
    def test_simulator_creation(self):
        """Test creating an ecosystem simulator."""
        forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=10).to(DEVICE)
        
        sim = EcosystemSimulator(
            forest,
            competition_fairness=0.3,
            selection_threshold=0.2
        )
        
        assert sim.forest == forest
        assert sim.selection_threshold == 0.2
        assert sim.generation == 0
    
    def test_simulate_generation(self):
        """Test simulating a single generation."""
        forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=10).to(DEVICE)
        
        for _ in range(3):
            forest._plant_tree()
        
        sim = EcosystemSimulator(forest)
        
        batch_x = torch.randn(80, 2).to(DEVICE)
        batch_y = torch.randn(80, 1).to(DEVICE)
        
        stats = sim.simulate_generation(batch_x, batch_y)
        
        assert stats.generation == 1
        assert stats.num_trees == forest.num_trees()
        assert stats.total_data_allocated > 0
    
    def test_simulate_with_disruption(self):
        """Test simulation with environmental disruption."""
        forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=10).to(DEVICE)
        
        for _ in range(2):
            forest._plant_tree()
        
        sim = EcosystemSimulator(forest)
        
        batch_x = torch.randn(100, 2).to(DEVICE)
        batch_y = torch.randn(100, 1).to(DEVICE)
        
        # Simulate with drought
        stats = sim.simulate_generation(
            batch_x, batch_y,
            disruption_type="drought",
            disruption_severity=0.5
        )
        
        assert stats.disruption_type == "drought"
        assert stats.disruption_severity == 0.5
        # Less data should be allocated due to drought
        assert stats.total_data_allocated < 100
    
    def test_selection_pressure(self):
        """Test identifying weak trees for pruning."""
        forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=10).to(DEVICE)
        
        # Plant trees with varied fitness
        for _ in range(5):
            forest._plant_tree()
        
        for i, tree in enumerate(forest.trees):
            tree.fitness = float(i) * 2.0
        
        sim = EcosystemSimulator(forest, selection_threshold=0.4)
        
        to_remove, selection_rate = sim.selection_pressure(min_keep=2)
        
        # Should identify some trees for removal
        assert len(to_remove) > 0
        assert selection_rate > 0
        assert len(to_remove) <= forest.num_trees() - 2
    
    def test_prune_weak_trees(self):
        """Test pruning weak trees."""
        forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=10).to(DEVICE)
        
        # Plant many trees
        for _ in range(7):
            forest._plant_tree()
        
        # Set different fitness levels
        for i, tree in enumerate(forest.trees):
            tree.fitness = float(i) * 1.5
        
        initial_count = forest.num_trees()
        
        sim = EcosystemSimulator(forest, selection_threshold=0.3)
        pruned = sim.prune_weak_trees(min_keep=4)
        
        final_count = forest.num_trees()
        
        assert pruned > 0
        assert final_count < initial_count
        assert final_count >= 4
    
    def test_plant_trees(self):
        """Test planting new trees."""
        forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=10).to(DEVICE)
        
        initial_count = forest.num_trees()
        
        sim = EcosystemSimulator(forest)
        planted = sim.plant_trees(count=3)
        
        final_count = forest.num_trees()
        
        assert planted == 3
        assert final_count == initial_count + 3
    
    def test_plant_trees_respects_max(self):
        """Test that planting respects max_trees limit."""
        forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=5).to(DEVICE)
        
        # Already has 1 tree, max is 5
        sim = EcosystemSimulator(forest)
        
        # Try to plant more than capacity
        planted = sim.plant_trees(count=10)
        
        # Should only plant up to max_trees
        assert planted <= 4  # 5 max - 1 existing
        assert forest.num_trees() <= 5
    
    def test_stats_history(self):
        """Test statistics history tracking."""
        forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=10).to(DEVICE)
        
        for _ in range(2):
            forest._plant_tree()
        
        sim = EcosystemSimulator(forest, max_history=100)
        
        # Run multiple generations
        for _ in range(5):
            batch_x = torch.randn(50, 2).to(DEVICE)
            batch_y = torch.randn(50, 1).to(DEVICE)
            sim.simulate_generation(batch_x, batch_y)
        
        history = sim.get_stats_history()
        
        assert len(history) == 5
        assert all(isinstance(s, dict) for s in history)
    
    def test_get_summary(self):
        """Test getting ecosystem summary."""
        forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=10).to(DEVICE)
        
        for _ in range(3):
            forest._plant_tree()
        
        sim = EcosystemSimulator(forest)
        
        # Run a few generations
        for _ in range(3):
            batch_x = torch.randn(60, 2).to(DEVICE)
            batch_y = torch.randn(60, 1).to(DEVICE)
            sim.simulate_generation(batch_x, batch_y)
        
        summary = sim.get_summary()
        
        assert 'current_generation' in summary
        assert 'total_trees' in summary
        assert 'current_fitness' in summary
        assert summary['current_generation'] == 3


def run_all_tests():
    """Run all Phase 2 ecosystem tests."""
    print("\n" + "=" * 70)
    print("Running Phase 2 Ecosystem Tests")
    print("=" * 70)
    
    # Competition tests
    print("\n--- Competition System Tests ---")
    test_comp = TestCompetitionSystem()
    test_comp.test_competition_system_creation()
    print("✓ Competition system creation")
    test_comp.test_data_allocation()
    print("✓ Data allocation")
    test_comp.test_allocation_history()
    print("✓ Allocation history")
    
    # Robustness tests
    print("\n--- Robustness Tester Tests ---")
    test_robust = TestRobustnessTester()
    test_robust.test_drought_reduces_data()
    print("✓ Drought reduces data")
    test_robust.test_drought_severity_levels()
    print("✓ Drought severity levels")
    test_robust.test_flood_adds_noise()
    print("✓ Flood adds noise")
    test_robust.test_flood_severity_levels()
    print("✓ Flood severity levels")
    test_robust.test_disruption_dispatcher()
    print("✓ Disruption dispatcher")
    
    # Stats tests
    print("\n--- Ecosystem Stats Tests ---")
    test_stats = TestEcosystemStats()
    test_stats.test_stats_creation()
    print("✓ Stats creation")
    test_stats.test_stats_to_dict()
    print("✓ Stats to dict")
    
    # Simulator tests
    print("\n--- Ecosystem Simulator Tests ---")
    test_sim = TestEcosystemSimulator()
    test_sim.test_simulator_creation()
    print("✓ Simulator creation")
    test_sim.test_simulate_generation()
    print("✓ Simulate generation")
    test_sim.test_simulate_with_disruption()
    print("✓ Simulate with disruption")
    test_sim.test_selection_pressure()
    print("✓ Selection pressure")
    test_sim.test_prune_weak_trees()
    print("✓ Prune weak trees")
    test_sim.test_plant_trees()
    print("✓ Plant trees")
    test_sim.test_plant_trees_respects_max()
    print("✓ Plant trees respects max")
    test_sim.test_stats_history()
    print("✓ Stats history")
    test_sim.test_get_summary()
    print("✓ Get summary")
    
    print("\n" + "=" * 70)
    print("✅ All Phase 2 Ecosystem tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
