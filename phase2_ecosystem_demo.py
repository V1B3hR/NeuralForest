"""
Demo: Phase 2 - Forest Ecosystem Simulation (roadmap2.md)

Demonstrates:
- Competition for resources (data batches)
- Robustness tests (drought, flood scenarios)
- Statistics logging
- Fitness-based selection and pruning
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from NeuralForest import ForestEcosystem, DEVICE
from ecosystem_simulation import EcosystemSimulator, RobustnessTester


def demo_basic_competition():
    """Demo basic resource competition among trees."""
    print("\n" + "=" * 70)
    print("Demo 1: Basic Resource Competition")
    print("=" * 70)
    
    # Create a forest with multiple trees
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=10).to(DEVICE)
    
    # Plant additional trees
    for _ in range(4):
        forest._plant_tree()
    
    print(f"Forest initialized with {forest.num_trees()} trees")
    
    # Set different fitness levels to demonstrate competition
    for i, tree in enumerate(forest.trees):
        tree.fitness = float(i + 1) * 2.0  # Increasing fitness
    
    print("\nTree fitness levels:")
    for tree in forest.trees:
        print(f"  Tree {tree.id}: fitness = {tree.fitness:.2f}")
    
    # Create ecosystem simulator
    simulator = EcosystemSimulator(forest, competition_fairness=0.2)
    
    # Generate sample data
    batch_size = 100
    batch_x = torch.randn(batch_size, 2).to(DEVICE)
    batch_y = torch.randn(batch_size, 1).to(DEVICE)
    
    print(f"\nTotal data available: {batch_size} samples")
    
    # Allocate data through competition
    allocations = simulator.competition.allocate_data(forest, batch_x, batch_y)
    
    print("\nResource allocation results:")
    total_allocated = 0
    for tree_id, (data_x, data_y) in allocations.items():
        allocated_size = data_x.shape[0]
        total_allocated += allocated_size
        tree = [t for t in forest.trees if t.id == tree_id][0]
        print(f"  Tree {tree_id} (fitness={tree.fitness:.2f}): {allocated_size} samples ({allocated_size/batch_size*100:.1f}%)")
    
    print(f"\nTotal allocated: {total_allocated} samples")
    print("✅ Resource competition demo successful!")
    
    return forest, simulator


def demo_robustness_drought():
    """Demo drought (data scarcity) robustness test."""
    print("\n" + "=" * 70)
    print("Demo 2: Robustness Test - Drought (Data Scarcity)")
    print("=" * 70)
    
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=8).to(DEVICE)
    
    # Plant trees
    for _ in range(3):
        forest._plant_tree()
    
    print(f"Forest with {forest.num_trees()} trees")
    
    # Create test data
    batch_size = 50
    batch_x = torch.randn(batch_size, 2).to(DEVICE)
    batch_y = torch.randn(batch_size, 1).to(DEVICE)
    
    print(f"Original batch size: {batch_size}")
    
    # Test different drought severities
    severities = [0.0, 0.3, 0.6, 0.9]
    
    print("\nDrought impact:")
    for severity in severities:
        drought_x, drought_y = RobustnessTester.apply_drought(batch_x, batch_y, severity)
        remaining = drought_x.shape[0]
        print(f"  Severity {severity:.1f}: {remaining} samples remaining ({remaining/batch_size*100:.1f}%)")
    
    print("✅ Drought robustness test successful!")


def demo_robustness_flood():
    """Demo flood (data corruption/noise) robustness test."""
    print("\n" + "=" * 70)
    print("Demo 3: Robustness Test - Flood (Data Corruption)")
    print("=" * 70)
    
    # Create clean data
    batch_size = 20
    batch_x = torch.randn(batch_size, 2).to(DEVICE)
    batch_y = torch.randn(batch_size, 1).to(DEVICE)
    
    print(f"Original data statistics:")
    print(f"  Input mean: {batch_x.mean().item():.4f}, std: {batch_x.std().item():.4f}")
    print(f"  Target mean: {batch_y.mean().item():.4f}, std: {batch_y.std().item():.4f}")
    
    # Test different flood severities
    severities = [0.0, 0.3, 0.6, 0.9]
    
    print("\nFlood impact (noise injection):")
    for severity in severities:
        flood_x, flood_y = RobustnessTester.apply_flood(batch_x, batch_y, severity)
        noise_x = (flood_x - batch_x).abs().mean().item()
        noise_y = (flood_y - batch_y).abs().mean().item()
        print(f"  Severity {severity:.1f}: input noise={noise_x:.4f}, target noise={noise_y:.4f}")
    
    print("✅ Flood robustness test successful!")


def demo_ecosystem_simulation():
    """Demo full ecosystem simulation with generations."""
    print("\n" + "=" * 70)
    print("Demo 4: Full Ecosystem Simulation")
    print("=" * 70)
    
    # Create forest
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=12).to(DEVICE)
    
    # Plant initial trees
    for _ in range(5):
        forest._plant_tree()
    
    print(f"Starting forest with {forest.num_trees()} trees")
    
    # Create simulator
    simulator = EcosystemSimulator(
        forest,
        competition_fairness=0.3,
        selection_threshold=0.2
    )
    
    # Simulate multiple generations
    num_generations = 5
    
    print(f"\nSimulating {num_generations} generations...")
    
    for gen in range(num_generations):
        # Generate data for this generation
        batch_x = torch.randn(100, 2).to(DEVICE)
        batch_y = torch.randn(100, 1).to(DEVICE)
        
        # Randomly apply disruptions
        disruption = None
        severity = 0.0
        
        if gen == 2:
            disruption = "drought"
            severity = 0.4
        elif gen == 3:
            disruption = "flood"
            severity = 0.3
        
        # Simulate generation
        stats = simulator.simulate_generation(batch_x, batch_y, disruption, severity)
        
        # Update fitness based on some criteria (simplified)
        for i, tree in enumerate(forest.trees):
            tree.fitness += np.random.uniform(0.5, 2.0)
        
        print(f"\nGeneration {stats.generation}:")
        print(f"  Trees: {stats.num_trees}")
        print(f"  Avg fitness: {stats.avg_fitness:.2f} (min: {stats.min_fitness:.2f}, max: {stats.max_fitness:.2f})")
        print(f"  Architecture diversity: {stats.unique_architectures} unique types")
        print(f"  Data allocated: {stats.total_data_allocated} samples")
        
        if disruption:
            print(f"  Disruption: {disruption} (severity={severity:.1f})")
    
    # Get ecosystem summary
    summary = simulator.get_summary()
    
    print("\n" + "-" * 70)
    print("Ecosystem Summary:")
    print(f"  Total generations: {summary['current_generation']}")
    print(f"  Current trees: {summary['total_trees']}")
    print(f"  Current fitness: avg={summary['current_fitness']['avg']:.2f}, "
          f"max={summary['current_fitness']['max']:.2f}")
    print(f"  Architecture diversity: {summary['architecture_diversity']} types")
    print(f"  Average tree age: {summary['avg_tree_age']:.1f} steps")
    
    print("✅ Ecosystem simulation demo successful!")
    
    return simulator


def demo_selection_and_pruning():
    """Demo fitness-based selection and pruning."""
    print("\n" + "=" * 70)
    print("Demo 5: Fitness-Based Selection and Pruning")
    print("=" * 70)
    
    # Create forest with many trees
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=15).to(DEVICE)
    
    # Plant trees
    for _ in range(9):
        forest._plant_tree()
    
    print(f"Forest with {forest.num_trees()} trees")
    
    # Assign varied fitness levels
    for i, tree in enumerate(forest.trees):
        tree.fitness = float(i) * 1.5 + np.random.uniform(0, 2)
    
    print("\nInitial fitness levels:")
    for tree in sorted(forest.trees, key=lambda t: t.fitness):
        print(f"  Tree {tree.id}: fitness = {tree.fitness:.2f}")
    
    # Create simulator
    simulator = EcosystemSimulator(
        forest,
        selection_threshold=0.3  # Remove bottom 30%
    )
    
    # Identify weak trees
    to_remove, selection_rate = simulator.selection_pressure(min_keep=5)
    
    print(f"\nSelection pressure (threshold: bottom 30%):")
    print(f"  Trees to prune: {len(to_remove)}")
    print(f"  Selection rate: {selection_rate:.2%}")
    print(f"  IDs to remove: {to_remove}")
    
    # Prune weak trees
    pruned = simulator.prune_weak_trees(min_keep=5)
    
    print(f"\nAfter pruning:")
    print(f"  Trees removed: {pruned}")
    print(f"  Trees remaining: {forest.num_trees()}")
    
    print("\nRemaining trees:")
    for tree in sorted(forest.trees, key=lambda t: t.fitness, reverse=True):
        print(f"  Tree {tree.id}: fitness = {tree.fitness:.2f}")
    
    print("✅ Selection and pruning demo successful!")


def demo_planting_and_growth():
    """Demo dynamic tree planting."""
    print("\n" + "=" * 70)
    print("Demo 6: Dynamic Tree Planting")
    print("=" * 70)
    
    # Create forest with few trees
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=12).to(DEVICE)
    
    print(f"Starting with {forest.num_trees()} tree")
    
    # Create simulator
    simulator = EcosystemSimulator(forest)
    
    # Plant trees over multiple rounds
    rounds = [2, 3, 2, 1]
    
    print("\nPlanting trees:")
    for i, count in enumerate(rounds):
        planted = simulator.plant_trees(count)
        print(f"  Round {i+1}: attempted to plant {count}, actually planted {planted}")
        print(f"          Current total: {forest.num_trees()} trees")
    
    print(f"\nFinal forest size: {forest.num_trees()} trees")
    print("✅ Tree planting demo successful!")


def demo_statistics_tracking():
    """Demo comprehensive statistics tracking."""
    print("\n" + "=" * 70)
    print("Demo 7: Statistics Tracking")
    print("=" * 70)
    
    # Create and populate forest
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=10).to(DEVICE)
    
    for _ in range(4):
        forest._plant_tree()
    
    # Create simulator
    simulator = EcosystemSimulator(forest, competition_fairness=0.3)
    
    # Run several generations
    print("Running 3 generations with statistics tracking...\n")
    
    for gen in range(3):
        # Generate data
        batch_x = torch.randn(80, 2).to(DEVICE)
        batch_y = torch.randn(80, 1).to(DEVICE)
        
        # Update fitness
        for tree in forest.trees:
            tree.fitness += np.random.uniform(1.0, 3.0)
        
        # Simulate
        stats = simulator.simulate_generation(batch_x, batch_y)
        
        print(f"Generation {stats.generation}:")
        print(f"  Timestamp: {stats.timestamp:.2f}")
        print(f"  Trees: {stats.num_trees}")
        print(f"  Fitness: avg={stats.avg_fitness:.2f}, std={stats.fitness_std:.2f}")
        print(f"  Architecture diversity: {stats.unique_architectures}")
        print(f"  Avg tree age: {stats.avg_tree_age:.1f}")
        print(f"  Data allocated: {stats.total_data_allocated}")
        print()
    
    # Get full history
    history = simulator.get_stats_history()
    
    print("Statistics History:")
    print(f"  Total generations tracked: {len(history)}")
    print(f"  Keys tracked per generation: {list(history[0].keys())[:8]}...")
    
    print("✅ Statistics tracking demo successful!")


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 2: Forest Ecosystem Simulation Demo (roadmap2.md)")
    print("=" * 70)
    
    # Run all demos
    demo_basic_competition()
    demo_robustness_drought()
    demo_robustness_flood()
    demo_ecosystem_simulation()
    demo_selection_and_pruning()
    demo_planting_and_growth()
    demo_statistics_tracking()
    
    print("\n" + "=" * 70)
    print("All Phase 2 Ecosystem demos completed successfully!")
    print("=" * 70)
