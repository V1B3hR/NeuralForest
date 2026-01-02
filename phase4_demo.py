#!/usr/bin/env python3
"""
Phase 4 Demo: Seasonal Cycles (Training Regimes)

Demonstrates the seasonal training cycle system with:
- SeasonalCycle controller
- Spring growth phase
- Summer productivity phase
- Autumn pruning phase
- Winter consolidation phase
"""

import torch
import random
from seasons import (
    SeasonalCycle,
    SpringGrowth,
    SummerProductivity,
    AutumnPruning,
    WinterConsolidation
)
from NeuralForest import ForestEcosystem


def demo_seasonal_cycle():
    """Demonstrate the seasonal cycle controller."""
    print("=" * 60)
    print("Phase 4 Demo: Seasonal Cycles")
    print("=" * 60)
    print()
    
    # Create seasonal cycle
    print("1. Creating Seasonal Cycle Controller")
    print("-" * 60)
    cycle = SeasonalCycle(steps_per_season=100)
    
    # Show initial status
    status = cycle.get_status()
    print(f"Current Season: {status['current_season']}")
    print(f"Season Progress: {status['season_progress']:.2%}")
    print(f"Year: {status['year']}")
    print()
    
    # Demonstrate season progression
    print("2. Simulating Season Progression")
    print("-" * 60)
    for step in range(450):
        result = cycle.step()
        
        # Print on season transitions
        if result.get('season_transition'):
            print(f"Step {result['step']}: {result['message']}")
            config = cycle.get_training_config()
            print(f"  Config: LR={config['learning_rate']}, "
                  f"Plasticity={config['plasticity']}, "
                  f"Growth Prob={config['growth_probability']}")
        
        # Print on year completion
        if result.get('year_completed'):
            print(f"\n{result['message']}\n")
    
    print()
    
    # Show final status
    print("3. Final Status")
    print("-" * 60)
    status = cycle.get_status()
    print(f"Total Steps: {status['current_step']}")
    print(f"Seasons Completed: {status['total_seasons_completed']}")
    print(f"Years Completed: {status['year']}")
    print()


def demo_spring_growth():
    """Demonstrate spring growth phase."""
    print("4. Spring Growth Phase")
    print("-" * 60)
    
    # Create a small forest
    forest = ForestEcosystem(
        input_dim=10,
        hidden_dim=32,
        max_trees=12
    )
    
    spring = SpringGrowth(forest)
    config = {'growth_probability': 0.8}
    
    print(f"Initial trees: {len(forest.trees)}")
    
    # Try planting trees
    loss_trend = [2.5, 2.3, 2.4, 2.6, 2.8]  # Increasing loss
    planted = spring.maybe_plant_trees(loss_trend, config)
    print(f"Trees after spring planting: {len(forest.trees)}")
    
    # Get recommendations
    recommendations = spring.get_growth_recommendations()
    print("\nSpring Recommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    print()


def demo_summer_productivity():
    """Demonstrate summer productivity phase."""
    print("5. Summer Productivity Phase")
    print("-" * 60)
    
    forest = ForestEcosystem(
        input_dim=10,
        hidden_dim=32,
        max_trees=12
    )
    # Plant a few more trees
    forest._plant_tree()
    forest._plant_tree()
    
    summer = SummerProductivity(forest)
    
    # Simulate training
    batch_x = torch.randn(8, 10)
    batch_y = torch.randn(8, 1)
    optimizer = torch.optim.Adam(forest.parameters(), lr=0.02)
    criterion = torch.nn.MSELoss()
    
    for _ in range(5):
        metrics = summer.intensive_training_pass(batch_x, batch_y, optimizer, criterion)
    
    # Get productivity metrics
    metrics = summer.get_productivity_metrics()
    print(f"Training Steps: {metrics['training_steps']}")
    print(f"Average Loss: {metrics['avg_loss']:.4f}")
    print(f"Average Tree Fitness: {metrics.get('avg_tree_fitness', 0):.2f}")
    
    # Get recommendations
    recommendations = summer.recommend_optimizations()
    print("\nSummer Recommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    print()


def demo_autumn_pruning():
    """Demonstrate autumn pruning phase."""
    print("6. Autumn Pruning Phase")
    print("-" * 60)
    
    forest = ForestEcosystem(
        input_dim=10,
        hidden_dim=32,
        max_trees=12
    )
    # Plant a few more trees
    for _ in range(4):
        forest._plant_tree()
    
    # Age some trees and vary fitness
    for i, tree in enumerate(forest.trees):
        tree.age = 60 + i * 20
        tree.fitness = 1.5 + i * 1.5
    
    autumn = AutumnPruning(forest)
    
    # Evaluate forest health
    health = autumn.evaluate_forest_health()
    print(f"Total Trees: {health['total_trees']}")
    print(f"Overall Health: {health['overall_health']}")
    print(f"\nTree Status:")
    for tree_info in health['trees']:
        print(f"  Tree {tree_info['id']}: fitness={tree_info['fitness']:.2f}, "
              f"age={tree_info['age']}, status={tree_info['status']}")
    
    # Try pruning
    config = {'prune_probability': 1.0}  # Force pruning
    pruned = autumn.prune_weakest(config, min_keep=3)
    print(f"\nPruned {len(pruned)} tree(s)")
    print(f"Trees remaining: {len(forest.trees)}")
    print()


def demo_winter_consolidation():
    """Demonstrate winter consolidation phase."""
    print("7. Winter Consolidation Phase")
    print("-" * 60)
    
    forest = ForestEcosystem(
        input_dim=10,
        hidden_dim=32,
        max_trees=12
    )
    # Plant a few more trees
    for _ in range(3):
        forest._plant_tree()
    
    # Add some anchor memories
    for _ in range(20):
        x = torch.randn(1, 10)
        y = torch.randn(1, 1)
        forest.anchors.add(x, y)
    
    winter = WinterConsolidation(forest)
    
    # Perform deep consolidation
    results = winter.deep_consolidation(num_rounds=5, batch_size=10)
    print(f"Teacher Snapshot: {results['teacher_snapshot']}")
    print(f"Anchor Rounds: {results['anchor_rounds']}")
    print(f"Knowledge Transfers: {results['knowledge_transfers']}")
    
    # Strengthen bark
    bark_results = winter.strengthen_bark()
    print(f"\nBark Strengthening:")
    print(f"  Trees strengthened: {bark_results['trees_strengthened']}")
    print(f"  Avg bark before: {bark_results['avg_bark_before']:.3f}")
    print(f"  Avg bark after: {bark_results['avg_bark_after']:.3f}")
    
    # Get recommendations
    recommendations = winter.get_recommendations()
    print("\nWinter Recommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    print()


def main():
    """Run all Phase 4 demonstrations."""
    print("\n" + "=" * 60)
    print("NEURALFOREST PHASE 4: SEASONAL CYCLES DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    demo_seasonal_cycle()
    demo_spring_growth()
    demo_summer_productivity()
    demo_autumn_pruning()
    demo_winter_consolidation()
    
    print("=" * 60)
    print("Phase 4 Demo Complete! ✅")
    print("=" * 60)
    print()
    print("Summary:")
    print("- Seasonal cycles provide adaptive training regimes")
    print("- Spring encourages growth and exploration")
    print("- Summer maximizes productivity and learning")
    print("- Autumn evaluates health and prunes weak trees")
    print("- Winter consolidates knowledge and strengthens memories")
    print()


if __name__ == "__main__":
    main()
