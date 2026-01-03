#!/usr/bin/env python3
"""
Phase 4 Comprehensive Demo: AutoML, Testing, and Monitoring

Demonstrates:
1. Integration with seasonal evolution
2. Advanced genealogy tracking and visualization
3. Real-time monitoring of evolutionary progress
4. AutoML orchestration
5. Continuous generalization testing
6. Automated regression validation
7. Metric alerting
"""

import torch
import random
import numpy as np
from pathlib import Path

from NeuralForest import ForestEcosystem
from seasons import SeasonalCycle
from evolution import (
    GenealogyTracker,
    SeasonalEvolution,
    EvolutionMonitor,
    AutoMLOrchestrator,
    ContinuousGeneralizationTester,
    RegressionValidator,
    MetricAlerter,
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(input_dim: int, num_samples: int, noise: float = 0.1):
    """Generate synthetic training data."""
    X = torch.randn(num_samples, input_dim)
    # Simple target: sum of inputs with noise
    y = X.sum(dim=1, keepdim=True) + torch.randn(num_samples, 1) * noise
    return X, y


def demo_seasonal_evolution_integration():
    """Demo 1: Seasonal Evolution Integration"""
    print("\n" + "="*70)
    print("DEMO 1: Seasonal Evolution Integration")
    print("="*70 + "\n")
    
    # Create seasonal cycle and evolution controller
    seasonal_cycle = SeasonalCycle(steps_per_season=50)
    seasonal_evo = SeasonalEvolution(base_mutation_rate=0.1, base_crossover_prob=0.3)
    
    # Simulate through seasons
    print("Simulating evolutionary parameters across seasons:\n")
    
    for step in range(220):
        result = seasonal_cycle.step()
        
        if result.get("season_transition"):
            season = result["to_season"]
            evo_params = seasonal_evo.get_evolutionary_params(season)
            nas_params = seasonal_evo.get_nas_parameters(season)
            
            print(f"🌍 Step {step}: {result['message']}")
            print(f"   Mutation Rate: {evo_params['mutation_rate']:.3f}")
            print(f"   Crossover Prob: {evo_params['crossover_prob']:.3f}")
            print(f"   Selection Pressure: {evo_params['selection_pressure']:.2f}")
            print(f"   NAS Population: {nas_params['population_size']}")
            print()
            
            # Get recommendations
            forest_state = {
                "num_trees": 8,
                "avg_fitness": 5.5,
                "diversity": 0.6,
            }
            recommendations = seasonal_evo.get_recommendations(season, forest_state)
            for rec in recommendations:
                print(f"   {rec}")
            print()
    
    print("✅ Demo 1 Complete: Seasonal evolution parameters adapt dynamically\n")


def demo_genealogy_tracking():
    """Demo 2: Advanced Genealogy Tracking"""
    print("\n" + "="*70)
    print("DEMO 2: Advanced Genealogy Tracking & Visualization")
    print("="*70 + "\n")
    
    # Create genealogy tracker
    genealogy = GenealogyTracker(save_dir=Path("/tmp/genealogy"))
    
    # Simulate tree evolution
    print("Simulating evolutionary lineages...\n")
    
    # Generation 0: Initial trees
    for i in range(3):
        genealogy.register_tree(
            tree_id=i,
            generation=0,
            parent_ids=[],
            creation_method="random",
            birth_fitness=random.uniform(3.0, 5.0)
        )
    
    # Generation 1: Mutations and crossovers
    tree_id = 3
    for gen in range(1, 4):
        num_new = random.randint(2, 4)
        for _ in range(num_new):
            # Randomly choose creation method
            method = random.choice(["mutation", "crossover"])
            
            if method == "mutation":
                parent = random.randint(0, tree_id - 1)
                genealogy.register_tree(
                    tree_id=tree_id,
                    generation=gen,
                    parent_ids=[parent],
                    creation_method="mutation",
                    mutation_type=random.choice(["layer_add", "dropout_change"]),
                    birth_fitness=random.uniform(4.0, 8.0)
                )
            else:  # crossover
                parent1 = random.randint(0, tree_id - 1)
                parent2 = random.randint(0, tree_id - 1)
                if parent1 != parent2:
                    genealogy.register_tree(
                        tree_id=tree_id,
                        generation=gen,
                        parent_ids=[parent1, parent2],
                        creation_method="crossover",
                        birth_fitness=random.uniform(5.0, 9.0)
                    )
            
            # Update fitness over time
            for _ in range(5):
                genealogy.update_fitness(tree_id, random.uniform(4.0, 10.0))
            
            # Some trees get eliminated
            if random.random() < 0.3:
                genealogy.mark_eliminated(
                    tree_id,
                    age=random.randint(10, 50),
                    reason=random.choice(["low_fitness", "pruned", "old_age"])
                )
            
            tree_id += 1
    
    # Get statistics
    stats = genealogy.get_lineage_statistics()
    print(f"📊 Genealogy Statistics:")
    print(f"   Total trees: {stats['total_trees']}")
    print(f"   Alive: {stats['alive_trees']}")
    print(f"   Dead: {stats['dead_trees']}")
    print(f"   Avg peak fitness: {stats['avg_peak_fitness']:.2f}")
    print(f"   Generations: {stats['total_generations']}")
    print(f"   Creation methods: {stats['creation_methods']}")
    print()
    
    # Find most successful lineage
    if genealogy.lineages:
        result = genealogy.find_most_successful_lineage()
        if result:
            root_id, lineage = result
            print(f"🏆 Most successful lineage:")
            print(f"   Root tree: {root_id}")
            print(f"   Lineage: {lineage}")
            print()
    
    # Get family tree for a random tree
    if genealogy.lineages:
        sample_id = random.choice(list(genealogy.lineages.keys()))
        family = genealogy.get_family_tree(sample_id)
        print(f"👨‍👩‍👧‍👦 Family tree for tree {sample_id}:")
        print(f"   Ancestors: {family['ancestors'][:5]}{'...' if len(family['ancestors']) > 5 else ''}")
        print(f"   Descendants: {family['descendants'][:5]}{'...' if len(family['descendants']) > 5 else ''}")
        print(f"   Siblings: {family['siblings'][:5]}{'...' if len(family['siblings']) > 5 else ''}")
        print()
    
    # Export genealogy
    export_path = Path("/tmp/genealogy/genealogy_graph.json")
    genealogy.export_genealogy_graph(export_path)
    print(f"💾 Genealogy exported to {export_path}")
    
    # Save genealogy
    genealogy.save()
    print(f"💾 Genealogy data saved")
    
    print("\n✅ Demo 2 Complete: Genealogy tracking and analysis working\n")


def demo_realtime_monitoring():
    """Demo 3: Real-time Monitoring"""
    print("\n" + "="*70)
    print("DEMO 3: Real-time Evolutionary Monitoring")
    print("="*70 + "\n")
    
    # Create monitor
    monitor = EvolutionMonitor(
        window_size=50,
        save_dir=Path("/tmp/monitoring")
    )
    
    # Register alert callback
    def alert_handler(alert_type: str, alert: dict):
        print(f"   🚨 ALERT: {alert['message']}")
    
    monitor.register_alert_callback(alert_handler)
    
    print("Simulating evolutionary progress with monitoring...\n")
    
    # Simulate evolution with varying metrics
    for step in range(100):
        generation = step // 10
        
        # Simulate changing metrics
        if step < 30:
            # Early phase: growing population, improving fitness
            num_trees = 3 + step // 5
            avg_fitness = 3.0 + step * 0.1
            diversity = 0.6 + step * 0.01
        elif step < 60:
            # Mid phase: stable
            num_trees = 9
            avg_fitness = 6.0 + random.uniform(-0.2, 0.3)
            diversity = 0.7 + random.uniform(-0.1, 0.1)
        else:
            # Late phase: some challenges
            num_trees = max(2, 9 - (step - 60) // 5)  # Population declining
            avg_fitness = 6.0 + random.uniform(-0.5, 0.2)  # Fitness stagnating
            diversity = max(0.05, 0.7 - (step - 60) * 0.02)  # Diversity dropping
        
        forest_state = {
            "num_trees": num_trees,
            "alive_trees": num_trees,
            "dead_trees": step // 10,
            "avg_fitness": avg_fitness,
            "max_fitness": avg_fitness + random.uniform(0.5, 2.0),
            "min_fitness": max(0, avg_fitness - random.uniform(1.0, 2.0)),
            "fitness_std": random.uniform(0.5, 1.5),
            "architecture_diversity": diversity,
            "fitness_diversity": random.uniform(0.3, 0.7),
        }
        
        evolution_state = {
            "mutations": random.randint(0, 3),
            "crossovers": random.randint(0, 2),
            "births": random.randint(0, 2),
            "deaths": random.randint(0, 1),
        }
        
        season = ["spring", "summer", "autumn", "winter"][generation % 4]
        
        monitor.record_snapshot(
            generation=generation,
            step=step,
            forest_state=forest_state,
            evolution_state=evolution_state,
            season=season
        )
    
    # Print status
    print("\n")
    monitor.print_status(detailed=True)
    
    # Get statistics
    stats = monitor.get_statistics()
    print(f"📈 Monitoring Statistics:")
    print(f"   Total snapshots: {stats['total_snapshots']}")
    print(f"   Runtime: {stats['runtime_seconds']:.1f}s")
    print(f"   Generations/sec: {stats['generations_per_second']:.2f}")
    print(f"   Fitness improvement: {stats['fitness_improvement']:.2f}")
    print(f"   Total alerts: {stats['alerts_count']}")
    if stats['alerts_by_type']:
        print(f"   Alerts by type: {stats['alerts_by_type']}")
    print()
    
    # Export metrics
    export_path = Path("/tmp/monitoring/metrics.json")
    monitor.export_metrics(export_path)
    print(f"💾 Metrics exported to {export_path}")
    
    print("\n✅ Demo 3 Complete: Real-time monitoring captures all events\n")


def demo_automl_orchestrator():
    """Demo 4: AutoML Orchestration"""
    print("\n" + "="*70)
    print("DEMO 4: AutoML Orchestrator (Testing, Validation, Alerts)")
    print("="*70 + "\n")
    
    # Create forest
    forest = ForestEcosystem(input_dim=10, hidden_dim=32, max_trees=5)
    
    # Generate test data
    test_X, test_y = generate_synthetic_data(10, 50)
    test_data = [(test_X[i:i+5], test_y[i:i+5]) for i in range(0, 50, 5)]
    
    # Create AutoML components
    gen_tester = ContinuousGeneralizationTester(test_frequency=10)
    reg_validator = RegressionValidator(regression_threshold=0.15, checkpoint_frequency=20)
    metric_alerter = MetricAlerter()
    
    # Create orchestrator
    automl = AutoMLOrchestrator(
        generalization_tester=gen_tester,
        regression_validator=reg_validator,
        metric_alerter=metric_alerter,
    )
    
    print("Running AutoML orchestration...\n")
    
    # Simulate training loop
    for step in range(50):
        # Simulate training
        X, y = generate_synthetic_data(10, 8)
        optimizer = torch.optim.Adam(forest.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        optimizer.zero_grad()
        pred, _, _ = forest.forward_forest(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        # Collect metrics
        metrics = {
            "train_loss": loss.item(),
            "avg_fitness": sum(t.fitness for t in forest.trees) / len(forest.trees) if forest.trees else 0,
            "diversity": random.uniform(0.2, 0.8),
            "alive_trees": len(forest.trees),
        }
        
        # Run AutoML step
        automl.step(forest, metrics, test_data if step % 10 == 0 else None)
        
        # Print progress occasionally
        if step % 20 == 0:
            print(f"Step {step}: Loss={loss.item():.4f}, Trees={len(forest.trees)}")
    
    print("\n")
    automl.print_status()
    
    print("✅ Demo 4 Complete: AutoML orchestration with testing and validation\n")


def demo_integrated_system():
    """Demo 5: Fully Integrated System"""
    print("\n" + "="*70)
    print("DEMO 5: Fully Integrated System (All Components)")
    print("="*70 + "\n")
    
    print("Creating integrated system with all Phase 3 & 4 components...\n")
    
    # Create all components
    forest = ForestEcosystem(input_dim=8, hidden_dim=24, max_trees=6)
    seasonal_cycle = SeasonalCycle(steps_per_season=30)
    seasonal_evo = SeasonalEvolution()
    genealogy = GenealogyTracker()
    monitor = EvolutionMonitor(window_size=100)
    automl = AutoMLOrchestrator()
    
    print("Components initialized:")
    print("  ✅ ForestEcosystem")
    print("  ✅ SeasonalCycle")
    print("  ✅ SeasonalEvolution")
    print("  ✅ GenealogyTracker")
    print("  ✅ EvolutionMonitor")
    print("  ✅ AutoMLOrchestrator")
    print()
    
    # Register initial trees in genealogy
    for i, tree in enumerate(forest.trees):
        genealogy.register_tree(
            tree_id=tree.id,
            generation=0,
            creation_method="random",
            birth_fitness=tree.fitness
        )
    
    print("Running integrated evolution cycle...\n")
    
    # Generate test data
    test_X, test_y = generate_synthetic_data(8, 40)
    test_data = [(test_X[i:i+4], test_y[i:i+4]) for i in range(0, 40, 4)]
    
    # Run integrated loop
    for step in range(100):
        # Get seasonal parameters
        season = seasonal_cycle.current_season
        evo_params = seasonal_evo.get_evolutionary_params(season)
        
        # Training step
        X, y = generate_synthetic_data(8, 6)
        optimizer = torch.optim.Adam(forest.parameters(), lr=evo_params.get("learning_rate", 0.01))
        criterion = torch.nn.MSELoss()
        
        optimizer.zero_grad()
        pred, _, _ = forest.forward_forest(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        # Update genealogy
        for tree in forest.trees:
            genealogy.update_fitness(tree.id, tree.fitness)
        
        # Collect metrics
        forest_state = {
            "num_trees": len(forest.trees),
            "alive_trees": len(forest.trees),
            "dead_trees": 0,
            "avg_fitness": sum(t.fitness for t in forest.trees) / len(forest.trees) if forest.trees else 0,
            "max_fitness": max((t.fitness for t in forest.trees), default=0),
            "min_fitness": min((t.fitness for t in forest.trees), default=0),
            "fitness_std": 0.5,
            "architecture_diversity": random.uniform(0.4, 0.7),
            "fitness_diversity": random.uniform(0.3, 0.6),
        }
        
        metrics = {
            "train_loss": loss.item(),
            "avg_fitness": forest_state["avg_fitness"],
            "diversity": forest_state["architecture_diversity"],
            "alive_trees": len(forest.trees),
        }
        
        # Monitor evolution
        monitor.record_snapshot(
            generation=step // 30,
            step=step,
            forest_state=forest_state,
            season=season
        )
        
        # AutoML orchestration
        automl.step(forest, metrics, test_data if step % 20 == 0 else None)
        
        # Advance season
        seasonal_cycle.step()
        
        # Print status occasionally
        if step % 30 == 0 and step > 0:
            print(f"\n--- Step {step} ---")
            print(f"Season: {season}")
            print(f"Trees: {len(forest.trees)}")
            print(f"Avg Fitness: {forest_state['avg_fitness']:.2f}")
            print(f"Loss: {loss.item():.4f}")
    
    # Final status
    print("\n\n" + "="*70)
    print("FINAL STATUS")
    print("="*70)
    
    print("\n📊 Genealogy:")
    gen_stats = genealogy.get_lineage_statistics()
    print(f"   Total trees tracked: {gen_stats['total_trees']}")
    print(f"   Generations: {gen_stats['total_generations']}")
    
    print("\n📈 Monitoring:")
    mon_stats = monitor.get_statistics()
    print(f"   Total snapshots: {mon_stats['total_snapshots']}")
    print(f"   Fitness improvement: {mon_stats['fitness_improvement']:.2f}")
    print(f"   Alerts: {mon_stats['alerts_count']}")
    
    print("\n🤖 AutoML:")
    automl_status = automl.get_status()
    print(f"   Tests run: {automl_status['generalization']['total_tests']}")
    print(f"   Checkpoints: {automl_status['regression']['total_checkpoints']}")
    print(f"   Alerts: {automl_status['alerts']['total']}")
    
    print("\n✅ Demo 5 Complete: All systems integrated successfully\n")


def main():
    """Run all Phase 4 demonstrations."""
    print("\n" + "="*70)
    print("NEURALFOREST PHASE 4 COMPREHENSIVE DEMONSTRATION")
    print("Automated Learning, Testing, and Monitoring")
    print("="*70)
    
    set_seed(42)
    
    # Run all demos
    demo_seasonal_evolution_integration()
    demo_genealogy_tracking()
    demo_realtime_monitoring()
    demo_automl_orchestrator()
    demo_integrated_system()
    
    print("\n" + "="*70)
    print("🎉 ALL PHASE 4 DEMOS COMPLETE!")
    print("="*70)
    print("\nSummary of Implemented Features:")
    print("  ✅ 1. Integration with dynamic environment/seasons system")
    print("  ✅ 2. Advanced genealogy visualization")
    print("  ✅ 3. Real-time monitoring of evolutionary progress")
    print("  ✅ 4. AutoML Forest orchestration")
    print("  ✅ 5. Continuous generalization tests")
    print("  ✅ 6. Automated regression and validator tests")
    print("  ✅ 7. Alerts and metric checking")
    print("\nPhase 3 Pending Items: COMPLETED ✅")
    print("Phase 4 Requirements: COMPLETED ✅")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
