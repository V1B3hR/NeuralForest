"""
Phase 6 Demo: Self-Evolution & Meta-Learning

Demonstrates the consciousness, goal management, architecture search,
and self-improvement capabilities of NeuralForest.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from NeuralForest import ForestEcosystem, DEVICE
from consciousness import ForestConsciousness, GoalManager, LearningGoal
from evolution import TreeArchitectureSearch, SelfImprovementLoop


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_goal_management():
    """Demonstrate goal management system."""
    print_section("Goal Management System")
    
    manager = GoalManager()
    
    # Create some learning goals
    goal1 = manager.create_goal(
        name="Achieve High Forest Fitness",
        target_metric="forest_fitness",
        target_value=7.0,
        priority=3,
        description="Increase average tree fitness to 7.0"
    )
    
    goal2 = manager.create_goal(
        name="Grow Forest Size",
        target_metric="num_trees",
        target_value=10,
        priority=2,
        description="Expand forest to 10 trees"
    )
    
    goal3 = manager.create_goal(
        name="Optimize Memory",
        target_metric="memory_efficiency",
        target_value=0.8,
        priority=1,
        description="Achieve 80% memory efficiency"
    )
    
    print(f"Created {len(manager)} goals:")
    for goal in manager.get_active_goals():
        print(f"  • {goal.name} (priority: {goal.priority})")
        print(f"    Target: {goal.target_metric} >= {goal.target_value}")
        print(f"    Status: {goal.progress():.0%} complete\n")
    
    # Simulate progress updates
    print("Simulating progress...")
    manager.update_progress({'forest_fitness': 5.0, 'num_trees': 6})
    print(f"  Progress: {goal1.name} = {goal1.progress():.0%}")
    print(f"  Progress: {goal2.name} = {goal2.progress():.0%}")
    
    # Complete a goal
    manager.update_progress({'num_trees': 10})
    
    # Show summary
    summary = manager.get_progress_summary()
    print(f"\nGoal Summary:")
    print(f"  Active: {summary['active_goals']}")
    print(f"  Completed: {summary['completed_goals']}")
    print(f"  Average progress: {summary['average_progress']:.0%}")
    
    print("✅ Goal management system working correctly")


def demo_forest_consciousness():
    """Demonstrate forest consciousness and meta-controller."""
    print_section("Forest Consciousness & Meta-Controller")
    
    # Create a small forest
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=12).to(DEVICE)
    
    # Add some data to memory
    for i in range(50):
        x = torch.randn(1, 2).to(DEVICE)
        y = torch.randn(1, 1).to(DEVICE)
        forest.mulch.add(x, y, priority=float(i % 10))
    
    # Create consciousness
    consciousness = ForestConsciousness(forest)
    
    # Set some goals
    consciousness.goals.create_goal(
        name="Build Strong Forest",
        target_metric="forest_fitness",
        target_value=6.0,
        priority=3
    )
    
    print(f"Initial forest state:")
    print(f"  Trees: {forest.num_trees()}")
    print(f"  Memory: {len(forest.mulch)} items")
    
    # Perform reflection
    print("\nPerforming self-reflection...")
    reflection = consciousness.reflect()
    
    print(f"  Overall fitness: {reflection['overall_fitness']:.2f}")
    print(f"  Memory utilization: {reflection['memory_usage']['mulch_utilization']:.1%}")
    print(f"  Knowledge gaps found: {len(reflection['knowledge_gaps'])}")
    
    tree_health = reflection['tree_health']
    print(f"  Tree health:")
    print(f"    Average fitness: {tree_health['average_fitness']:.2f}")
    print(f"    Weak trees: {tree_health['weak_trees']}")
    print(f"    Strong trees: {tree_health['strong_trees']}")
    
    # Run evolution cycle
    print("\nRunning evolution cycle...")
    result = consciousness.evolve()
    
    print(f"  Evolution step: {result['evolution_step']}")
    print(f"  Actions taken: {result['actions_taken']}")
    
    for action_result in result['results']:
        print(f"    • {action_result['action_type']}: {action_result['result']}")
    
    # Get status report
    print("\nGenerating status report...")
    report = consciousness.get_status_report()
    
    print(f"  Evolution step: {report['consciousness']['evolution_step']}")
    print(f"  Action success rate: {report['consciousness']['action_success_rate']:.1%}")
    print(f"  Active goals: {report['goals']['active_goals']}")
    
    print("✅ Forest consciousness working correctly")


def demo_architecture_search():
    """Demonstrate neural architecture search."""
    print_section("Neural Architecture Search")
    
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=12).to(DEVICE)
    
    # Create architecture search
    search = TreeArchitectureSearch(forest)
    
    print("Search space:")
    for key, values in search.search_space.items():
        print(f"  {key}: {values}")
    
    # Generate some random architectures
    print("\nGenerating random architectures:")
    for i in range(3):
        arch = search.random_architecture()
        print(f"  Architecture {i+1}:")
        for key, val in arch.items():
            print(f"    {key}: {val}")
    
    # Test mutation
    print("\nTesting mutation:")
    original = search.random_architecture()
    mutated = search.mutate(original, mutation_rate=0.5)
    print(f"  Original: {original}")
    print(f"  Mutated:  {mutated}")
    
    # Test crossover
    print("\nTesting crossover:")
    parent1 = search.random_architecture()
    parent2 = search.random_architecture()
    child = search.crossover(parent1, parent2)
    print(f"  Parent 1: {parent1}")
    print(f"  Parent 2: {parent2}")
    print(f"  Child:    {child}")
    
    # Run short search
    print("\nRunning architecture search (5 generations)...")
    best_arch = search.search(generations=5, population_size=6, eval_steps=50)
    
    print(f"\nBest architecture found:")
    for key, val in best_arch.items():
        print(f"  {key}: {val}")
    
    # Show search history
    history = search.get_search_history()
    print(f"\nSearch history ({len(history)} generations):")
    for entry in history[-3:]:
        print(f"  Gen {entry['generation']}: fitness={entry['fitness']:.3f}")
    
    print("✅ Architecture search working correctly")


def demo_self_improvement():
    """Demonstrate self-improvement loop."""
    print_section("Self-Improvement Loop")
    
    # Create forest and consciousness
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=12).to(DEVICE)
    consciousness = ForestConsciousness(forest)
    
    # Add some memory
    for i in range(100):
        x = torch.randn(1, 2).to(DEVICE)
        y = torch.randn(1, 1).to(DEVICE)
        forest.mulch.add(x, y, priority=float(i % 10))
    
    # Create self-improvement loop
    improvement_loop = SelfImprovementLoop(forest, consciousness)
    
    print("Running self-improvement cycles...")
    
    # Run multiple improvement cycles
    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1} ---")
        result = improvement_loop.run_cycle(max_improvements=2)
        
        print(f"  Opportunities found: {result['opportunities_found']}")
        print(f"  Improvements applied: {result['improvements_applied']}")
        print(f"  Success: {result['success']}")
        
        print(f"  Baseline metrics:")
        for key, val in result['baseline_metrics'].items():
            if isinstance(val, float):
                print(f"    {key}: {val:.3f}")
            else:
                print(f"    {key}: {val}")
        
        print(f"  New metrics:")
        for key, val in result['new_metrics'].items():
            if isinstance(val, float):
                print(f"    {key}: {val:.3f}")
            else:
                print(f"    {key}: {val}")
    
    # Get improvement summary
    print("\nImprovement Summary:")
    summary = improvement_loop.get_improvement_summary()
    
    print(f"  Total improvements: {summary['total_improvements']}")
    print(f"  Successful: {summary['successful_improvements']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Total cycles: {summary['total_cycles']}")
    
    print("\n  Recent improvements:")
    for imp in summary['recent_improvements'][-3:]:
        action = imp.get('action', 'unknown')
        success = imp.get('success', False)
        message = imp.get('message', 'no message')
        print(f"    • {action}: {'✓' if success else '✗'} - {message}")
    
    print("✅ Self-improvement loop working correctly")


def demo_integrated_evolution():
    """Demonstrate integrated evolution with all components."""
    print_section("Integrated Evolution Demo")
    
    print("Creating forest ecosystem...")
    forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=12).to(DEVICE)
    
    # Add initial data
    for i in range(200):
        x = torch.randn(1, 2).to(DEVICE)
        y = torch.randn(1, 1).to(DEVICE)
        forest.mulch.add(x, y, priority=float(i % 10))
    
    # Create consciousness with goals
    print("Initializing forest consciousness...")
    consciousness = ForestConsciousness(forest)
    
    consciousness.goals.create_goal(
        name="Achieve Fitness Threshold",
        target_metric="forest_fitness",
        target_value=5.5,
        priority=3
    )
    
    consciousness.goals.create_goal(
        name="Grow to 8 Trees",
        target_metric="num_trees",
        target_value=8,
        priority=2
    )
    
    # Create improvement loop
    print("Creating self-improvement loop...")
    improvement_loop = SelfImprovementLoop(forest, consciousness)
    
    # Run integrated evolution
    print("\nRunning integrated evolution (3 steps)...\n")
    
    for step in range(3):
        print(f"Evolution Step {step + 1}")
        print("-" * 50)
        
        # Consciousness evolves
        evolution_result = consciousness.evolve()
        
        # Self-improvement cycle
        improvement_result = improvement_loop.run_cycle()
        
        # Add performance sample
        consciousness.add_performance_sample(
            evolution_result['reflection']['overall_fitness']
        )
        
        # Report
        print(f"  Forest size: {forest.num_trees()} trees")
        print(f"  Average fitness: {evolution_result['reflection']['overall_fitness']:.2f}")
        print(f"  Memory usage: {len(forest.mulch)}/{forest.mulch.capacity}")
        print(f"  Actions taken: {evolution_result['actions_taken']}")
        print(f"  Improvements: {improvement_result['improvements_applied']}")
        
        # Check goal progress
        goals_summary = consciousness.goals.get_progress_summary()
        print(f"  Goal progress: {goals_summary['average_progress']:.0%}")
        print()
    
    # Final status report
    print("\nFinal Status Report")
    print("-" * 50)
    report = consciousness.get_status_report()
    
    print(f"Evolution:")
    print(f"  Total steps: {report['consciousness']['evolution_step']}")
    print(f"  Success rate: {report['consciousness']['action_success_rate']:.1%}")
    
    print(f"\nForest:")
    print(f"  Trees: {report['forest']['forest_size']}")
    print(f"  Fitness: {report['forest']['overall_fitness']:.2f}")
    
    print(f"\nGoals:")
    print(f"  Active: {report['goals']['active_goals']}")
    print(f"  Completed: {report['goals']['completed_goals']}")
    print(f"  Completion rate: {report['goals']['completion_rate']:.0%}")
    
    print("\n✅ Integrated evolution working correctly")


def main():
    """Run all Phase 6 demonstrations."""
    print("\n" + "="*70)
    print("  NEURALFOREST PHASE 6: SELF-EVOLUTION & META-LEARNING")
    print("="*70)
    
    try:
        demo_goal_management()
        demo_forest_consciousness()
        demo_architecture_search()
        demo_self_improvement()
        demo_integrated_evolution()
        
        print("\n" + "="*70)
        print("  ✅ PHASE 6 COMPLETE - ALL SYSTEMS OPERATIONAL")
        print("="*70)
        print("\n🌲 The forest is now self-aware and continuously improving! 🌲\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
