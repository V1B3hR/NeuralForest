"""
Phase 6 Demo: Self-Evolution, Meta-Learning, Cooperation & Environmental Adaptation

Demonstrates the consciousness, goal management, architecture search,
self-improvement, tree cooperation, and environmental simulation capabilities
of NeuralForest.
"""

import torch
import sys
import os
import numpy as np
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from NeuralForest import ForestEcosystem, DEVICE
from consciousness import ForestConsciousness, GoalManager
from evolution import (
    TreeArchitectureSearch,
    SelfImprovementLoop,
    CooperationSystem,
    EnvironmentalSimulator,
    ClimateType,
    StressorType,
    DataDistributionShift,
)


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
        description="Increase average tree fitness to 7.0",
    )

    goal2 = manager.create_goal(
        name="Grow Forest Size",
        target_metric="num_trees",
        target_value=10,
        priority=2,
        description="Expand forest to 10 trees",
    )

    # Create third goal (optimize memory)
    manager.create_goal(
        name="Optimize Memory",
        target_metric="memory_efficiency",
        target_value=0.8,
        priority=1,
        description="Achieve 80% memory efficiency",
    )

    print(f"Created {len(manager)} goals:")
    for goal in manager.get_active_goals():
        print(f"  • {goal.name} (priority: {goal.priority})")
        print(f"    Target: {goal.target_metric} >= {goal.target_value}")
        print(f"    Status: {goal.progress():.0%} complete\n")

    # Simulate progress updates
    print("Simulating progress...")
    manager.update_progress({"forest_fitness": 5.0, "num_trees": 6})
    print(f"  Progress: {goal1.name} = {goal1.progress():.0%}")
    print(f"  Progress: {goal2.name} = {goal2.progress():.0%}")

    # Complete a goal
    manager.update_progress({"num_trees": 10})

    # Show summary
    summary = manager.get_progress_summary()
    print("\nGoal Summary:")
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
        priority=3,
    )

    print("Initial forest state:")
    print(f"  Trees: {forest.num_trees()}")
    print(f"  Memory: {len(forest.mulch)} items")

    # Perform reflection
    print("\nPerforming self-reflection...")
    reflection = consciousness.reflect()

    print(f"  Overall fitness: {reflection['overall_fitness']:.2f}")
    print(
        f"  Memory utilization: {reflection['memory_usage']['mulch_utilization']:.1%}"
    )
    print(f"  Knowledge gaps found: {len(reflection['knowledge_gaps'])}")

    tree_health = reflection["tree_health"]
    print("  Tree health:")
    print(f"    Average fitness: {tree_health['average_fitness']:.2f}")
    print(f"    Weak trees: {tree_health['weak_trees']}")
    print(f"    Strong trees: {tree_health['strong_trees']}")

    # Run evolution cycle
    print("\nRunning evolution cycle...")
    result = consciousness.evolve()

    print(f"  Evolution step: {result['evolution_step']}")
    print(f"  Actions taken: {result['actions_taken']}")

    for action_result in result["results"]:
        print(f"    • {action_result['action_type']}: {action_result['result']}")

    # Get status report
    print("\nGenerating status report...")
    report = consciousness.get_status_report()

    print(f"  Evolution step: {report['consciousness']['evolution_step']}")
    print(
        f"  Action success rate: {report['consciousness']['action_success_rate']:.1%}"
    )
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

    print("\nBest architecture found:")
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

        print("  Baseline metrics:")
        for key, val in result["baseline_metrics"].items():
            if isinstance(val, float):
                print(f"    {key}: {val:.3f}")
            else:
                print(f"    {key}: {val}")

        print("  New metrics:")
        for key, val in result["new_metrics"].items():
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
    for imp in summary["recent_improvements"][-3:]:
        action = imp.get("action", "unknown")
        success = imp.get("success", False)
        message = imp.get("message", "no message")
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
        priority=3,
    )

    consciousness.goals.create_goal(
        name="Grow to 8 Trees", target_metric="num_trees", target_value=8, priority=2
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
            evolution_result["reflection"]["overall_fitness"]
        )

        # Report
        print(f"  Forest size: {forest.num_trees()} trees")
        print(
            f"  Average fitness: {evolution_result['reflection']['overall_fitness']:.2f}"
        )
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

    print("Evolution:")
    print(f"  Total steps: {report['consciousness']['evolution_step']}")
    print(f"  Success rate: {report['consciousness']['action_success_rate']:.1%}")

    print("\nForest:")
    print(f"  Trees: {report['forest']['forest_size']}")
    print(f"  Fitness: {report['forest']['overall_fitness']:.2f}")

    print("\nGoals:")
    print(f"  Active: {report['goals']['active_goals']}")
    print(f"  Completed: {report['goals']['completed_goals']}")
    print(f"  Completion rate: {report['goals']['completion_rate']:.0%}")

    print("\n✅ Integrated evolution working correctly")


def demo_tree_cooperation():
    """Demonstrate tree cooperation system."""
    print_section("Tree Cooperation & Federated Learning")
    
    # Create cooperation system
    cooperation = CooperationSystem()
    print("✓ Cooperation system initialized")
    
    # Enable communication for trees
    for tree_id in range(5):
        cooperation.enable_tree_communication(tree_id, can_send=True, can_receive=True)
    print(f"✓ Communication enabled for 5 trees")
    
    # Test communication
    print("\n1. Tree Communication")
    print("-" * 50)
    cooperation.communication.send_message(
        sender_id=0,
        receiver_id=1,
        message_type='knowledge',
        content={'tip': 'Try lower learning rate'},
        priority=2.0
    )
    cooperation.communication.broadcast_message(
        sender_id=1,
        receiver_ids=[2, 3, 4],
        message_type='alert',
        content={'warning': 'Environment becoming harsh'},
        priority=3.0
    )
    print("  ✓ Sent unicast message from tree 0 to tree 1")
    print("  ✓ Broadcast alert from tree 1 to trees 2, 3, 4")
    
    # Receive messages
    messages = cooperation.communication.receive_messages(receiver_id=1, max_messages=1)
    print(f"  ✓ Tree 1 received {len(messages)} message(s)")
    if messages:
        msg = messages[0]
        print(f"    - From tree {msg.sender_id}: {msg.message_type}")
    
    # Test federated learning
    print("\n2. Federated Learning")
    print("-" * 50)
    
    # Create mock tree parameters
    participating_trees = {}
    for tree_id in range(3):
        # Mock parameters
        params = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
        }
        participating_trees[tree_id] = {
            'parameters': params,
            'fitness': 3.0 + tree_id * 0.5 + np.random.rand()
        }
    
    # Coordinate federated learning
    result = cooperation.coordinate_learning(
        participating_trees,
        coordination_type='federated'
    )
    
    print(f"  ✓ Federated learning round complete")
    print(f"    - Participants: {result['num_participants']}")
    print(f"    - Success: {result['success']}")
    print(f"    - Parameters averaged: {len(result.get('averaged_parameters', {}))} layers")
    
    # Test knowledge distillation coordination
    print("\n3. Knowledge Distillation")
    print("-" * 50)
    
    result = cooperation.coordinate_learning(
        participating_trees,
        coordination_type='distillation'
    )
    
    print(f"  ✓ Distillation coordination complete")
    print(f"    - Teacher tree: {result['teacher_tree']}")
    print(f"    - Student trees: {result['student_trees']}")
    
    # Get cooperation summary
    print("\n4. Cooperation Summary")
    print("-" * 50)
    summary = cooperation.get_cooperation_summary()
    print(f"  Total messages: {summary['communication_stats']['total_messages']}")
    print(f"  Federated rounds: {summary['federated_rounds']}")
    print(f"  Transfer operations: {summary['transfer_operations']}")
    
    print("\n✅ Tree cooperation system working correctly")


def demo_environmental_simulation():
    """Demonstrate environmental simulation and adaptation."""
    print_section("Environmental Simulation & Adaptation")
    
    # Create environmental simulator
    env = EnvironmentalSimulator(
        initial_climate=ClimateType.TEMPERATE,
        stressor_probability=0.3,
        climate_change_rate=0.1
    )
    print("✓ Environmental simulator initialized")
    print(f"  Initial climate: {env.current_state.climate.value}")
    
    # Simulate environmental steps
    print("\n1. Environmental Evolution (10 steps)")
    print("-" * 50)
    
    for step in range(10):
        state = env.step()
        
        if step % 3 == 0:
            print(f"\n  Step {step}:")
            print(f"    Climate: {state.climate.value}")
            print(f"    Temperature: {state.temperature:.2f}")
            print(f"    Resources: {state.resource_availability:.2f}")
            print(f"    Data quality: {state.data_quality:.2f}")
            print(f"    Competition: {state.competition_level:.2f}")
            if state.active_stressors:
                stressors = [s.value for s in state.active_stressors]
                print(f"    Active stressors: {', '.join(stressors)}")
    
    # Test climate changes
    print("\n2. Climate Variation")
    print("-" * 50)
    
    climates = [ClimateType.TROPICAL, ClimateType.ARCTIC, ClimateType.DESERT]
    for climate in climates:
        env.set_climate(climate)
        state = env.step()
        print(f"  {climate.value.capitalize()}:")
        print(f"    Resources: {state.resource_availability:.2f}")
        print(f"    Quality: {state.data_quality:.2f}")
    
    # Test environmental stressors
    print("\n3. Environmental Stressors")
    print("-" * 50)
    
    # Reset to temperate
    env.set_climate(ClimateType.TEMPERATE)
    
    stressors = [StressorType.DROUGHT, StressorType.FLOOD, StressorType.DISEASE]
    for stressor in stressors:
        env.trigger_stressor(stressor)
        state = env.step()
        severity = env._calculate_severity()
        print(f"  {stressor.value.capitalize()}:")
        print(f"    Resources: {state.resource_availability:.2f}")
        print(f"    Quality: {state.data_quality:.2f}")
        print(f"    Severity: {severity}")
    
    # Test data transformation
    print("\n4. Environmental Effects on Data")
    print("-" * 50)
    
    # Create test data
    original_x = torch.randn(100, 2)
    original_y = torch.randn(100, 1)
    
    # Apply environmental effects
    modified_x, modified_y = env.apply_to_data(original_x, original_y)
    
    print(f"  Original data: {original_x.shape[0]} samples")
    print(f"  Modified data: {modified_x.shape[0]} samples")
    print(f"  Sample reduction: {(1 - modified_x.shape[0] / original_x.shape[0]) * 100:.1f}%")
    
    # Data distribution shift
    print("\n5. Data Distribution Shift")
    print("-" * 50)
    
    shift_types = ['gradual', 'sudden', 'cyclical']
    for shift_type in shift_types:
        shifter = DataDistributionShift(shift_type=shift_type, shift_rate=0.02)
        
        # Simulate several steps
        for _ in range(10):
            test_data = torch.randn(50, 2)
            shifted_data, _ = shifter.apply_shift(test_data)
        
        info = shifter.get_shift_info()
        print(f"  {shift_type.capitalize()}:")
        print(f"    Time steps: {info['time_step']}")
        print(f"    Shifts recorded: {info['num_shifts']}")
    
    # Get environmental summary
    print("\n6. Environmental Summary")
    print("-" * 50)
    summary = env.get_state_summary()
    print(f"  Current climate: {summary['climate']}")
    print(f"  Resource availability: {summary['resource_availability']:.2f}")
    print(f"  Data quality: {summary['data_quality']:.2f}")
    print(f"  Overall severity: {summary['severity']}")
    
    print("\n✅ Environmental simulation system working correctly")


def main():
    """Run all Phase 6 demonstrations."""
    print("\n" + "=" * 70)
    print("  NEURALFOREST PHASE 6: COOPERATION & ENVIRONMENTAL ADAPTATION")
    print("=" * 70)

    try:
        demo_goal_management()
        demo_forest_consciousness()
        demo_architecture_search()
        demo_self_improvement()
        demo_integrated_evolution()
        demo_tree_cooperation()
        demo_environmental_simulation()

        print("\n" + "=" * 70)
        print("  ✅ PHASE 6 COMPLETE - ALL SYSTEMS OPERATIONAL")
        print("=" * 70)
        print("\n🌲 The forest cooperates, adapts, and thrives! 🌲\n")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NeuralForest Phase 6 Demo')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Set device if specified (though demos use DEVICE from NeuralForest)
    if args.device:
        print(f"Using device: {args.device}")
    
    main()
