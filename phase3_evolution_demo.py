"""
Phase 3 Demo: Evolution, Generational Progress, and Legacy Management

This demo showcases:
1. Evolutionary mechanisms (crossover, mutation, fitness-based selection)
2. Hall-of-Fame repository for top architectures
3. Tree Graveyard for eliminated tree archival
4. Resurrection mechanism for reintroducing trees
5. Post-mortem analysis and evolutionary insights
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from NeuralForest import ForestEcosystem, set_seed
from evolution import TreeArchitectureSearch, TreeGraveyard, TreeRecord


def demo_basic_graveyard():
    """Demo 1: Basic Tree Graveyard functionality."""
    print("\n" + "=" * 70)
    print("Demo 1: Tree Graveyard - Basic Archival and Query")
    print("=" * 70)
    
    set_seed(42)
    
    # Create forest with graveyard enabled
    forest = ForestEcosystem(input_dim=4, hidden_dim=32, max_trees=10, enable_graveyard=True)
    
    print(f"\n✓ Created forest with {forest.num_trees()} trees")
    print(f"✓ Graveyard enabled: {forest.graveyard is not None}")
    
    # Plant more trees
    for _ in range(5):
        forest._plant_tree()
    
    print(f"✓ Planted more trees, total: {forest.num_trees()}")
    
    # Simulate some training to give trees different fitness
    for tree in forest.trees:
        tree.fitness = np.random.uniform(1.0, 8.0)
        tree.age = np.random.randint(1, 20)
    
    print("\nTree fitness before pruning:")
    for i, tree in enumerate(forest.trees):
        print(f"  Tree {tree.id}: fitness={tree.fitness:.2f}, age={tree.age}")
    
    # Prune weak trees (will be archived)
    weak_trees = sorted(forest.trees, key=lambda t: t.fitness)[:2]
    weak_ids = [t.id for t in weak_trees]
    
    print(f"\n✓ Pruning {len(weak_ids)} weak trees: {weak_ids}")
    forest._prune_trees(weak_ids, reason="low_fitness")
    
    print(f"✓ Forest now has {forest.num_trees()} trees")
    
    # Check graveyard
    if forest.graveyard:
        stats = forest.graveyard.get_stats()
        print(f"\n📊 Graveyard Statistics:")
        print(f"  Total eliminated: {stats['total_eliminated']}")
        print(f"  Avg fitness at elimination: {stats['avg_fitness_at_elimination']:.2f}")
        print(f"  Avg age at elimination: {stats['avg_age_at_elimination']:.1f}")
        
        # Query records
        print(f"\n✓ Querying graveyard records...")
        for record in forest.graveyard.records:
            print(f"  Tree {record.tree_id}: fitness={record.final_fitness:.2f}, "
                  f"age={record.age_at_elimination}, reason={record.elimination_reason}")
    
    print("\n✅ Demo 1 complete!")
    return forest


def demo_resurrection():
    """Demo 2: Tree Resurrection mechanism."""
    print("\n" + "=" * 70)
    print("Demo 2: Tree Resurrection - Bringing Back Eliminated Trees")
    print("=" * 70)
    
    set_seed(43)
    
    # Create forest
    forest = ForestEcosystem(input_dim=4, hidden_dim=32, max_trees=10, enable_graveyard=True)
    
    # Plant and train trees
    for _ in range(7):
        forest._plant_tree()
    
    print(f"\n✓ Created forest with {forest.num_trees()} trees")
    
    # Assign fitness
    for tree in forest.trees:
        tree.fitness = np.random.uniform(2.0, 9.0)
        tree.age = np.random.randint(5, 30)
    
    # Identify and prune some trees
    sorted_trees = sorted(forest.trees, key=lambda t: t.fitness)
    trees_to_prune = sorted_trees[:3]
    prune_ids = [t.id for t in trees_to_prune]
    
    print(f"\nPruning trees: {prune_ids}")
    print("Pruned tree fitness:")
    for tree in trees_to_prune:
        print(f"  Tree {tree.id}: fitness={tree.fitness:.2f}")
    
    forest._prune_trees(prune_ids, reason="fitness_selection")
    print(f"\n✓ Forest now has {forest.num_trees()} trees")
    
    # Check resurrection candidates
    if forest.graveyard:
        print("\n🔍 Finding resurrection candidates...")
        candidates = forest.graveyard.get_resurrection_candidates(
            min_fitness=3.0,
            limit=5
        )
        
        print(f"Found {len(candidates)} candidates:")
        for cand in candidates:
            print(f"  Tree {cand.tree_id}: fitness={cand.final_fitness:.2f}, "
                  f"age={cand.age_at_elimination}")
        
        # Resurrect the best candidate
        if candidates:
            print(f"\n♻️  Attempting resurrection of tree {candidates[0].tree_id}...")
            resurrected = forest.resurrect_tree(tree_id=candidates[0].tree_id)
            
            if resurrected:
                print(f"✓ Successfully resurrected tree as new ID {resurrected.id}")
                print(f"  Original fitness: {candidates[0].final_fitness:.2f}")
                print(f"  Current fitness: {resurrected.fitness:.2f}")
                print(f"✓ Forest now has {forest.num_trees()} trees")
            else:
                print("✗ Resurrection failed")
        
        # Stats after resurrection
        stats = forest.graveyard.get_stats()
        print(f"\n📊 Graveyard Stats:")
        print(f"  Total eliminated: {stats['total_eliminated']}")
        print(f"  Resurrection count: {stats['resurrection_count']}")
    
    print("\n✅ Demo 2 complete!")
    return forest


def demo_evolutionary_nas():
    """Demo 3: Evolutionary NAS with Hall of Fame."""
    print("\n" + "=" * 70)
    print("Demo 3: Evolutionary NAS - Architecture Search & Hall of Fame")
    print("=" * 70)
    
    set_seed(44)
    
    # Create forest and add some data
    forest = ForestEcosystem(input_dim=4, hidden_dim=32, max_trees=8, enable_graveyard=True)
    
    # Generate synthetic data for NAS
    X = torch.randn(200, 4)
    y = (X[:, 0] ** 2 + X[:, 1] - X[:, 2] * 0.5 + X[:, 3]).unsqueeze(1)
    
    # Add to forest memory
    for i in range(len(X)):
        forest.mulch.add(X[i], y[i], priority=1.0)
        if i < 50:  # Add some to anchors
            forest.anchors.add(X[i], y[i])
    
    print(f"\n✓ Added {len(forest.mulch)} samples to forest memory")
    print(f"✓ Added {len(forest.anchors)} anchor samples")
    
    # Create NAS config for quick demo
    from evolution.architecture_search import NASConfig
    nas_config = NASConfig(
        generations=5,  # Reduced for demo
        population_size=8,
        train_steps=30,
        val_steps=10,
        batch_size=16,
        early_stop_patience=3,
    )
    
    # Run evolutionary search
    print("\n🧬 Starting evolutionary architecture search...")
    nas = TreeArchitectureSearch(forest, config=nas_config)
    best_arch = nas.search()
    
    print("\n✓ Search complete!")
    print(f"📈 Hall of Fame (top architectures):")
    for i, (arch, fitness) in enumerate(nas.hall_of_fame[:3]):
        arch_dict = arch.to_dict() if hasattr(arch, 'to_dict') else arch
        print(f"\n  Rank {i+1}: Fitness = {fitness:.4f}")
        print(f"    Architecture: {arch_dict}")
    
    # Plant tree with best architecture
    print(f"\n🌱 Planting new tree with best discovered architecture...")
    result = nas.plant_tree_with_best_arch()
    print(f"✓ Planted tree with architecture: {result['planted_architecture']}")
    print(f"  Best fitness: {result['best_fitness']:.4f}")
    print(f"✓ Forest now has {forest.num_trees()} trees")
    
    print("\n✅ Demo 3 complete!")
    return forest, nas


def demo_post_mortem_analysis():
    """Demo 4: Post-mortem analysis and evolutionary insights."""
    print("\n" + "=" * 70)
    print("Demo 4: Post-Mortem Analysis - Understanding Evolution")
    print("=" * 70)
    
    set_seed(45)
    
    # Create forest
    forest = ForestEcosystem(input_dim=4, hidden_dim=32, max_trees=12, enable_graveyard=True)
    
    # Plant many trees with various architectures
    from NeuralForest import TreeArch
    
    architectures = [
        TreeArch(num_layers=1, hidden_dim=16, activation="tanh", dropout=0.0, normalization="none", residual=False),
        TreeArch(num_layers=2, hidden_dim=32, activation="relu", dropout=0.1, normalization="layer", residual=True),
        TreeArch(num_layers=3, hidden_dim=64, activation="gelu", dropout=0.2, normalization="batch", residual=True),
        TreeArch(num_layers=2, hidden_dim=16, activation="tanh", dropout=0.0, normalization="none", residual=False),
        TreeArch(num_layers=1, hidden_dim=32, activation="relu", dropout=0.1, normalization="layer", residual=False),
    ]
    
    print(f"\n✓ Planting trees with diverse architectures...")
    for arch in architectures:
        forest._plant_tree(arch=arch)
    
    print(f"✓ Forest has {forest.num_trees()} trees")
    
    # Simulate multiple generations of evolution
    print("\n🔄 Simulating evolutionary cycles...")
    for gen in range(3):
        forest.current_generation = gen
        
        # Assign random fitness (simulating training)
        for tree in forest.trees:
            base_fitness = np.random.uniform(2.0, 8.0)
            # Add some architecture bias (larger = potentially better but riskier)
            arch_bias = tree.arch.num_layers * 0.3 - tree.arch.dropout * 2.0
            tree.fitness = max(0.5, base_fitness + arch_bias + np.random.normal(0, 0.5))
            tree.age += 1
        
        # Prune weakest trees
        if forest.num_trees() > 5:
            sorted_trees = sorted(forest.trees, key=lambda t: t.fitness)
            weak_count = min(2, forest.num_trees() - 5)
            weak_ids = [t.id for t in sorted_trees[:weak_count]]
            
            print(f"  Generation {gen}: Pruning {len(weak_ids)} weak trees")
            reasons = ["low_fitness", "poor_adaptation", "resource_inefficient"]
            forest._prune_trees(weak_ids, reason=np.random.choice(reasons))
        
        # Plant new trees
        if forest.num_trees() < 8:
            new_arch = np.random.choice(architectures)
            forest._plant_tree(arch=new_arch)
    
    # Analyze graveyard
    if forest.graveyard:
        print("\n📊 Performing post-mortem analysis...")
        
        analysis = forest.graveyard.analyze_elimination_patterns()
        
        print(f"\n  Total eliminated trees: {analysis['total_records']}")
        print(f"\n  Fitness at elimination:")
        print(f"    Mean: {analysis['fitness_stats']['mean']:.2f}")
        print(f"    Min: {analysis['fitness_stats']['min']:.2f}")
        print(f"    Max: {analysis['fitness_stats']['max']:.2f}")
        
        print(f"\n  Age at elimination:")
        print(f"    Mean: {analysis['age_stats']['mean']:.1f}")
        print(f"    Min: {analysis['age_stats']['min']}")
        print(f"    Max: {analysis['age_stats']['max']}")
        
        print(f"\n  Elimination reasons:")
        for reason, count in analysis['elimination_reasons'].items():
            print(f"    {reason}: {count}")
        
        # Identify dead-ends
        print("\n🚫 Identifying architectural dead-ends...")
        dead_ends = forest.graveyard.identify_dead_ends(threshold=3.0)
        
        if dead_ends:
            print(f"  Found {len(dead_ends)} dead-end patterns:")
            for i, de in enumerate(dead_ends[:3]):
                print(f"\n    Dead-end {i+1}:")
                print(f"      Occurrences: {de['count']}")
                print(f"      Avg fitness: {de['avg_fitness']:.2f}")
                print(f"      Architecture: {de['architecture']}")
        else:
            print("  No clear dead-ends identified yet")
        
        # Identify successful patterns
        print("\n✨ Identifying successful patterns...")
        successful = forest.graveyard.get_successful_patterns(threshold=5.0)
        
        if successful:
            print(f"  Found {len(successful)} successful patterns:")
            for i, sp in enumerate(successful[:3]):
                print(f"\n    Pattern {i+1}:")
                print(f"      Occurrences: {sp['count']}")
                print(f"      Avg fitness: {sp['avg_fitness']:.2f}")
                print(f"      Architecture: {sp['architecture']}")
        else:
            print("  No strongly successful patterns yet")
    
    print("\n✅ Demo 4 complete!")
    return forest


def demo_full_lifecycle():
    """Demo 5: Complete lifecycle with evolution, elimination, and resurrection."""
    print("\n" + "=" * 70)
    print("Demo 5: Complete Lifecycle - Evolution to Resurrection")
    print("=" * 70)
    
    set_seed(46)
    
    # Create forest
    forest = ForestEcosystem(input_dim=4, hidden_dim=32, max_trees=10, enable_graveyard=True)
    
    # Add training data
    X = torch.randn(300, 4)
    y = (X[:, 0] ** 2 + torch.sin(X[:, 1]) + X[:, 2] * X[:, 3]).unsqueeze(1)
    
    for i in range(len(X)):
        forest.mulch.add(X[i], y[i], priority=abs(y[i].item()))
        if i < 100:
            forest.anchors.add(X[i], y[i])
    
    print(f"\n✓ Initialized forest with data")
    print(f"  Memory size: {len(forest.mulch)}")
    print(f"  Anchor size: {len(forest.anchors)}")
    
    # Lifecycle simulation
    print("\n🔄 Starting evolutionary lifecycle...")
    
    for generation in range(5):
        forest.current_generation = generation
        print(f"\n--- Generation {generation} ---")
        
        # Plant new trees
        if forest.num_trees() < 8:
            for _ in range(2):
                forest._plant_tree()
            print(f"  Planted trees, total: {forest.num_trees()}")
        
        # Simulate training
        for tree in forest.trees:
            tree.fitness = np.random.uniform(1.0, 10.0)
            tree.age += 1
        
        print(f"  Tree fitness: min={min(t.fitness for t in forest.trees):.2f}, "
              f"max={max(t.fitness for t in forest.trees):.2f}")
        
        # Selection and elimination
        if generation > 1 and forest.num_trees() > 4:
            sorted_trees = sorted(forest.trees, key=lambda t: t.fitness)
            eliminate_count = max(1, forest.num_trees() // 4)
            eliminate_ids = [t.id for t in sorted_trees[:eliminate_count]]
            
            print(f"  Eliminating {len(eliminate_ids)} trees")
            forest._prune_trees(eliminate_ids, reason="natural_selection")
        
        # Consider resurrection every other generation
        if generation % 2 == 0 and generation > 0 and forest.graveyard:
            candidates = forest.graveyard.get_resurrection_candidates(
                min_fitness=5.0,
                limit=1
            )
            if candidates and forest.num_trees() < forest.max_trees:
                print(f"  💫 Resurrecting tree {candidates[0].tree_id}...")
                forest.resurrect_tree(tree_id=candidates[0].tree_id)
    
    # Final summary
    print("\n" + "=" * 70)
    print("Lifecycle Summary")
    print("=" * 70)
    
    print(f"\n🌳 Final forest state:")
    print(f"  Active trees: {forest.num_trees()}")
    print(f"  Generation: {forest.current_generation}")
    
    if forest.graveyard:
        stats = forest.graveyard.get_stats()
        print(f"\n📚 Graveyard statistics:")
        print(f"  Total eliminated: {stats['total_eliminated']}")
        print(f"  Resurrections: {stats['resurrection_count']}")
        print(f"  Avg age at elimination: {stats['avg_age_at_elimination']:.1f}")
        print(f"  Avg fitness at elimination: {stats['avg_fitness_at_elimination']:.2f}")
        
        print(f"\n  Elimination reasons:")
        for reason, count in stats['elimination_reasons'].items():
            print(f"    {reason}: {count}")
    
    print("\n✅ Demo 5 complete!")
    return forest


if __name__ == "__main__":
    print("=" * 70)
    print("NeuralForest Phase 3 Demo Suite")
    print("Evolution, Generational Progress, and Legacy Management")
    print("=" * 70)
    
    # Run all demos
    try:
        demo_basic_graveyard()
        demo_resurrection()
        demo_evolutionary_nas()
        demo_post_mortem_analysis()
        demo_full_lifecycle()
        
        print("\n" + "=" * 70)
        print("All Phase 3 demos completed successfully! ✅")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
