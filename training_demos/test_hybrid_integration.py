"""
Smoke test for hybrid training components.
Tests the integration without requiring dataset downloads.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from NeuralForest import ForestEcosystem, DEVICE
from training_demos.layer_wise_optimizer import LayerWiseConfig, LayerWiseOptimizer
from training_demos.enhanced_task_head import EnhancedTaskHead

def test_integration():
    """Test full integration of hybrid training components."""
    print("=" * 70)
    print("Hybrid Training Components Integration Test")
    print("=" * 70)
    
    # Create forest
    print("\n1. Creating forest...")
    forest = ForestEcosystem(
        input_dim=3072,
        hidden_dim=128,
        max_trees=6
    ).to(DEVICE)
    
    # Plant some trees
    for _ in range(3):
        forest._plant_tree()
    
    # Initialize tree ages
    for i, tree in enumerate(forest.trees):
        tree.epoch_age = i * 5
        tree.fitness = 5.0 + i * 0.5
    
    print(f"   Trees: {forest.num_trees()}")
    print("   ✅ Forest created")
    
    # Create enhanced task head
    print("\n2. Creating enhanced task head...")
    task_head = EnhancedTaskHead(
        input_dim=128,
        hidden_dim=64,
        num_classes=10,
        dropout=0.2
    ).to(DEVICE)
    print("   ✅ Task head created")
    
    # Create layer-wise optimizer
    print("\n3. Creating layer-wise optimizer...")
    config = LayerWiseConfig(
        base_lr=0.01,
        half_life=60.0,
        fitness_aware=True,
        warmup_epochs=5,
        total_epochs=100
    )
    opt_factory = LayerWiseOptimizer(config)
    print("   ✅ Optimizer factory created")
    
    # Test optimizer creation for different epochs
    print("\n4. Testing dynamic optimizer creation...")
    for epoch in [0, 5, 30]:
        print(f"\n   Epoch {epoch}:")
        optimizer = opt_factory.create_optimizer(forest, task_head, epoch)
        
        # Count parameter groups
        num_groups = len(optimizer.param_groups)
        total_params = sum(
            sum(p.numel() for p in group['params']) 
            for group in optimizer.param_groups
        )
        
        print(f"     Parameter groups: {num_groups}")
        print(f"     Total parameters: {total_params:,}")
        
        # Test a forward pass
        batch_size = 16
        x = torch.randn(batch_size, 3072).to(DEVICE)
        
        # Extract forest features
        T = forest.num_trees()
        scores = forest.router(x, num_trees=T)
        weights = torch.softmax(scores, dim=1)
        
        features = []
        for tree in forest.trees:
            h = tree.trunk(x)
            if tree.use_residual and tree.skip_proj is not None:
                skip = tree.skip_proj(x)
                if skip.shape == h.shape:
                    h = h + skip
            features.append(h)
        
        feature_stack = torch.stack(features, dim=1)
        weighted_features = (feature_stack * weights.unsqueeze(-1)).sum(dim=1)
        
        # Task head forward
        logits = task_head(weighted_features)
        
        # Loss
        targets = torch.randint(0, 10, (batch_size,)).to(DEVICE)
        loss = task_head.get_loss(logits, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"     Loss: {loss.item():.4f}")
        print(f"     ✅ Forward/backward pass successful")
    
    # Test tree age updates
    print("\n5. Testing tree age updates...")
    initial_ages = [tree.epoch_age for tree in forest.trees]
    opt_factory.update_tree_ages(forest)
    updated_ages = [tree.epoch_age for tree in forest.trees]
    
    print(f"   Initial ages: {initial_ages}")
    print(f"   Updated ages: {updated_ages}")
    
    for init, upd in zip(initial_ages, updated_ages):
        assert upd == init + 1, "Age should increment by 1"
    
    print("   ✅ Tree age updates working")
    
    # Test LR summary printing
    print("\n6. Testing LR summary...")
    optimizer = opt_factory.create_optimizer(forest, task_head, 10)
    opt_factory.print_lr_summary(optimizer)
    opt_factory.log_tree_lr_factors(forest)
    print("   ✅ LR summary printing working")
    
    # Test unique tree seeding
    print("\n7. Testing unique tree seeding...")
    
    def plant_tree_with_unique_seed(forest, base_seed, tree_id):
        """Plant tree with unique initialization seed."""
        tree_seed = base_seed + tree_id * 1000
        
        # Save RNG state
        cpu_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state()
        
        # Set unique seed
        torch.manual_seed(tree_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(tree_seed)
        
        # Plant tree
        forest._plant_tree()
        
        # Restore RNG state
        torch.set_rng_state(cpu_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_state)
    
    initial_trees = forest.num_trees()
    plant_tree_with_unique_seed(forest, 42, 999)
    final_trees = forest.num_trees()
    
    assert final_trees == initial_trees + 1, "Should plant one new tree"
    print(f"   Trees: {initial_trees} → {final_trees}")
    print("   ✅ Unique tree seeding working")
    
    print("\n" + "=" * 70)
    print("✅ All integration tests passed!")
    print("=" * 70)

if __name__ == "__main__":
    test_integration()
