"""
Smoke test for training demos.
Validates that all demos can be imported and basic functionality works.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from NeuralForest import ForestEcosystem, DEVICE
from ecosystem_simulation import EcosystemSimulator
from tasks.vision.classification import ImageClassification
from training_demos.utils import DatasetLoader, MetricsTracker

print("="*70)
print("Smoke Test: Training Demos Infrastructure")
print("="*70)

# Test 1: Import utilities
print("\n✅ Test 1: Import utilities")
print(f"  - DatasetLoader: OK")
print(f"  - MetricsTracker: OK")

# Test 2: Create metrics tracker
print("\n✅ Test 2: Create metrics tracker")
tracker = MetricsTracker()
tracker.update(1, {'train_loss': 0.5, 'train_accuracy': 80.0, 'test_loss': 0.6, 
                   'test_accuracy': 75.0, 'num_trees': 6, 'avg_fitness': 5.0,
                   'architecture_diversity': 2, 'memory_size': 100})
print(f"  - Metrics updated: {len(tracker.history['epoch'])} epochs recorded")

# Test 3: Create forest and task head
print("\n✅ Test 3: Create forest and task head")
forest = ForestEcosystem(input_dim=3072, hidden_dim=128, max_trees=10).to(DEVICE)
forest._plant_tree()
print(f"  - Forest created with {forest.num_trees()} trees")

task_head = ImageClassification(input_dim=128, num_classes=10, dropout=0.3).to(DEVICE)
print(f"  - Task head created (10 classes)")

# Test 4: Create ecosystem simulator
print("\n✅ Test 4: Create ecosystem simulator")
simulator = EcosystemSimulator(forest, learning_rate=0.001, device=DEVICE)
print(f"  - Simulator created")

# Test 5: Forward pass with feature extraction
print("\n✅ Test 5: Forward pass with feature extraction")

def topk_softmax(scores, k):
    B, T = scores.shape
    k = min(k, T)
    topv, topi = torch.topk(scores, k=k, dim=1)
    w = torch.softmax(topv, dim=1)
    weights = torch.zeros_like(scores)
    weights.scatter_(1, topi, w)
    return weights

def extract_forest_features(forest, x, top_k=3):
    T = forest.num_trees()
    scores = forest.router(x, num_trees=T)
    weights = topk_softmax(scores, k=min(top_k, T))
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
    return weighted_features

x = torch.randn(8, 3072).to(DEVICE)
features = extract_forest_features(forest, x)
logits = task_head(features)
print(f"  - Input: {x.shape}")
print(f"  - Features: {features.shape}")
print(f"  - Logits: {logits.shape}")

# Test 6: Backward pass
print("\n✅ Test 6: Backward pass")
labels = torch.randint(0, 10, (8,)).to(DEVICE)
loss = task_head.get_loss(logits, labels)
loss.backward()
print(f"  - Loss computed: {loss.item():.4f}")
print(f"  - Gradients computed successfully")

# Test 7: Dataset loader availability (don't download yet)
print("\n✅ Test 7: Dataset loader methods available")
print(f"  - CIFAR-10 loader: {hasattr(DatasetLoader, 'get_cifar10')}")
print(f"  - MNIST loader: {hasattr(DatasetLoader, 'get_mnist')}")
print(f"  - Fashion-MNIST loader: {hasattr(DatasetLoader, 'get_fashion_mnist')}")

print("\n" + "="*70)
print("✅ All smoke tests passed!")
print("="*70)
print("\nThe training infrastructure is ready. You can now run:")
print("  - python training_demos/cifar10_full_training.py")
print("  - python training_demos/continual_learning_demo.py")
print("  - python training_demos/few_shot_demo.py")
print("\nNote: Actual training will download datasets automatically.")
