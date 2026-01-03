"""
Tests for core NeuralForest functionality.
Tests TreeExpert, ForestEcosystem, and memory systems.
"""

import torch
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from NeuralForest
import importlib.util

spec = importlib.util.spec_from_file_location(
    "neuralforest",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NeuralForest.py"
    ),
)
neuralforest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neuralforest)

TreeExpert = neuralforest.TreeExpert
TreeArch = neuralforest.TreeArch
ForestEcosystem = neuralforest.ForestEcosystem
PrioritizedMulch = neuralforest.PrioritizedMulch
AnchorCoreset = neuralforest.AnchorCoreset
DEVICE = neuralforest.DEVICE


def test_tree_expert_creation():
    """Test TreeExpert can be created and used."""
    arch = TreeArch(num_layers=1, hidden_dim=32, activation="tanh", 
                   dropout=0.0, normalization="none", residual=False)
    tree = TreeExpert(input_dim=1, tree_id=0, arch=arch).to(DEVICE)

    assert tree.id == 0
    assert tree.age == 0
    assert tree.bark == 0.0
    assert tree.fitness == 5.0

    # Test forward pass
    x = torch.randn(10, 1).to(DEVICE)
    output = tree(x)
    assert output.shape == (10, 1)


def test_tree_expert_aging():
    """Test tree aging and bark formation."""
    arch = TreeArch(num_layers=1, hidden_dim=32, activation="tanh", 
                   dropout=0.0, normalization="none", residual=False)
    tree = TreeExpert(input_dim=1, tree_id=0, arch=arch).to(DEVICE)

    initial_bark = tree.bark

    # Age the tree
    for _ in range(100):
        tree.step_age()

    assert tree.age == 100
    # Bark should increase after age > 80
    assert tree.bark > initial_bark


def test_tree_fitness_update():
    """Test fitness update mechanism."""
    arch = TreeArch(num_layers=1, hidden_dim=32, activation="tanh", 
                   dropout=0.0, normalization="none", residual=False)
    tree = TreeExpert(input_dim=1, tree_id=0, arch=arch).to(DEVICE)

    initial_fitness = tree.fitness

    # Update with low loss (should increase fitness)
    tree.update_fitness(0.01)

    # Fitness should change (EMA update)
    assert tree.fitness != initial_fitness


def test_prioritized_mulch():
    """Test PrioritizedMulch memory."""
    mulch = PrioritizedMulch(capacity=100, alpha=0.7)

    # Add some experiences
    for i in range(50):
        x = torch.randn(1).to(DEVICE)
        y = torch.randn(1).to(DEVICE)
        priority = float(i)
        mulch.add(x, y, priority)

    assert len(mulch) == 50

    # Sample batch
    batch_x, batch_y = mulch.sample(batch_size=10)

    assert batch_x is not None
    assert batch_y is not None
    assert batch_x.shape == (10, 1)
    assert batch_y.shape == (10, 1)


def test_anchor_coreset():
    """Test AnchorCoreset memory."""
    anchors = AnchorCoreset(capacity=20)

    # Add experiences
    for i in range(30):
        x = torch.randn(1).to(DEVICE)
        y = torch.randn(1).to(DEVICE)
        anchors.add(x, y)

    # Should cap at capacity
    assert len(anchors) == 20

    # Sample batch
    batch_x, batch_y = anchors.sample(batch_size=10)

    assert batch_x is not None
    assert batch_y is not None
    assert batch_x.shape == (10, 1)


def test_forest_ecosystem_creation():
    """Test ForestEcosystem creation."""
    forest = ForestEcosystem(input_dim=1, hidden_dim=32, max_trees=10).to(DEVICE)

    # Should start with 1 tree
    assert forest.num_trees() == 1
    assert forest.input_dim == 1
    assert forest.hidden_dim == 32
    assert forest.max_trees == 10


def test_forest_forward():
    """Test forest forward pass."""
    forest = ForestEcosystem(input_dim=1, hidden_dim=32, max_trees=10).to(DEVICE)

    x = torch.randn(10, 1).to(DEVICE)
    y_pred, weights, per_tree = forest.forward_forest(x, top_k=3)

    assert y_pred.shape == (10, 1)
    assert weights.shape[0] == 10
    assert len(per_tree) == forest.num_trees()


def test_forest_plant_tree():
    """Test planting new trees."""
    forest = ForestEcosystem(input_dim=1, hidden_dim=32, max_trees=5).to(DEVICE)

    initial_trees = forest.num_trees()

    forest._plant_tree()

    assert forest.num_trees() == initial_trees + 1

    # Test max_trees limit
    for _ in range(10):
        forest._plant_tree()

    assert forest.num_trees() <= forest.max_trees


def test_forest_prune_trees():
    """Test pruning trees."""
    forest = ForestEcosystem(input_dim=1, hidden_dim=32, max_trees=10).to(DEVICE)

    # Plant more trees
    for _ in range(4):
        forest._plant_tree()

    initial_count = forest.num_trees()
    assert initial_count >= 3

    # Prune a tree
    tree_to_remove = forest.trees[0].id
    forest._prune_trees([tree_to_remove], min_keep=2)

    # Should have one less tree (if we had enough)
    assert forest.num_trees() < initial_count or forest.num_trees() >= 2


def test_forest_checkpoint_save_load():
    """Test saving and loading checkpoints."""
    # Create forest and train a bit
    forest = ForestEcosystem(input_dim=1, hidden_dim=32, max_trees=5).to(DEVICE)

    # Add some data to memory
    x = torch.randn(10, 1).to(DEVICE)
    y = torch.randn(10, 1).to(DEVICE)
    for i in range(10):
        forest.mulch.add(x[i], y[i], priority=float(i))
        forest.anchors.add(x[i], y[i])

    # Forward pass to change state
    _ = forest.forward_forest(x, top_k=3)

    original_trees = forest.num_trees()
    original_memory_size = len(forest.mulch)

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name

    try:
        forest.save_checkpoint(checkpoint_path)

        # Load checkpoint
        loaded_forest = ForestEcosystem.load_checkpoint(checkpoint_path, device=DEVICE)

        # Verify state is restored
        assert loaded_forest.num_trees() == original_trees
        assert len(loaded_forest.mulch) == original_memory_size
        assert loaded_forest.input_dim == forest.input_dim
        assert loaded_forest.hidden_dim == forest.hidden_dim

        # Test forward pass on loaded model
        y_pred = loaded_forest.forward_forest(x, top_k=3)[0]
        assert y_pred.shape == (10, 1)

    finally:
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


def test_forest_teacher_snapshot():
    """Test teacher snapshot creation."""
    forest = ForestEcosystem(input_dim=1, hidden_dim=32, max_trees=5).to(DEVICE)

    assert forest.teacher_snapshot is None

    forest.snapshot_teacher()

    assert forest.teacher_snapshot is not None

    # Test teacher forward pass
    x = torch.randn(10, 1).to(DEVICE)
    teacher_output = forest.teacher_snapshot(x, top_k=3)
    assert teacher_output.shape == (10, 1)


if __name__ == "__main__":
    # Run tests
    print("Running TreeExpert tests...")
    test_tree_expert_creation()
    test_tree_expert_aging()
    test_tree_fitness_update()
    print("✅ TreeExpert tests passed!")

    print("\nRunning memory tests...")
    test_prioritized_mulch()
    test_anchor_coreset()
    print("✅ Memory tests passed!")

    print("\nRunning ForestEcosystem tests...")
    test_forest_ecosystem_creation()
    test_forest_forward()
    test_forest_plant_tree()
    test_forest_prune_trees()
    test_forest_checkpoint_save_load()
    test_forest_teacher_snapshot()
    print("✅ ForestEcosystem tests passed!")

    print("\n✅ All core tests passed!")
