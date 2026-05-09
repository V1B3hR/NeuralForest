"""
Test suite for Phase 2 components: Groves and Mycelium.
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groves import VisualGrove, AudioGrove, TextGrove
from groves.base_grove import SpecialistTree
from mycelium import KnowledgeTransfer


class TestSpecialistTree(unittest.TestCase):
    """Test SpecialistTree functionality."""

    def test_tree_creation(self):
        """Test creating a specialist tree."""
        tree = SpecialistTree(
            input_dim=512,
            hidden_dim=64,
            tree_id=0,
            specialization="classification",
            modality="image",
        )

        self.assertEqual(tree.id, 0)
        self.assertEqual(tree.specialization, "classification")
        self.assertEqual(tree.modality, "image")
        self.assertEqual(tree.age, 0)
        self.assertEqual(tree.fitness, 5.0)

    def test_tree_forward(self):
        """Test forward pass through tree."""
        tree = SpecialistTree(
            input_dim=512,
            hidden_dim=64,
            tree_id=0,
            specialization="classification",
            modality="image",
        )

        x = torch.randn(4, 512)
        output = tree(x)

        self.assertEqual(output.shape, (4, 1))

    def test_tree_aging(self):
        """Test tree aging mechanism."""
        tree = SpecialistTree(
            input_dim=512,
            hidden_dim=64,
            tree_id=0,
            specialization="classification",
            modality="image",
        )

        initial_bark = tree.bark
        for _ in range(100):
            tree.step_age()

        self.assertEqual(tree.age, 100)
        self.assertGreater(tree.bark, initial_bark)


class TestGrove(unittest.TestCase):
    """Test Grove functionality."""

    def test_grove_creation(self):
        """Test creating a grove."""
        grove = VisualGrove(input_dim=512, hidden_dim=64, max_trees=12)

        self.assertEqual(grove.modality, "image")
        self.assertGreater(grove.num_trees(), 0)
        self.assertLessEqual(grove.num_trees(), grove.max_trees)

    def test_grove_forward(self):
        """Test forward pass through grove."""
        grove = VisualGrove(input_dim=512, hidden_dim=64, max_trees=8)

        x = torch.randn(4, 512)
        output, weights = grove(x, top_k=3)

        self.assertEqual(output.shape, (4, 1))
        self.assertEqual(weights.shape, (4, grove.num_trees()))

    def test_plant_specialist(self):
        """Test planting new specialist trees."""
        grove = AudioGrove(input_dim=512, hidden_dim=64, max_trees=12)

        initial_trees = grove.num_trees()
        tree_id = grove.plant_specialist("speaker_recognition")

        self.assertIsNotNone(tree_id)
        self.assertEqual(grove.num_trees(), initial_trees + 1)

    def test_grove_stats(self):
        """Test grove statistics."""
        grove = TextGrove(input_dim=512, hidden_dim=64, max_trees=8)

        stats = grove.get_grove_stats()

        self.assertIn("modality", stats)
        self.assertIn("num_trees", stats)
        self.assertIn("trees", stats)
        self.assertEqual(stats["modality"], "text")


class TestLitterAbsorption(unittest.TestCase):
    """Test litter absorption via PrioritizedMulch."""

    def test_litter_absorption_loss_empty_mulch(self):
        """Returns zero loss when mulch has no features."""

        class MockMulch:
            pass

        student_features = torch.randn(4, 64)
        loss = KnowledgeTransfer.litter_absorption_loss(
            student_features, MockMulch(), batch_size=4
        )
        self.assertEqual(loss.item(), 0.0)

    def test_litter_absorption_loss_with_features(self):
        """Returns valid loss when mulch has features."""
        from NeuralForest import PrioritizedMulch

        mulch = PrioritizedMulch(capacity=100)
        for _ in range(20):
            x = torch.randn(4)
            y = torch.randn(1)
            feat = torch.randn(64)
            mulch.add(x, y, priority=1.0, features=feat)

        student_features = torch.randn(4, 64)
        loss = KnowledgeTransfer.litter_absorption_loss(student_features, mulch, batch_size=4)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0.0)


class TestKnowledgeTransfer(unittest.TestCase):
    """Test KnowledgeTransfer utilities."""

    def test_distillation_loss(self):
        """Test distillation loss calculation."""
        teacher_output = torch.randn(4, 1)
        student_output = torch.randn(4, 1)

        loss = KnowledgeTransfer.distillation_loss(
            teacher_output, student_output, temperature=2.0
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)  # Loss can be 0 or positive

    def test_feature_alignment_loss(self):
        """Test feature alignment loss."""
        source_features = torch.randn(4, 64)
        target_features = torch.randn(4, 64)

        loss = KnowledgeTransfer.feature_alignment_loss(
            source_features, target_features, margin=0.5
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
