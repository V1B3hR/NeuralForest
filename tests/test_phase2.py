"""
Test suite for Phase 2 components: Groves and Mycelium.
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groves import Grove, VisualGrove, AudioGrove, TextGrove
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

    def test_mycelium_connections_respect_max_neighbors(self):
        """Each tree should stay within the configured mycelium neighbor cap."""
        torch.manual_seed(0)
        grove = Grove(modality="image", input_dim=16, hidden_dim=8, max_trees=6)
        grove.min_mycelium_distance = 0.0
        grove.max_mycelium_neighbors = 2

        for _ in range(grove.max_trees):
            tree_id = grove.plant_specialist("classification")
            self.assertIsNotNone(tree_id)

        for tree in grove.trees:
            self.assertLessEqual(
                len(grove.mycelium_connections[tree.id]),
                grove.max_mycelium_neighbors,
            )

    def test_mycelium_connections_skip_trees_below_min_distance(self):
        """Trees below the minimum parameter distance should not link."""
        grove = Grove(modality="image", input_dim=16, hidden_dim=8, max_trees=4)
        grove.min_mycelium_distance = 1e-6

        first_id = grove.plant_specialist("classification")
        self.assertIsNotNone(first_id)
        first_tree = grove.trees[0]

        clone_tree = SpecialistTree(
            input_dim=16,
            hidden_dim=8,
            tree_id=grove.tree_counter,
            specialization="classification",
            modality="image",
        )
        clone_tree.load_state_dict(first_tree.state_dict())
        grove.trees.append(clone_tree)
        grove.tree_counter += 1

        grove._connect_to_similar_trees(clone_tree)

        self.assertEqual(grove.mycelium_connections[first_tree.id], [])
        self.assertEqual(grove.mycelium_connections[clone_tree.id], [])


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


class TestFloweringFlora(unittest.TestCase):
    """Phase 2: Flowering Flora — bloom signals and symbiotic clusters."""

    def _make_blooming_tree(self, tree_id: int, specialization: str = "classification") -> "SpecialistTree":
        from groves.base_grove import SpecialistTree
        tree = SpecialistTree(
            input_dim=512,
            hidden_dim=64,
            tree_id=tree_id,
            specialization=specialization,
            modality="image",
        )
        tree.fitness = 9.0
        tree.expertise_score = 0.8
        return tree

    def test_is_blooming_true_when_fit_and_expert(self):
        """A tree with high fitness and expertise should be blooming."""
        from groves.base_grove import SpecialistTree
        tree = self._make_blooming_tree(0)
        self.assertTrue(tree.is_blooming)

    def test_is_blooming_false_low_fitness(self):
        """A tree with low fitness should not bloom."""
        from groves.base_grove import SpecialistTree
        tree = SpecialistTree(input_dim=512, hidden_dim=64, tree_id=0,
                              specialization="classification", modality="image")
        tree.fitness = 3.0
        tree.expertise_score = 0.9
        self.assertFalse(tree.is_blooming)

    def test_is_blooming_false_low_expertise(self):
        """A tree with low expertise should not bloom."""
        from groves.base_grove import SpecialistTree
        tree = SpecialistTree(input_dim=512, hidden_dim=64, tree_id=0,
                              specialization="classification", modality="image")
        tree.fitness = 9.0
        tree.expertise_score = 0.1
        self.assertFalse(tree.is_blooming)

    def test_bloom_signal_keys(self):
        """bloom_signal() should contain expected keys."""
        tree = self._make_blooming_tree(1)
        signal = tree.bloom_signal()
        for key in ("tree_id", "modality", "specialization", "fitness",
                    "expertise_score", "age", "is_blooming"):
            self.assertIn(key, signal)
        self.assertTrue(signal["is_blooming"])

    def test_get_blooming_trees(self):
        """Grove.get_blooming_trees() returns only blooming trees."""
        grove = VisualGrove(input_dim=512, hidden_dim=64, max_trees=8)
        # Force all trees to bloom
        for tree in grove.trees:
            tree.fitness = 9.0
            tree.expertise_score = 0.8
        blooming = grove.get_blooming_trees()
        self.assertEqual(len(blooming), grove.num_trees())

    def test_get_blooming_trees_none_when_unfit(self):
        """No trees bloom when all fitness values are low."""
        grove = VisualGrove(input_dim=512, hidden_dim=64, max_trees=6)
        for tree in grove.trees:
            tree.fitness = 1.0
            tree.expertise_score = 0.0
        self.assertEqual(grove.get_blooming_trees(), [])

    def test_form_symbiotic_clusters(self):
        """Symbiotic clusters group blooming trees by specialization."""
        grove = VisualGrove(input_dim=512, hidden_dim=64, max_trees=8)
        # Force all trees to bloom
        for tree in grove.trees:
            tree.fitness = 9.0
            tree.expertise_score = 0.8
        clusters = grove.form_symbiotic_clusters()
        self.assertIsInstance(clusters, list)
        # Each cluster has the expected keys
        for cluster in clusters:
            for key in ("specialization", "tree_ids", "avg_fitness",
                        "avg_expertise", "size"):
                self.assertIn(key, cluster)

    def test_no_clusters_when_not_blooming(self):
        """form_symbiotic_clusters() returns empty list when no trees bloom."""
        grove = VisualGrove(input_dim=512, hidden_dim=64, max_trees=6)
        for tree in grove.trees:
            tree.fitness = 1.0
            tree.expertise_score = 0.0
        self.assertEqual(grove.form_symbiotic_clusters(), [])


if __name__ == "__main__":
    unittest.main()
