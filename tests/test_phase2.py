"""
Test suite for Phase 2 components: Groves and Mycelium.
"""

import unittest
from unittest.mock import patch
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

        with patch.object(grove, "_connect_to_similar_trees"):
            clone_id = grove.plant_specialist("classification")
        self.assertIsNotNone(clone_id)
        clone_tree = grove.trees[-1]
        clone_tree.load_state_dict(first_tree.state_dict())

        self.assertLess(
            grove._tree_param_distance(first_tree, clone_tree),
            grove.min_mycelium_distance,
        )

        Grove._connect_to_similar_trees(grove, clone_tree)

        self.assertEqual(grove.mycelium_connections[first_tree.id], [])
        self.assertEqual(grove.mycelium_connections[clone_tree.id], [])

    def test_mycelium_prefers_same_specialization_before_cross_specialization(self):
        """Matching specializations should be prioritized before cross-links."""
        same_specialization_strength = 1.0
        grove = Grove(modality="image", input_dim=16, hidden_dim=8, max_trees=4)
        grove.min_mycelium_distance = 0.0
        grove.max_mycelium_neighbors = 1

        same_id = grove.plant_specialist("classification")
        self.assertIsNotNone(same_id)

        with patch.object(grove, "_connect_to_similar_trees"):
            cross_id = grove.plant_specialist("segmentation")
            new_id = grove.plant_specialist("classification")
        self.assertIsNotNone(cross_id)
        self.assertIsNotNone(new_id)

        same_tree = grove.trees[0]
        cross_tree = grove.trees[1]
        new_tree = grove.trees[2]

        distances = {
            same_tree.id: 0.2,
            cross_tree.id: 0.1,
        }
        with patch.object(
            grove,
            "_tree_param_distance",
            side_effect=lambda source, target: distances[target.id],
        ), patch("groves.base_grove.random.random", return_value=0.0):
            Grove._connect_to_similar_trees(grove, new_tree)

        self.assertEqual(
            grove.mycelium_connections[new_tree.id],
            [(same_tree.id, same_specialization_strength)],
        )
        self.assertEqual(
            grove.mycelium_connections[same_tree.id],
            [(new_tree.id, same_specialization_strength)],
        )
        self.assertEqual(grove.mycelium_connections[cross_tree.id], [])

    def test_mycelium_cross_specialization_links_respect_probability(self):
        """Cross-specialization links should follow the configured probability."""
        cross_specialization_strength = 0.3
        for random_value, expected_connected in ((0.0, True), (0.9, False)):
            grove = Grove(modality="image", input_dim=16, hidden_dim=8, max_trees=3)
            grove.min_mycelium_distance = 0.0
            grove.max_mycelium_neighbors = 1
            grove.cross_specialization_link_probability = 0.25

            first_id = grove.plant_specialist("classification")
            self.assertIsNotNone(first_id)

            with patch.object(grove, "_connect_to_similar_trees"):
                second_id = grove.plant_specialist("segmentation")
            self.assertIsNotNone(second_id)

            first_tree = grove.trees[0]
            second_tree = grove.trees[1]

            with patch.object(grove, "_tree_param_distance", return_value=0.1), patch(
                "groves.base_grove.random.random", return_value=random_value
            ):
                Grove._connect_to_similar_trees(grove, second_tree)

            if expected_connected:
                self.assertEqual(
                    grove.mycelium_connections[second_tree.id],
                    [(first_tree.id, cross_specialization_strength)],
                )
                self.assertEqual(
                    grove.mycelium_connections[first_tree.id],
                    [(second_tree.id, cross_specialization_strength)],
                )
            else:
                self.assertEqual(grove.mycelium_connections[second_tree.id], [])
                self.assertEqual(grove.mycelium_connections[first_tree.id], [])


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
