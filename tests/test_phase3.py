"""
Test suite for Phase 3 components: Canopy, routing, and attention.
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from canopy import (
    ForestCanopy,
    ModalityDetector,
    CanopyBalancer,
    CrossGroveAttention,
    GroveRouter,
)
from groves import VisualGrove, AudioGrove, TextGrove


class TestModalityDetector(unittest.TestCase):
    """Test ModalityDetector functionality."""

    def test_detector_creation(self):
        """Test creating modality detector."""
        detector = ModalityDetector(embedding_dim=512)

        self.assertEqual(detector.embedding_dim, 512)
        self.assertEqual(len(detector.modality_names), 4)

    def test_detector_forward(self):
        """Test forward pass through detector."""
        detector = ModalityDetector(embedding_dim=512)

        x = torch.randn(4, 512)
        probs = detector(x)

        self.assertIsInstance(probs, dict)
        self.assertEqual(len(probs), 4)

        # Check probabilities sum to ~1.0
        prob_sum = sum(probs[k][0].item() for k in probs)
        self.assertAlmostEqual(prob_sum, 1.0, places=5)

    def test_get_top_modality(self):
        """Test getting top modality."""
        detector = ModalityDetector(embedding_dim=512)

        x = torch.randn(4, 512)
        modalities = detector.get_top_modality(x, threshold=0.3)

        self.assertIsInstance(modalities, list)
        self.assertGreater(len(modalities), 0)


class TestGroveRouter(unittest.TestCase):
    """Test GroveRouter functionality."""

    def test_router_creation(self):
        """Test creating grove router."""
        router = GroveRouter(num_groves=4, embedding_dim=512)

        self.assertEqual(router.num_groves, 4)
        self.assertEqual(router.embedding_dim, 512)

    def test_router_forward(self):
        """Test forward pass through router."""
        router = GroveRouter(num_groves=4, embedding_dim=512)

        x = torch.randn(4, 512)
        scores = router(x)

        self.assertEqual(scores.shape, (4, 4))

    def test_get_top_groves(self):
        """Test getting top groves."""
        router = GroveRouter(num_groves=4, embedding_dim=512)

        x = torch.randn(4, 512)
        indices, weights = router.get_top_groves(x, top_k=2)

        self.assertEqual(indices.shape, (4, 2))
        self.assertEqual(weights.shape, (4, 2))

        # Check weights sum to ~1.0
        weight_sum = weights[0].sum().item()
        self.assertAlmostEqual(weight_sum, 1.0, places=5)


class TestCanopyBalancer(unittest.TestCase):
    """Test CanopyBalancer functionality."""

    def test_balancer_creation(self):
        """Test creating load balancer."""
        balancer = CanopyBalancer(target_utilization=0.7)

        self.assertEqual(balancer.target, 0.7)

    def test_record_usage(self):
        """Test recording tree usage."""
        balancer = CanopyBalancer()

        tree_ids = [0, 1, 2, 3]
        weights = torch.rand(4, 4)
        weights = weights / weights.sum(dim=1, keepdim=True)

        balancer.record_usage(weights, tree_ids)

        utilizations = balancer.get_all_utilizations()
        self.assertEqual(len(utilizations), 4)

    def test_balance_stats(self):
        """Test balance statistics."""
        balancer = CanopyBalancer()

        tree_ids = [0, 1, 2]
        for _ in range(10):
            weights = torch.rand(4, 3)
            weights = weights / weights.sum(dim=1, keepdim=True)
            balancer.record_usage(weights, tree_ids)

        stats = balancer.get_balance_stats()

        self.assertIn("num_trees", stats)
        self.assertIn("balance_score", stats)
        self.assertGreaterEqual(stats["balance_score"], 0.0)
        self.assertLessEqual(stats["balance_score"], 1.0)


class TestCrossGroveAttention(unittest.TestCase):
    """Test CrossGroveAttention functionality."""

    def test_attention_creation(self):
        """Test creating cross-grove attention."""
        attention = CrossGroveAttention(embed_dim=512, num_heads=8)

        self.assertEqual(attention.embed_dim, 512)
        self.assertEqual(attention.num_heads, 8)

    def test_attention_forward(self):
        """Test forward pass through attention."""
        attention = CrossGroveAttention(embed_dim=512, num_heads=8)

        grove_outputs = torch.randn(4, 3, 512)
        aggregated, attn_weights = attention(grove_outputs)

        self.assertEqual(aggregated.shape, (4, 512))
        self.assertEqual(attn_weights.shape, (4, 3, 3))

    def test_weighted_aggregation(self):
        """Test weighted aggregation."""
        attention = CrossGroveAttention(embed_dim=512, num_heads=8)

        grove_outputs = torch.randn(4, 3, 512)
        grove_weights = torch.rand(4, 3)

        aggregated = attention.forward_with_weights(grove_outputs, grove_weights)

        self.assertEqual(aggregated.shape, (4, 512))


class TestForestCanopy(unittest.TestCase):
    """Test ForestCanopy functionality."""

    def test_canopy_creation(self):
        """Test creating forest canopy."""
        groves = {
            "image": VisualGrove(input_dim=512, hidden_dim=64, max_trees=6),
            "audio": AudioGrove(input_dim=512, hidden_dim=64, max_trees=6),
        }

        canopy = ForestCanopy(grove_dict=groves, embedding_dim=512)

        self.assertEqual(len(canopy.groves), 2)
        self.assertEqual(canopy.embedding_dim, 512)

    def test_canopy_forward(self):
        """Test forward pass through canopy."""
        groves = {
            "image": VisualGrove(input_dim=512, hidden_dim=64, max_trees=6),
            "audio": AudioGrove(input_dim=512, hidden_dim=64, max_trees=6),
        }

        canopy = ForestCanopy(grove_dict=groves, embedding_dim=512)

        x = torch.randn(4, 512)
        output, routing_info = canopy(x, modality_hint="image", top_k_groves=2)

        self.assertEqual(output.shape, (4, 1))
        self.assertIn("modalities", routing_info)
        self.assertIn("groves_used", routing_info)

    def test_add_grove(self):
        """Test adding grove to canopy."""
        canopy = ForestCanopy(grove_dict=None, embedding_dim=512)

        initial_groves = len(canopy.groves)

        grove = TextGrove(input_dim=512, hidden_dim=64, max_trees=6)
        canopy.add_grove("text", grove)

        self.assertEqual(len(canopy.groves), initial_groves + 1)
        self.assertIn("text", canopy.groves)

    def test_route_summary(self):
        """Test routing summary."""
        groves = {
            "image": VisualGrove(input_dim=512, hidden_dim=64, max_trees=6),
        }

        canopy = ForestCanopy(grove_dict=groves, embedding_dim=512)

        x = torch.randn(4, 512)
        summary = canopy.route_summary(x)

        self.assertIn("detected_modality", summary)
        self.assertIn("modality_confidence", summary)
        self.assertIn("grove_routing_probs", summary)


if __name__ == "__main__":
    unittest.main()
