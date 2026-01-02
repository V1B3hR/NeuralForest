"""
Demo: Phase 3 - The Canopy (Advanced Routing & Attention)
Shows hierarchical routing, modality detection, and load balancing.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import canopy components
from canopy import (
    ForestCanopy,
    ModalityDetector,
    CanopyBalancer,
    CrossGroveAttention,
    GroveRouter,
)

# Import groves
from groves import VisualGrove, AudioGrove, TextGrove, VideoGrove


def demo_modality_detector():
    """Demo automatic modality detection."""
    print("\n=== Modality Detector Demo ===")

    # Create modality detector
    detector = ModalityDetector(embedding_dim=512)

    # Simulate features from different modalities
    batch_size = 4

    modalities_to_test = {
        "image": torch.randn(batch_size, 512),
        "audio": torch.randn(batch_size, 512) * 0.5,
        "text": torch.randn(batch_size, 512) * 2.0,
        "video": torch.randn(batch_size, 512) * 1.5,
    }

    print("Testing modality detection:")
    for modality_name, features in modalities_to_test.items():
        probs = detector(features)
        detected, confidence = detector.get_modality_confidence(features)

        print(f"\n  Input: Simulated {modality_name} features")
        print(f"  Detected: {detected} (confidence: {confidence:.3f})")
        print(
            f"  Probabilities: {', '.join([f'{k}: {v[0].item():.3f}' for k, v in probs.items()])}"
        )

    print("\n✅ Modality detector demo successful!")

    return detector


def demo_grove_router():
    """Demo grove-level routing."""
    print("\n=== Grove Router Demo ===")

    # Create grove router
    router = GroveRouter(num_groves=4, embedding_dim=512)

    # Simulate input
    batch_size = 4
    features = torch.randn(batch_size, 512)

    # Get routing scores
    scores = router(features)
    print(f"Routing scores shape: {scores.shape}")
    print(f"Scores for first sample: {scores[0]}")

    # Get top groves
    indices, weights = router.get_top_groves(features, top_k=2)
    print("\nTop 2 groves:")
    print(f"  Indices: {indices[0]}")
    print(f"  Weights: {weights[0]}")

    print("✅ Grove router demo successful!")

    return router


def demo_load_balancer():
    """Demo load balancing across trees."""
    print("\n=== Load Balancer Demo ===")

    # Create load balancer
    balancer = CanopyBalancer(target_utilization=0.7)

    # Simulate routing decisions
    num_trees = 6
    tree_ids = list(range(num_trees))

    print(f"Simulating routing decisions for {num_trees} trees...")

    # Simulate 100 routing decisions with imbalanced usage
    for _ in range(100):
        # Create imbalanced routing (some trees used more than others)
        weights = torch.rand(4, num_trees)
        # Make tree 0 and 1 more popular
        weights[:, 0] *= 2.0
        weights[:, 1] *= 1.5
        # Normalize
        weights = weights / weights.sum(dim=1, keepdim=True)

        balancer.record_usage(weights, tree_ids)

    # Get utilization stats
    utilizations = balancer.get_all_utilizations()
    print("\nTree utilizations:")
    for tree_id, util in sorted(utilizations.items()):
        status = (
            "overutilized"
            if util > 0.8
            else "underutilized" if util < 0.3 else "balanced"
        )
        print(f"  Tree {tree_id}: {util:.3f} ({status})")

    # Get balance statistics
    stats = balancer.get_balance_stats()
    print("\nBalance Statistics:")
    print(f"  Average utilization: {stats['avg_utilization']:.3f}")
    print(f"  Min utilization: {stats['min_utilization']:.3f}")
    print(f"  Max utilization: {stats['max_utilization']:.3f}")
    print(f"  Balance score: {stats['balance_score']:.3f} (1.0 is perfect)")

    # Get suggestions
    suggestions = balancer.suggest_actions()
    if suggestions:
        print("\nSuggested actions:")
        for suggestion in suggestions:
            print(f"  - {suggestion['action']}: {suggestion['reason']}")

    print("✅ Load balancer demo successful!")

    return balancer


def demo_cross_grove_attention():
    """Demo cross-grove attention for aggregation."""
    print("\n=== Cross-Grove Attention Demo ===")

    # Create attention module
    attention = CrossGroveAttention(embed_dim=512, num_heads=8)

    # Simulate outputs from 3 groves
    batch_size = 4
    num_groves = 3
    grove_outputs = torch.randn(batch_size, num_groves, 512)

    print(f"Grove outputs shape: {grove_outputs.shape}")

    # Apply cross-grove attention
    aggregated, attn_weights = attention(grove_outputs)

    print(f"Aggregated output shape: {aggregated.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")

    # Test weighted aggregation
    grove_weights = torch.tensor([0.5, 0.3, 0.2]).unsqueeze(0).repeat(batch_size, 1)
    weighted_output = attention.forward_with_weights(grove_outputs, grove_weights)

    print(f"Weighted aggregation output shape: {weighted_output.shape}")

    print("✅ Cross-grove attention demo successful!")

    return attention


def demo_forest_canopy():
    """Demo the full forest canopy system."""
    print("\n=== Forest Canopy System Demo ===")

    # Create groves
    groves = {
        "image": VisualGrove(input_dim=512, hidden_dim=64, max_trees=8),
        "audio": AudioGrove(input_dim=512, hidden_dim=64, max_trees=8),
        "text": TextGrove(input_dim=512, hidden_dim=64, max_trees=8),
        "video": VideoGrove(input_dim=512, hidden_dim=64, max_trees=8),
    }

    # Create forest canopy
    canopy = ForestCanopy(grove_dict=groves, embedding_dim=512, num_heads=8)

    print(f"Forest Canopy created with {len(groves)} groves")

    # Get canopy stats
    stats = canopy.get_canopy_stats()
    print("\nCanopy Statistics:")
    print(f"  Number of groves: {stats['num_groves']}")
    print(f"  Grove names: {', '.join(stats['grove_names'])}")
    print(f"  Embedding dimension: {stats['embedding_dim']}")

    # Simulate input
    batch_size = 4
    features = torch.randn(batch_size, 512)

    # Route with modality hint
    print("\nRouting with modality hint='image':")
    output, routing_info = canopy(features, modality_hint="image", top_k_groves=2)
    print(f"  Output shape: {output.shape}")
    print(f"  Groves used: {routing_info['groves_used']}")
    print(f"  Grove weights: {routing_info['grove_weights']}")

    # Route without hint (auto-detect)
    print("\nRouting with automatic modality detection:")
    output, routing_info = canopy(features, modality_hint=None, top_k_groves=2)
    print(f"  Output shape: {output.shape}")
    print(f"  Detected modalities: {routing_info['modalities']}")
    print(f"  Groves used: {routing_info['groves_used']}")

    # Get routing summary
    summary = canopy.route_summary(features)
    print("\nRouting Summary:")
    print(f"  Detected modality: {summary['detected_modality']}")
    print(f"  Modality confidence: {summary['modality_confidence']:.3f}")
    print("  Grove routing probabilities:")
    for grove_name, prob in summary["grove_routing_probs"].items():
        print(f"    {grove_name}: {prob:.3f}")

    print("✅ Forest canopy demo successful!")

    return canopy


def demo_end_to_end_canopy():
    """Demo end-to-end routing through the canopy."""
    print("\n=== End-to-End Canopy Routing Demo ===")

    # Create complete system
    groves = {
        "image": VisualGrove(input_dim=512, hidden_dim=64, max_trees=6),
        "audio": AudioGrove(input_dim=512, hidden_dim=64, max_trees=6),
        "text": TextGrove(input_dim=512, hidden_dim=64, max_trees=6),
    }

    canopy = ForestCanopy(grove_dict=groves, embedding_dim=512)

    # Simulate different inputs
    batch_size = 2

    test_cases = [
        ("Image features", torch.randn(batch_size, 512) * 1.0, None),
        ("Audio features", torch.randn(batch_size, 512) * 0.5, None),
        ("Text features", torch.randn(batch_size, 512) * 2.0, None),
        ("Multi-modal features", torch.randn(batch_size, 512) * 1.5, None),
    ]

    print("Testing different input types:")
    for name, features, hint in test_cases:
        output, info = canopy(features, modality_hint=hint, top_k_groves=2)
        print(f"\n  {name}:")
        print(f"    Output shape: {output.shape}")
        print(f"    Detected: {info['modalities']}")
        print(f"    Routed to: {info['groves_used']}")

    print("\n✅ End-to-end canopy routing demo successful!")


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 3 Demo: The Canopy (Advanced Routing & Attention)")
    print("=" * 70)

    # Run demos
    demo_modality_detector()
    demo_grove_router()
    demo_load_balancer()
    demo_cross_grove_attention()
    demo_forest_canopy()
    demo_end_to_end_canopy()

    print("\n" + "=" * 70)
    print("All Phase 3 demos completed successfully!")
    print("=" * 70)
