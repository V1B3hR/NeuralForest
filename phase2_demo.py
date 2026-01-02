"""
Demo: Phase 2 - Specialized Groves (Expert Tree Clusters)
Shows how groves work with specialized trees and knowledge transfer.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import grove implementations
from groves import VisualGrove, AudioGrove, TextGrove, VideoGrove
from groves.base_grove import SpecialistTree

# Import mycelium for knowledge transfer
from mycelium import MyceliumNetwork, KnowledgeTransfer


def demo_visual_grove():
    """Demo visual grove with specialist trees."""
    print("\n=== Visual Grove Demo ===")

    # Create visual grove
    visual_grove = VisualGrove(input_dim=512, hidden_dim=64, max_trees=12)

    print(f"Visual Grove created with {visual_grove.num_trees()} initial trees")

    # Simulate image features
    batch_size = 4
    image_features = torch.randn(batch_size, 512)

    # Route through grove
    output, weights = visual_grove(image_features, top_k=3)

    print(f"Input features shape: {image_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Routing weights shape: {weights.shape}")
    print(f"Top 3 trees activated with weights: {weights[0].topk(3)}")

    # Get grove statistics
    stats = visual_grove.get_grove_stats()
    print("\nGrove Statistics:")
    print(f"  Modality: {stats['modality']}")
    print(f"  Number of trees: {stats['num_trees']}")
    print(f"  Memory size: {stats['memory_size']}")

    print("  Tree details:")
    for tree_info in stats["trees"]:
        print(
            f"    Tree {tree_info['id']}: {tree_info['specialization']} "
            f"(age={tree_info['age']}, fitness={tree_info['fitness']:.2f})"
        )

    print("✅ Visual Grove demo successful!")

    return visual_grove


def demo_multi_grove_system():
    """Demo multiple groves working together."""
    print("\n=== Multi-Grove System Demo ===")

    # Create groves for different modalities
    visual_grove = VisualGrove(input_dim=512, hidden_dim=64, max_trees=8)
    audio_grove = AudioGrove(input_dim=512, hidden_dim=64, max_trees=8)
    text_grove = TextGrove(input_dim=512, hidden_dim=64, max_trees=8)
    video_grove = VideoGrove(input_dim=512, hidden_dim=64, max_trees=8)

    groves = {
        "image": visual_grove,
        "audio": audio_grove,
        "text": text_grove,
        "video": video_grove,
    }

    print(f"Created {len(groves)} specialized groves:")
    for name, grove in groves.items():
        print(f"  {name}: {grove.num_trees()} trees")

    # Simulate different modality inputs
    batch_size = 2

    print("\nTesting each grove:")
    for modality, grove in groves.items():
        features = torch.randn(batch_size, 512)
        output, weights = grove(features, top_k=2)
        print(
            f"  {modality} grove: output shape {output.shape}, "
            f"used {(weights[0] > 0).sum().item()} trees"
        )

    print("✅ Multi-grove system demo successful!")

    return groves


def demo_specialist_planting():
    """Demo planting specialist trees dynamically."""
    print("\n=== Specialist Tree Planting Demo ===")

    # Create a grove
    grove = AudioGrove(input_dim=512, hidden_dim=64, max_trees=12)

    initial_trees = grove.num_trees()
    print(f"Initial number of trees: {initial_trees}")

    # Plant new specialists
    new_specializations = [
        "speaker_recognition",
        "music_classification",
        "sound_event_detection",
    ]

    print("\nPlanting new specialist trees:")
    for spec in new_specializations:
        tree_id = grove.plant_specialist(spec)
        if tree_id is not None:
            print(f"  ✓ Planted {spec} tree (ID: {tree_id})")
        else:
            print(f"  ✗ Grove full, couldn't plant {spec} tree")

    final_trees = grove.num_trees()
    print(f"\nFinal number of trees: {final_trees}")
    print(f"Added {final_trees - initial_trees} new specialist trees")

    # Show all tree specializations
    print("\nAll tree specializations:")
    stats = grove.get_grove_stats()
    for tree_info in stats["trees"]:
        print(f"  Tree {tree_info['id']}: {tree_info['specialization']}")

    print("✅ Specialist planting demo successful!")

    return grove


def demo_mycelium_network():
    """Demo mycelium network for knowledge transfer."""
    print("\n=== Mycelium Network Demo ===")

    # Create a grove with multiple trees
    grove = VisualGrove(input_dim=512, hidden_dim=64, max_trees=8)

    # Plant additional trees
    grove.plant_specialist("object_detection")
    grove.plant_specialist("segmentation")

    # Create mycelium network
    mycelium = MyceliumNetwork(num_groves=4)

    # Connect trees
    trees = list(grove.trees)
    if len(trees) >= 2:
        # Connect trees with similar specializations
        for i in range(len(trees) - 1):
            mycelium.connect(trees[i].id, trees[i + 1].id, strength=0.8)

        print(f"Created mycelium connections between {len(trees)} trees")

        # Simulate knowledge transfer
        batch_size = 4
        x = torch.randn(batch_size, 512)

        # Transfer from first tree (teacher) to second tree (student)
        teacher = trees[0]
        student = trees[1]

        # Artificially age the teacher
        for _ in range(100):
            teacher.step_age()

        transfer_loss = mycelium.transfer_knowledge(teacher, student, x)

        print("\nKnowledge Transfer:")
        print(f"  Teacher tree ID: {teacher.id} (age: {teacher.age})")
        print(f"  Student tree ID: {student.id} (age: {student.age})")
        print(f"  Transfer loss: {transfer_loss.item():.4f}")

        # Get network statistics
        stats = mycelium.get_network_stats()
        print("\nMycelium Network Statistics:")
        print(f"  Number of trees: {stats['num_trees']}")
        print(f"  Total connections: {stats['total_connections']}")
        print(f"  Avg connections per tree: {stats['avg_connections_per_tree']:.2f}")

    print("✅ Mycelium network demo successful!")

    return mycelium


def demo_knowledge_transfer_methods():
    """Demo different knowledge transfer methods."""
    print("\n=== Knowledge Transfer Methods Demo ===")

    # Create two trees
    teacher = SpecialistTree(
        input_dim=512,
        hidden_dim=64,
        tree_id=0,
        specialization="classification",
        modality="image",
    )

    student = SpecialistTree(
        input_dim=512,
        hidden_dim=64,
        tree_id=1,
        specialization="classification",
        modality="image",
    )

    # Age teacher
    for _ in range(100):
        teacher.step_age()

    # Simulate input
    batch_size = 4
    x = torch.randn(batch_size, 512)

    # Test distillation loss
    teacher_output = teacher(x)
    student_output = student(x)

    distill_loss = KnowledgeTransfer.distillation_loss(
        teacher_output, student_output, temperature=2.0
    )

    print(f"Distillation Loss: {distill_loss.item():.4f}")

    # Test feature alignment
    teacher_features = teacher.get_features(x)
    student_features = student.get_features(x)

    alignment_loss = KnowledgeTransfer.feature_alignment_loss(
        teacher_features, student_features, margin=0.5
    )

    print(f"Feature Alignment Loss: {alignment_loss.item():.4f}")

    # Test progressive transfer
    teachers = [teacher]
    transfer_loss = KnowledgeTransfer.progressive_knowledge_transfer(
        teachers, student, x
    )

    print(f"Progressive Transfer Loss: {transfer_loss.item():.4f}")

    print("✅ Knowledge transfer methods demo successful!")


if __name__ == "__main__":
    print("=" * 70)
    print("Phase 2 Demo: Specialized Groves & Knowledge Transfer")
    print("=" * 70)

    # Run demos
    demo_visual_grove()
    demo_multi_grove_system()
    demo_specialist_planting()
    demo_mycelium_network()
    demo_knowledge_transfer_methods()

    print("\n" + "=" * 70)
    print("All Phase 2 demos completed successfully!")
    print("=" * 70)
