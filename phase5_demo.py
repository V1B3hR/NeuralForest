#!/usr/bin/env python3
"""
Phase 5 Demo: Multi-Modal Understanding & Visualization

Demonstrates the multi-modal task system with:
- Vision tasks (classification, detection, segmentation)
- Audio tasks (speech recognition, classification)
- Text tasks (classification, NER, generation)
- Video tasks (classification, action recognition)
- Cross-modal tasks (image-text matching, audio-visual)
- Comprehensive evolution visualization tools
"""

import torch
import numpy as np
from tasks import TaskRegistry, TaskConfig, vision, audio, text, video, cross_modal
from evolution import ForestVisualizer


def demo_task_registry():
    """Demonstrate the task registry system."""
    print("=" * 60)
    print("Phase 5 Demo: Multi-Modal Understanding")
    print("=" * 60)
    print()

    print("1. Task Registry")
    print("-" * 60)
    print("Available tasks:")
    for task_name in sorted(TaskRegistry.list_tasks()):
        print(f"  - {task_name}")
    print()


def demo_vision_tasks():
    """Demonstrate vision tasks."""
    print("2. Vision Tasks")
    print("-" * 60)

    # Image Classification
    print("a) Image Classification")
    img_classifier = vision.ImageClassification(input_dim=512, num_classes=1000)
    print(f"  Task: {img_classifier.task_name}")
    print(f"  Supported datasets: {img_classifier.SUPPORTED_DATASETS[:3]}")

    # Test forward pass
    features = torch.randn(4, 512)
    logits = img_classifier(features)
    print(f"  Input shape: {features.shape}")
    print(f"  Output shape: {logits.shape}")
    print("  ✓ Classification head working")

    # Object Detection
    print("\nb) Object Detection")
    detector = vision.ObjectDetection(input_dim=512, num_classes=80)
    print(f"  Task: {detector.task_name}")
    detection_out = detector(features)
    print(f"  Outputs: {list(detection_out.keys())}")
    print(f"  BBox shape: {detection_out['bbox'].shape}")
    print(f"  Class logits shape: {detection_out['class_logits'].shape}")
    print("  ✓ Detection head working")

    # Segmentation
    print("\nc) Semantic Segmentation")
    segmenter = vision.SemanticSegmentation(
        input_dim=512, num_classes=21, spatial_size=32
    )
    print(f"  Task: {segmenter.task_name}")
    seg_out = segmenter(features)
    print(f"  Output shape: {seg_out.shape}")
    print("  ✓ Segmentation head working")
    print()


def demo_audio_tasks():
    """Demonstrate audio tasks."""
    print("3. Audio Tasks")
    print("-" * 60)

    # Speech Recognition
    print("a) Speech Recognition")
    speech_rec = audio.SpeechRecognition(input_dim=512, vocab_size=100, max_length=50)
    print(f"  Task: {speech_rec.task_name}")
    print(f"  Supported datasets: {speech_rec.SUPPORTED_DATASETS}")

    features = torch.randn(4, 512)
    transcription = speech_rec(features)
    print(f"  Output shape: {transcription.shape}")
    print("  ✓ Speech recognition head working")

    # Audio Classification
    print("\nb) Audio Classification")
    audio_clf = audio.AudioClassification(input_dim=512, num_classes=10)
    print(f"  Task: {audio_clf.task_name}")
    print(f"  Supported tasks: {audio_clf.SUPPORTED_TASKS[:2]}")

    audio_logits = audio_clf(features)
    print(f"  Output shape: {audio_logits.shape}")
    print("  ✓ Audio classification head working")
    print()


def demo_text_tasks():
    """Demonstrate text tasks."""
    print("4. Text Tasks")
    print("-" * 60)

    # Text Classification
    print("a) Text Classification / Sentiment Analysis")
    text_clf = text.TextClassification(input_dim=512, num_classes=3)
    print(f"  Task: {text_clf.task_name}")
    print(f"  Supported tasks: {text_clf.SUPPORTED_TASKS[:2]}")

    features = torch.randn(4, 512)
    text_logits = text_clf(features)
    print(f"  Output shape: {text_logits.shape}")
    print("  ✓ Text classification head working")

    # Named Entity Recognition
    print("\nb) Named Entity Recognition (NER)")
    ner = text.NamedEntityRecognition(input_dim=512, num_entity_types=6, max_length=128)
    print(f"  Task: {ner.task_name}")
    print(f"  Entity types: {ner.ENTITY_TYPES}")

    ner_out = ner(features)
    print(f"  Output shape: {ner_out.shape}")
    print("  ✓ NER head working")

    # Text Generation
    print("\nc) Text Generation / Summarization")
    text_gen = text.TextGeneration(input_dim=512, vocab_size=30000, max_length=100)
    print(f"  Task: {text_gen.task_name}")
    print(f"  Supported tasks: {text_gen.SUPPORTED_TASKS[:2]}")

    gen_out = text_gen(features)
    print(f"  Output shape: {gen_out.shape}")
    print("  ✓ Text generation head working")
    print()


def demo_video_tasks():
    """Demonstrate video tasks."""
    print("5. Video Tasks")
    print("-" * 60)

    # Video Classification
    print("a) Video Classification")
    video_clf = video.VideoClassification(input_dim=512, num_classes=400)
    print(f"  Task: {video_clf.task_name}")
    print(f"  Supported datasets: {video_clf.SUPPORTED_DATASETS[:2]}")

    features = torch.randn(4, 512)
    video_logits = video_clf(features)
    print(f"  Output shape: {video_logits.shape}")
    print("  ✓ Video classification head working")

    # Action Recognition
    print("\nb) Action Recognition")
    action_rec = video.ActionRecognition(input_dim=512, num_actions=100)
    print(f"  Task: {action_rec.task_name}")
    print(f"  Supported datasets: {action_rec.SUPPORTED_DATASETS}")

    action_out = action_rec(features)
    print(f"  Output shape: {action_out.shape}")
    print("  ✓ Action recognition head working")
    print()


def demo_cross_modal_tasks():
    """Demonstrate cross-modal tasks."""
    print("6. Cross-Modal Tasks")
    print("-" * 60)

    # Image-Text Matching
    print("a) Image-Text Matching")
    img_text_match = cross_modal.ImageTextMatching(input_dim=512)
    print(f"  Task: {img_text_match.task_name}")
    print(f"  Supported datasets: {img_text_match.SUPPORTED_DATASETS[:2]}")

    img_features = torch.randn(4, 512)
    text_features = torch.randn(4, 512)

    img_proj = img_text_match(img_features)
    text_proj = img_text_match(text_features)
    print(f"  Image projection shape: {img_proj.shape}")
    print(f"  Text projection shape: {text_proj.shape}")

    # Compute matching loss
    loss = img_text_match.get_loss(img_proj, text_proj)
    print(f"  Contrastive loss: {loss.item():.4f}")
    print("  ✓ Image-text matching head working")

    # Image Captioning
    print("\nb) Image Captioning")
    img_caption = cross_modal.ImageCaptioning(
        input_dim=512, vocab_size=10000, max_length=50
    )
    print(f"  Task: {img_caption.task_name}")

    captions = img_caption(img_features)
    print(f"  Caption shape: {captions.shape}")
    print("  ✓ Image captioning head working")

    # Audio-Visual Correspondence
    print("\nc) Audio-Visual Correspondence")
    av_corr = cross_modal.AudioVisualCorrespondence(input_dim=512)
    print(f"  Task: {av_corr.task_name}")
    print(f"  Supported datasets: {av_corr.SUPPORTED_DATASETS}")

    audio_features = torch.randn(4, 512)
    visual_features = torch.randn(4, 512)

    corr_out = av_corr(audio_features, visual_features)
    print(f"  Output shape: {corr_out.shape}")
    print("  ✓ Audio-visual correspondence head working")
    print()


def demo_task_config():
    """Demonstrate task configuration system."""
    print("7. Task Configuration System")
    print("-" * 60)

    # Create a task configuration
    config = TaskConfig(
        task_name="image_classification_imagenet",
        modality="image",
        head_type="image_classification",
        input_dim=512,
        output_params={"num_classes": 1000, "dropout": 0.3},
    )

    print(f"Task Name: {config.task_name}")
    print(f"Modality: {config.modality}")
    print(f"Head Type: {config.head_type}")
    print(f"Input Dim: {config.input_dim}")
    print(f"Output Params: {config.output_params}")

    # Create head from config
    head = config.create_head()
    print(f"\n✓ Created head: {head.task_name}")

    # Test it
    features = torch.randn(2, 512)
    output = head(features)
    print(f"  Output shape: {output.shape}")
    print()


def main():
    """Run all Phase 5 demonstrations."""
    print("\n" + "=" * 60)
    print("NEURALFOREST PHASE 5: MULTI-MODAL UNDERSTANDING DEMO")
    print("=" * 60)
    print()

    # Set seed for reproducibility
    torch.manual_seed(42)

    demo_task_registry()
    demo_vision_tasks()
    demo_audio_tasks()
    demo_text_tasks()
    demo_video_tasks()
    demo_cross_modal_tasks()
    demo_task_config()
    demo_visualization_system()

    print("=" * 60)
    print("Phase 5 Demo Complete! ✅")
    print("=" * 60)
    print()
    print("Summary:")
    print("- 15+ task implementations across 5 modalities")
    print("- Vision: Classification, Detection, Segmentation")
    print("- Audio: Speech Recognition, Classification")
    print("- Text: Classification, NER, Generation")
    print("- Video: Classification, Action Recognition")
    print("- Cross-modal: Image-Text, Audio-Visual")
    print("- Flexible TaskRegistry for dynamic task management")
    print("- Comprehensive visualization tools (fitness, diversity, heatmaps, etc.)")
    print()


def demo_visualization_system():
    """Demonstrate the visualization system."""
    print("7. Evolution Visualization System")
    print("-" * 60)
    
    # Create visualizer
    visualizer = ForestVisualizer(save_dir="./visualizations")
    print("✓ ForestVisualizer initialized")
    
    # Create mock evolution history data
    print("\na) Creating mock evolution data for demonstration...")
    history = []
    for gen in range(20):
        history.append({
            'generation': gen,
            'num_trees': 5 + np.random.randint(-1, 3),
            'avg_fitness': 2.0 + gen * 0.2 + np.random.randn() * 0.3,
            'max_fitness': 3.0 + gen * 0.25 + np.random.randn() * 0.2,
            'min_fitness': 1.0 + gen * 0.15 + np.random.randn() * 0.2,
            'fitness_std': 0.5 + np.random.rand() * 0.3,
            'architecture_diversity': 0.3 + np.random.rand() * 0.4,
            'fitness_diversity': 0.4 + np.random.rand() * 0.3,
            'mutations_count': np.random.randint(1, 5),
            'crossovers_count': np.random.randint(0, 3),
            'births_count': np.random.randint(0, 2),
            'deaths_count': np.random.randint(0, 2),
        })
    
    # Create mock tree data
    trees_data = []
    for i in range(6):
        trees_data.append({
            'id': i,
            'fitness': 3.0 + np.random.rand() * 3.0,
            'age': np.random.randint(1, 15),
            'bark': 1.0 + np.random.rand() * 0.5,
            'architecture': {
                'num_layers': np.random.choice([2, 3, 4, 5]),
                'hidden_dim': np.random.choice([32, 64, 128, 256]),
                'dropout': np.random.choice([0.0, 0.1, 0.2, 0.3]),
                'activation': np.random.choice(['relu', 'gelu', 'tanh']),
                'use_residual': np.random.choice([True, False]),
                'normalization': np.random.choice(['layer', 'batch', 'none']),
            }
        })
    
    # Create mock genealogy data
    genealogy_data = {
        0: {'fitness': 3.5, 'generation': 0, 'age': 19, 'parents': []},
        1: {'fitness': 4.2, 'generation': 5, 'age': 14, 'parents': [0]},
        2: {'fitness': 5.1, 'generation': 8, 'age': 11, 'parents': [0, 1]},
        3: {'fitness': 4.8, 'generation': 10, 'age': 9, 'parents': [1]},
        4: {'fitness': 5.5, 'generation': 15, 'age': 4, 'parents': [2]},
        5: {'fitness': 6.0, 'generation': 18, 'age': 1, 'parents': [2, 3]},
    }
    
    print("✓ Mock evolution data created")
    print(f"  - {len(history)} generations")
    print(f"  - {len(trees_data)} current trees")
    print(f"  - {len(genealogy_data)} genealogy records")
    
    # Test individual plots (without showing)
    print("\nb) Testing individual visualization components...")
    
    try:
        visualizer.plot_fitness_trends(history, show=False)
        print("  ✓ Fitness trends plot")
    except Exception as e:
        print(f"  ✗ Fitness trends plot failed: {e}")
    
    try:
        visualizer.plot_diversity_metrics(history, show=False)
        print("  ✓ Diversity metrics plot")
    except Exception as e:
        print(f"  ✗ Diversity metrics plot failed: {e}")
    
    try:
        visualizer.plot_evolution_events(history, show=False)
        print("  ✓ Evolution events plot")
    except Exception as e:
        print(f"  ✗ Evolution events plot failed: {e}")
    
    try:
        visualizer.plot_architecture_distribution(trees_data, method='pca', show=False)
        print("  ✓ Architecture distribution (PCA) plot")
    except Exception as e:
        print(f"  ✗ Architecture distribution plot: {e}")
    
    try:
        visualizer.plot_performance_heatmap(trees_data, show=False)
        print("  ✓ Performance heatmap")
    except Exception as e:
        print(f"  ✗ Performance heatmap failed: {e}")
    
    try:
        visualizer.plot_species_tree(genealogy_data, show=False)
        print("  ✓ Species/genealogy tree")
    except Exception as e:
        print(f"  ✗ Species tree plot: {e}")
    
    # Test dashboard
    print("\nc) Creating comprehensive evolution dashboard...")
    try:
        visualizer.create_evolution_dashboard(
            history, 
            trees_data, 
            genealogy_data,
            show=False
        )
        print("  ✓ Evolution dashboard created")
    except Exception as e:
        print(f"  ✗ Dashboard creation failed: {e}")
    
    # Export all plots
    print("\nd) Exporting all visualization plots...")
    try:
        saved_files = visualizer.export_all_plots(
            history,
            trees_data,
            genealogy_data,
            prefix="phase5_demo"
        )
        print(f"  ✓ Exported {len(saved_files)} visualizations")
        for plot_type, path in saved_files.items():
            print(f"    - {plot_type}: {path}")
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
    
    print("\n✓ Visualization system demonstration complete")
    print()


if __name__ == "__main__":
    main()
