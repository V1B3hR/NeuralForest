#!/usr/bin/env python3
"""
Phase 5 Demo: Multi-Modal Understanding

Demonstrates the multi-modal task system with:
- Vision tasks (classification, detection, segmentation)
- Audio tasks (speech recognition, classification)
- Text tasks (classification, NER, generation)
- Video tasks (classification, action recognition)
- Cross-modal tasks (image-text matching, audio-visual)
"""

import torch
from tasks import TaskRegistry, TaskConfig, vision, audio, text, video, cross_modal


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
    print()


if __name__ == "__main__":
    main()
