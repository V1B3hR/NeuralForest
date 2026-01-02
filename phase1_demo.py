"""
Demo: Using Phase 1 Soil Processors and Root Network
Shows how to process different modalities and combine them.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import soil processors
from soil import ImageSoil, AudioSoil
from soil.text_processor import SimpleTextSoil
from soil.video_processor import Frame2DVideoSoil

# Import root network
from roots import RootNetwork


def demo_image_processing():
    """Demo image processing with ImageSoil."""
    print("\n=== Image Processing Demo ===")

    # Create image soil processor
    image_soil = ImageSoil(input_channels=3, output_dim=512, image_size=224)

    # Simulate batch of images [B, C, H, W]
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)

    # Process images
    embeddings = image_soil(images)
    print(f"Input shape: {images.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print("✅ Image processing successful!")

    return embeddings


def demo_audio_processing():
    """Demo audio processing with AudioSoil."""
    print("\n=== Audio Processing Demo ===")

    # Create audio soil processor
    audio_soil = AudioSoil(input_channels=1, output_dim=512)

    # Simulate batch of audio [B, C, T]
    batch_size = 4
    audio = torch.randn(batch_size, 1, 16000)  # 1 second at 16kHz

    # Process audio
    embeddings = audio_soil(audio)
    print(f"Input shape: {audio.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print("✅ Audio processing successful!")

    return embeddings


def demo_text_processing():
    """Demo text processing with SimpleTextSoil."""
    print("\n=== Text Processing Demo ===")

    # Create text soil processor
    text_soil = SimpleTextSoil(vocab_size=30000, output_dim=512)

    # Simulate batch of token IDs [B, seq_len]
    batch_size = 4
    seq_length = 128
    text_tokens = torch.randint(1, 30000, (batch_size, seq_length))

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.ones_like(text_tokens)
    # Simulate some padding in last positions
    attention_mask[:, -10:] = 0

    # Process text
    embeddings = text_soil(text_tokens, attention_mask=attention_mask)
    print(f"Input shape: {text_tokens.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print("✅ Text processing successful!")

    return embeddings


def demo_video_processing():
    """Demo video processing with Frame2DVideoSoil."""
    print("\n=== Video Processing Demo ===")

    # Create video soil processor
    video_soil = Frame2DVideoSoil(input_channels=3, output_dim=512, num_frames=16)

    # Simulate batch of videos [B, C, T, H, W]
    batch_size = 2  # Smaller batch for memory
    video = torch.randn(batch_size, 3, 16, 112, 112)  # 16 frames, 112x112 resolution

    # Process video
    embeddings = video_soil(video)
    print(f"Input shape: {video.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")
    print("✅ Video processing successful!")

    return embeddings


def demo_multi_modal_fusion():
    """Demo multi-modal fusion with RootNetwork."""
    print("\n=== Multi-Modal Fusion Demo ===")

    # Create root network
    root_network = RootNetwork(embedding_dim=512)

    # Simulate embeddings from different modalities
    batch_size = 4
    image_embeddings = torch.randn(batch_size, 512)
    audio_embeddings = torch.randn(batch_size, 512)
    text_embeddings = torch.randn(batch_size, 512)

    # Test single modality
    print("\nSingle modality (image only):")
    unified_single = root_network({"image": image_embeddings})
    print(f"  Output shape: {unified_single.shape}")

    # Test dual modality
    print("\nDual modality (image + audio):")
    unified_dual = root_network({"image": image_embeddings, "audio": audio_embeddings})
    print(f"  Output shape: {unified_dual.shape}")

    # Test triple modality
    print("\nTriple modality (image + audio + text):")
    unified_triple = root_network(
        {"image": image_embeddings, "audio": audio_embeddings, "text": text_embeddings}
    )
    print(f"  Output shape: {unified_triple.shape}")

    # Get attention weights for analysis
    attention = root_network.get_attention_weights(
        {"image": image_embeddings, "audio": audio_embeddings, "text": text_embeddings}
    )
    if attention is not None:
        print(f"\nCross-modal attention weights shape: {attention.shape}")

    print("✅ Multi-modal fusion successful!")


def demo_end_to_end():
    """Demo end-to-end processing: raw input -> soil -> roots -> unified."""
    print("\n=== End-to-End Multi-Modal Pipeline Demo ===")

    # Create processors
    image_soil = ImageSoil(input_channels=3, output_dim=512, image_size=224)
    text_soil = SimpleTextSoil(vocab_size=30000, output_dim=512)
    root_network = RootNetwork(embedding_dim=512)

    # Simulate raw inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    text_tokens = torch.randint(1, 30000, (batch_size, 128))

    print("\nRaw inputs:")
    print(f"  Images: {images.shape}")
    print(f"  Text tokens: {text_tokens.shape}")

    # Process through soil
    print("\nProcessing through soil processors...")
    image_embeds = image_soil(images)
    text_embeds = text_soil(text_tokens)

    print(f"  Image embeddings: {image_embeds.shape}")
    print(f"  Text embeddings: {text_embeds.shape}")

    # Fuse through root network
    print("\nFusing through root network...")
    unified = root_network({"image": image_embeds, "text": text_embeds})

    print(f"  Unified embeddings: {unified.shape}")
    print("\n✅ End-to-end pipeline successful!")
    print(f"   Raw multi-modal data → Unified {unified.shape[1]}-dim representation")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 Demo: Soil Processors & Root Network")
    print("=" * 60)

    # Run individual modality demos
    demo_image_processing()
    demo_audio_processing()
    demo_text_processing()
    demo_video_processing()

    # Run fusion demo
    demo_multi_modal_fusion()

    # Run end-to-end demo
    demo_end_to_end()

    print("\n" + "=" * 60)
    print("All Phase 1 demos completed successfully!")
    print("=" * 60)
