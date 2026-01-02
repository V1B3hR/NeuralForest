"""Cross-modal tasks module."""

from .image_text import ImageTextMatching, ImageCaptioning
from .audio_visual import AudioVisualCorrespondence

__all__ = ["ImageTextMatching", "ImageCaptioning", "AudioVisualCorrespondence"]
