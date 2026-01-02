"""Soil package for modality-specific preprocessing."""
from .base import SoilProcessor
from .image_processor import ImageSoil
from .audio_processor import AudioSoil
from .text_processor import TextSoil
from .video_processor import VideoSoil

__all__ = ['SoilProcessor', 'ImageSoil', 'AudioSoil', 'TextSoil', 'VideoSoil']
