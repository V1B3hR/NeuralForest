"""
Tasks module for NeuralForest - Multi-modal task implementations.

This module provides task-specific heads for various modalities:
- Vision: Image classification, object detection, segmentation
- Audio: Speech recognition, audio classification
- Text: Text classification, NER, generation
- Video: Video classification, action recognition
- Cross-modal: Image-text matching, audio-visual correspondence
"""

from .base import (
    TaskHead,
    ClassificationHead,
    DetectionHead,
    SegmentationHead,
    RegressionHead,
    GenerationHead,
    TaskRegistry,
    TaskConfig
)

# Import modality-specific tasks
from . import vision
from . import audio
from . import text
from . import video
from . import cross_modal

__all__ = [
    # Base classes
    'TaskHead',
    'ClassificationHead',
    'DetectionHead',
    'SegmentationHead',
    'RegressionHead',
    'GenerationHead',
    'TaskRegistry',
    'TaskConfig',
    # Modality modules
    'vision',
    'audio',
    'text',
    'video',
    'cross_modal'
]
