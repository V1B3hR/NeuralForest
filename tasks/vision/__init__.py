"""Vision tasks module - Init file."""

from .classification import ImageClassification
from .detection import ObjectDetection
from .segmentation import SemanticSegmentation

__all__ = ["ImageClassification", "ObjectDetection", "SemanticSegmentation"]
