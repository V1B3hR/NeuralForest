"""Text tasks module."""

from .classification import TextClassification
from .ner import NamedEntityRecognition
from .generation import TextGeneration

__all__ = ['TextClassification', 'NamedEntityRecognition', 'TextGeneration']
