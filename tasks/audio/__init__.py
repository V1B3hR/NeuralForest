"""Audio tasks module."""

from .transcription import SpeechRecognition
from .classification import AudioClassification

__all__ = ['SpeechRecognition', 'AudioClassification']
