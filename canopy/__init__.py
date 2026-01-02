"""
Canopy module: Advanced routing and attention mechanisms.
The canopy directs inputs to the most appropriate groves and trees.
"""

from .hierarchical_router import ForestCanopy, GroveRouter
from .modality_detector import ModalityDetector
from .load_balancer import CanopyBalancer
from .attention_aggregator import CrossGroveAttention

__all__ = [
    "ForestCanopy",
    "GroveRouter",
    "ModalityDetector",
    "CanopyBalancer",
    "CrossGroveAttention",
]
