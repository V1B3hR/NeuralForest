"""
Evolution module for NeuralForest Phase 6: Self-Evolution & Meta-Learning

This module implements neural architecture search and self-improvement systems
that enable the forest to discover optimal tree architectures and continuously
improve its structure and performance.
"""

from .architecture_search import TreeArchitectureSearch
from .self_improvement import SelfImprovementLoop

__all__ = [
    'TreeArchitectureSearch',
    'SelfImprovementLoop',
]
