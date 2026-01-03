"""
Evolution module for NeuralForest Phase 3 & Phase 6: Evolution & Self-Improvement

This module implements neural architecture search, self-improvement systems,
and tree legacy management that enable the forest to discover optimal tree
architectures, continuously improve its structure and performance, and
maintain a memory of eliminated trees for evolutionary insights.
"""

from .architecture_search import TreeArchitectureSearch
from .self_improvement import SelfImprovementLoop
from .tree_graveyard import TreeGraveyard, TreeRecord, GraveyardStats

__all__ = [
    "TreeArchitectureSearch",
    "SelfImprovementLoop",
    "TreeGraveyard",
    "TreeRecord",
    "GraveyardStats",
]
