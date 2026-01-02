"""
Seasons module for NeuralForest - Training regime management.

This module implements seasonal training cycles inspired by natural forest seasons:
- Spring: Growth and exploration
- Summer: Maximum productivity
- Autumn: Pruning and fitness evaluation
- Winter: Consolidation and knowledge distillation
"""

from .cycle_controller import SeasonalCycle
from .spring_growth import SpringGrowth
from .summer_productivity import SummerProductivity
from .autumn_pruning import AutumnPruning
from .winter_consolidation import WinterConsolidation

__all__ = [
    "SeasonalCycle",
    "SpringGrowth",
    "SummerProductivity",
    "AutumnPruning",
    "WinterConsolidation",
]
