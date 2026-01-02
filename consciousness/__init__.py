"""
Consciousness module for NeuralForest Phase 6: Self-Evolution & Meta-Learning

This module implements the meta-controller and goal management systems that
enable the forest to monitor itself, set goals, and autonomously improve.
"""

from .meta_controller import ForestConsciousness, ConsciousnessMemory
from .goal_manager import GoalManager, LearningGoal

__all__ = [
    "ForestConsciousness",
    "ConsciousnessMemory",
    "GoalManager",
    "LearningGoal",
]
