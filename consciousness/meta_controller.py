"""
Meta-Controller and Forest Consciousness for NeuralForest

Implements high-level self-awareness and autonomous improvement capabilities.
The meta-controller monitors forest state, identifies improvement opportunities,
and executes strategic actions for continuous evolution.
"""

import time
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict
from dataclasses import dataclass

from .goal_manager import GoalManager


@dataclass
class Action:
    """Base class for actions the consciousness can take."""

    type: str
    priority: float
    details: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def execute(self, forest) -> Dict[str, Any]:
        """Execute action on forest. Override in subclasses."""
        raise NotImplementedError


class PlantSpecialistAction(Action):
    """Action to plant a new specialist tree."""

    def __init__(
        self, modality: str = None, specialization: str = None, priority: float = 0.7
    ):
        super().__init__(
            type="plant_specialist",
            priority=priority,
            details={"modality": modality, "specialization": specialization},
        )

    def execute(self, forest) -> Dict[str, Any]:
        """Plant a new tree in the forest."""
        before_count = forest.num_trees()
        forest._plant_tree()
        after_count = forest.num_trees()

        return {
            "success": after_count > before_count,
            "trees_added": after_count - before_count,
            "total_trees": after_count,
        }


class PruneMemoryAction(Action):
    """Action to prune memory when capacity is high."""

    def __init__(self, priority: float = 0.6):
        super().__init__(type="prune_memory", priority=priority, details={})

    def execute(self, forest) -> Dict[str, Any]:
        """Remove low-priority items from memory."""
        before_size = len(forest.mulch)

        # Keep only top 70% priority items
        if before_size > forest.mulch.capacity * 0.7:
            # Sort by priority and keep top items
            sorted_data = sorted(forest.mulch.data, key=lambda x: x[2], reverse=True)
            keep_count = int(len(sorted_data) * 0.7)
            forest.mulch.data = deque(
                sorted_data[:keep_count], maxlen=forest.mulch.capacity
            )

        after_size = len(forest.mulch)

        return {
            "success": True,
            "items_removed": before_size - after_size,
            "memory_size": after_size,
            "capacity": forest.mulch.capacity,
        }


class IncreaseReplayAction(Action):
    """Action to increase replay ratio when performance drops."""

    def __init__(self, priority: float = 0.5):
        super().__init__(type="increase_replay", priority=priority, details={})

    def execute(self, forest) -> Dict[str, Any]:
        """Signal to increase replay ratio (handled by training loop)."""
        return {
            "success": True,
            "recommendation": "increase_replay_ratio",
        }


class SnapshotTeacherAction(Action):
    """Action to snapshot current model as teacher for distillation."""

    def __init__(self, priority: float = 0.8):
        super().__init__(type="snapshot_teacher", priority=priority, details={})

    def execute(self, forest) -> Dict[str, Any]:
        """Create teacher snapshot for distillation."""
        forest.snapshot_teacher()
        return {
            "success": True,
            "teacher_snapshot_created": True,
        }


class ConsciousnessMemory:
    """
    Stores reflections, actions, and their outcomes for learning.

    The consciousness memory allows the forest to learn from past decisions
    and improve its meta-learning strategies over time.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.reflections = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.outcomes = deque(maxlen=capacity)

    def store_reflection(self, reflection: Dict[str, Any]):
        """Store a self-reflection about forest state."""
        self.reflections.append(
            {
                "timestamp": time.time(),
                "data": reflection,
            }
        )

    def store_action(self, action: Action, result: Dict[str, Any]):
        """Store an action and its outcome."""
        record = {
            "timestamp": time.time(),
            "action_type": action.type,
            "action_details": action.details,
            "priority": action.priority,
            "result": result,
        }
        self.actions.append(record)
        self.outcomes.append(result.get("success", False))

    def get_recent_reflections(self, n: int = 10) -> List[Dict]:
        """Retrieve the N most recent reflections."""
        return list(self.reflections)[-n:]

    def get_action_history(
        self, action_type: Optional[str] = None, n: int = 100
    ) -> List[Dict]:
        """
        Retrieve action history, optionally filtered by type.

        Args:
            action_type: If provided, filter to only this action type
            n: Maximum number of actions to return
        """
        actions = list(self.actions)[-n:]
        if action_type:
            actions = [a for a in actions if a["action_type"] == action_type]
        return actions

    def get_success_rate(self, action_type: Optional[str] = None) -> float:
        """Calculate success rate for actions."""
        if action_type:
            actions = self.get_action_history(action_type)
            if not actions:
                return 0.0
            successes = sum(1 for a in actions if a["result"].get("success", False))
            return successes / len(actions)
        else:
            if not self.outcomes:
                return 0.0
            return sum(self.outcomes) / len(self.outcomes)

    def __len__(self):
        return len(self.reflections)


class StrategyLibrary:
    """
    Library of strategies for prioritizing and executing actions.
    Learns which strategies work best over time.
    """

    def __init__(self):
        self.strategy_scores = defaultdict(lambda: 1.0)

    def prioritize(self, actions: List[Action]) -> List[Action]:
        """
        Sort actions by priority, considering learned strategy effectiveness.

        Args:
            actions: List of actions to prioritize

        Returns:
            Sorted list of actions (highest priority first)
        """
        # Adjust priorities based on historical performance
        for action in actions:
            strategy_score = self.strategy_scores.get(action.type, 1.0)
            action.priority *= strategy_score

        # Sort by adjusted priority
        return sorted(actions, key=lambda a: a.priority, reverse=True)

    def update_strategy_score(self, action_type: str, success: bool):
        """Update strategy score based on action outcome."""
        current = self.strategy_scores[action_type]
        # Exponential moving average
        alpha = 0.1
        new_score = (1 - alpha) * current + alpha * (1.0 if success else 0.5)
        self.strategy_scores[action_type] = new_score


class ForestConsciousness:
    """
    High-level meta-controller that monitors and improves the entire forest.
    Implements self-awareness and autonomous improvement.

    The consciousness system:
    - Reflects on current state and performance
    - Identifies improvement opportunities
    - Plans and executes strategic actions
    - Learns from outcomes to improve strategies
    """

    def __init__(self, forest):
        self.forest = forest
        self.memory = ConsciousnessMemory()
        self.goals = GoalManager()
        self.strategies = StrategyLibrary()

        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.evolution_step = 0

    def reflect(self) -> Dict[str, Any]:
        """
        Self-reflection: analyze current state and performance.

        Returns:
            Dictionary with comprehensive forest analysis
        """
        reflection = {
            "timestamp": time.time(),
            "evolution_step": self.evolution_step,
            "forest_size": self.forest.num_trees(),
            "overall_fitness": self._compute_forest_fitness(),
            "memory_usage": {
                "mulch_size": len(self.forest.mulch),
                "mulch_capacity": self.forest.mulch.capacity,
                "mulch_utilization": len(self.forest.mulch)
                / self.forest.mulch.capacity,
                "anchor_size": len(self.forest.anchors),
                "anchor_capacity": self.forest.anchors.capacity,
            },
            "tree_health": self._analyze_tree_health(),
            "knowledge_gaps": self._identify_knowledge_gaps(),
            "resource_utilization": self._analyze_resources(),
            "recent_performance": self._analyze_recent_performance(),
        }

        self.memory.store_reflection(reflection)
        return reflection

    def plan(self, reflection: Dict[str, Any]) -> List[Action]:
        """
        Strategic planning based on reflection.

        Args:
            reflection: Current state analysis from reflect()

        Returns:
            Prioritized list of actions to take
        """
        actions = []

        # Gap filling - plant trees if forest is small
        for gap in reflection["knowledge_gaps"]:
            actions.append(
                PlantSpecialistAction(
                    modality=gap.get("modality"),
                    specialization=gap.get("task"),
                    priority=0.7 * gap.get("severity", 1.0),
                )
            )

        # Resource optimization - prune memory if too full
        mem_util = reflection["resource_utilization"].get("memory", 0)
        if mem_util > 0.9:
            actions.append(PruneMemoryAction(priority=0.6))

        # Performance improvement - increase replay if performance declining
        perf_trend = reflection["recent_performance"].get("trend", 0)
        if perf_trend < 0:
            actions.append(IncreaseReplayAction(priority=0.5))
            actions.append(SnapshotTeacherAction(priority=0.8))

        # Use strategy library to prioritize
        return self.strategies.prioritize(actions)

    def evolve(self) -> Dict[str, Any]:
        """
        Main evolution loop: reflect, plan, act, learn.

        Returns:
            Dictionary with evolution results
        """
        # Reflect on current state
        reflection = self.reflect()

        # Plan actions
        actions = self.plan(reflection)

        # Execute actions
        results = []
        for action in actions[:3]:  # Limit to top 3 actions per evolution step
            result = action.execute(self.forest)
            self.memory.store_action(action, result)
            results.append(
                {
                    "action_type": action.type,
                    "result": result,
                }
            )

            # Update strategy scores
            self.strategies.update_strategy_score(
                action.type, result.get("success", False)
            )

        # Update goals with current metrics
        metrics = {
            "forest_fitness": reflection["overall_fitness"],
            "num_trees": reflection["forest_size"],
            "memory_utilization": reflection["memory_usage"]["mulch_utilization"],
        }
        self.goals.update_progress(metrics)

        # Learn from outcomes
        self._update_strategies()

        self.evolution_step += 1

        return {
            "evolution_step": self.evolution_step,
            "reflection": reflection,
            "actions_taken": len(results),
            "results": results,
            "active_goals": len(self.goals.get_active_goals()),
        }

    def _compute_forest_fitness(self) -> float:
        """Compute average fitness across all trees."""
        if not self.forest.trees:
            return 0.0
        total_fitness = sum(t.fitness for t in self.forest.trees)
        return total_fitness / len(self.forest.trees)

    def _analyze_tree_health(self) -> Dict[str, Any]:
        """Analyze health of individual trees."""
        if not self.forest.trees:
            return {"status": "no_trees"}

        fitnesses = [t.fitness for t in self.forest.trees]
        ages = [t.age for t in self.forest.trees]
        barks = [t.bark for t in self.forest.trees]

        return {
            "average_fitness": sum(fitnesses) / len(fitnesses),
            "min_fitness": min(fitnesses),
            "max_fitness": max(fitnesses),
            "average_age": sum(ages) / len(ages),
            "average_bark": sum(barks) / len(barks),
            "weak_trees": sum(1 for f in fitnesses if f < 2.0),
            "strong_trees": sum(1 for f in fitnesses if f > 8.0),
        }

    def _identify_knowledge_gaps(self) -> List[Dict]:
        """Identify areas where forest needs improvement."""
        gaps = []

        # Check if forest is too small
        if self.forest.num_trees() < 3:
            gaps.append(
                {
                    "modality": "general",
                    "task": "general",
                    "severity": 1.0,
                    "reason": "insufficient_trees",
                }
            )

        # Check if average fitness is low
        avg_fitness = self._compute_forest_fitness()
        if avg_fitness < 5.0:
            gaps.append(
                {
                    "modality": "general",
                    "task": "general",
                    "severity": 0.7,
                    "reason": "low_fitness",
                }
            )

        return gaps

    def _analyze_resources(self) -> Dict[str, float]:
        """Analyze resource utilization."""
        return {
            "memory": len(self.forest.mulch) / self.forest.mulch.capacity,
            "trees": self.forest.num_trees() / self.forest.max_trees,
        }

    def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent performance trends."""
        if len(self.performance_history) < 2:
            return {"trend": 0, "data_points": 0}

        recent = list(self.performance_history)[-10:]

        # Simple trend: compare first half to second half
        mid = len(recent) // 2
        first_half_avg = sum(recent[:mid]) / max(1, mid)
        second_half_avg = sum(recent[mid:]) / max(1, len(recent) - mid)

        trend = second_half_avg - first_half_avg

        return {
            "trend": trend,
            "recent_average": sum(recent) / len(recent),
            "data_points": len(recent),
        }

    def _update_strategies(self):
        """Update strategy effectiveness based on recent outcomes."""
        # Calculate success rates for each action type
        for action_type in [
            "plant_specialist",
            "prune_memory",
            "increase_replay",
            "snapshot_teacher",
        ]:
            success_rate = self.memory.get_success_rate(action_type)
            if success_rate > 0:
                # Boost scores for successful strategies
                self.strategies.strategy_scores[action_type] *= 0.9 + 0.2 * success_rate

    def add_performance_sample(self, value: float):
        """Add a performance sample for trend analysis."""
        self.performance_history.append(value)

    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report."""
        reflection = self.reflect()
        goals_summary = self.goals.get_progress_summary()

        return {
            "consciousness": {
                "evolution_step": self.evolution_step,
                "memory_size": len(self.memory),
                "action_success_rate": self.memory.get_success_rate(),
            },
            "forest": reflection,
            "goals": goals_summary,
            "strategies": dict(self.strategies.strategy_scores),
        }
