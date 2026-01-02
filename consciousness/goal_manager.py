"""
Goal Management System for NeuralForest

Manages learning objectives, priorities, and progress tracking for autonomous improvement.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import time


@dataclass
class LearningGoal:
    """
    Represents a specific learning objective for the forest.
    
    A goal has a target metric and value that defines success.
    Progress is tracked and goals can be marked complete when achieved.
    """
    name: str
    target_metric: str
    target_value: float
    priority: int = 1
    current_value: float = 0.0
    created_at: float = None
    completed_at: Optional[float] = None
    description: str = ""
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
    
    def is_complete(self) -> bool:
        """Check if goal has been achieved."""
        return self.current_value >= self.target_value
    
    def update(self, metrics: Dict[str, Any]):
        """Update progress based on current metrics."""
        if self.target_metric in metrics:
            self.current_value = float(metrics[self.target_metric])
            if self.is_complete() and self.completed_at is None:
                self.completed_at = time.time()
    
    def progress(self) -> float:
        """Return progress as ratio from 0.0 to 1.0."""
        if self.target_value == 0:
            return 1.0 if self.current_value == 0 else 0.0
        return min(1.0, self.current_value / self.target_value)
    
    def time_to_complete(self) -> Optional[float]:
        """Return time taken to complete goal, if completed."""
        if self.completed_at is None or self.created_at is None:
            return None
        return self.completed_at - self.created_at
    
    def to_dict(self) -> Dict:
        """Convert goal to dictionary representation."""
        return {
            'name': self.name,
            'target_metric': self.target_metric,
            'target_value': self.target_value,
            'current_value': self.current_value,
            'priority': self.priority,
            'progress': self.progress(),
            'is_complete': self.is_complete(),
            'description': self.description,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'time_to_complete': self.time_to_complete(),
        }


class GoalManager:
    """
    Manages forest's learning objectives and priorities.
    
    The GoalManager maintains a list of active goals, tracks their progress,
    and manages goal completion. Goals are automatically sorted by priority.
    """
    
    def __init__(self):
        self.goals: List[LearningGoal] = []
        self.completed: List[LearningGoal] = []
        self.history: List[Dict] = []
    
    def add_goal(self, goal: LearningGoal):
        """
        Add a new learning goal to the active goals list.
        Goals are automatically sorted by priority (higher first).
        """
        self.goals.append(goal)
        self.goals.sort(key=lambda g: g.priority, reverse=True)
        self.history.append({
            'timestamp': time.time(),
            'event': 'goal_added',
            'goal_name': goal.name,
            'priority': goal.priority,
        })
    
    def create_goal(
        self,
        name: str,
        target_metric: str,
        target_value: float,
        priority: int = 1,
        description: str = ""
    ) -> LearningGoal:
        """
        Create and add a new goal in one step.
        
        Returns:
            The created LearningGoal instance
        """
        goal = LearningGoal(
            name=name,
            target_metric=target_metric,
            target_value=target_value,
            priority=priority,
            description=description,
        )
        self.add_goal(goal)
        return goal
    
    def get_active_goals(self) -> List[LearningGoal]:
        """Return list of goals that are not yet complete."""
        return [g for g in self.goals if not g.is_complete()]
    
    def get_goal_by_name(self, name: str) -> Optional[LearningGoal]:
        """Find a goal by its name (from active or completed)."""
        for goal in self.goals + self.completed:
            if goal.name == name:
                return goal
        return None
    
    def update_progress(self, metrics: Dict[str, Any]):
        """
        Update all goals with current metrics.
        Moves completed goals to completed list and prints notification.
        """
        newly_completed = []
        
        for goal in self.goals[:]:  # Copy list to allow modification during iteration
            goal.update(metrics)
            if goal.is_complete() and goal not in self.completed:
                newly_completed.append(goal)
                self.completed.append(goal)
                self.goals.remove(goal)
                self.history.append({
                    'timestamp': time.time(),
                    'event': 'goal_completed',
                    'goal_name': goal.name,
                    'time_to_complete': goal.time_to_complete(),
                })
        
        # Print notifications for newly completed goals
        for goal in newly_completed:
            time_str = f" in {goal.time_to_complete():.1f}s" if goal.time_to_complete() else ""
            print(f"🎯 Goal achieved: {goal.name}{time_str}")
    
    def get_top_priority_goals(self, n: int = 3) -> List[LearningGoal]:
        """Return the top N priority active goals."""
        active = self.get_active_goals()
        return sorted(active, key=lambda g: g.priority, reverse=True)[:n]
    
    def get_progress_summary(self) -> Dict:
        """
        Generate a summary of goal progress.
        
        Returns:
            Dictionary with statistics about active and completed goals
        """
        active = self.get_active_goals()
        
        summary = {
            'active_goals': len(active),
            'completed_goals': len(self.completed),
            'total_goals': len(self.goals) + len(self.completed),
            'completion_rate': len(self.completed) / max(1, len(self.completed) + len(active)),
            'average_progress': sum(g.progress() for g in active) / max(1, len(active)),
            'goals': [],
        }
        
        # Add details for each active goal
        for goal in active:
            summary['goals'].append({
                'name': goal.name,
                'progress': goal.progress(),
                'priority': goal.priority,
                'target': f"{goal.target_metric} >= {goal.target_value}",
                'current': goal.current_value,
            })
        
        return summary
    
    def clear_completed(self):
        """Remove all completed goals from the completed list."""
        count = len(self.completed)
        self.completed.clear()
        return count
    
    def reset(self):
        """Clear all goals (active and completed) and history."""
        self.goals.clear()
        self.completed.clear()
        self.history.clear()
    
    def __len__(self):
        return len(self.goals)
    
    def __repr__(self):
        active = self.get_active_goals()
        return f"GoalManager(active={len(active)}, completed={len(self.completed)})"
