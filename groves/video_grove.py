"""
Video Grove: Specialized tree cluster for video processing tasks.
Handles video-related tasks like action recognition, video classification, etc.
"""

import torch
from typing import List
from .base_grove import Grove


class VideoGrove(Grove):
    """
    A grove specialized for video processing tasks.
    Trees in this grove handle various video understanding tasks.
    """
    
    # Video task specializations
    SPECIALIZATIONS = [
        "action_recognition",
        "video_classification",
        "scene_detection",
        "temporal_analysis",
        "event_detection",
        "motion_analysis",
    ]
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 64, max_trees: int = 12):
        """
        Initialize Video Grove.
        
        Args:
            input_dim: Dimension of input features from VideoSoil
            hidden_dim: Hidden dimension for tree networks
            max_trees: Maximum number of trees in this grove
        """
        super().__init__(
            modality="video",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_trees=max_trees
        )
        
        # Plant initial specialist trees
        self._plant_initial_specialists()
    
    def _plant_initial_specialists(self):
        """Plant a few initial specialist trees for common video tasks."""
        initial_specialists = ["action_recognition", "video_classification"]
        
        for spec in initial_specialists:
            self.plant_specialist(spec)
    
    def get_suggested_specializations(self, task_performance: dict) -> List[str]:
        """
        Suggest new specializations based on task performance gaps.
        
        Args:
            task_performance: Dict mapping task names to performance scores
            
        Returns:
            List of suggested specializations to plant
        """
        suggestions = []
        
        # Check which specializations are missing
        current_specs = {tree.specialization for tree in self.trees}
        
        for spec in self.SPECIALIZATIONS:
            if spec not in current_specs:
                # Suggest if this specialization might be needed
                if spec in task_performance and task_performance[spec] < 0.7:
                    suggestions.append(spec)
        
        return suggestions
