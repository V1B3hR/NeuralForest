"""
Action Recognition task for NeuralForest.
"""

import torch
import torch.nn as nn
from ..base import TaskHead, TaskRegistry


class ActionRecognition(TaskHead):
    """
    Action recognition in videos.
    Temporal action detection and classification.
    """

    SUPPORTED_DATASETS = ["activitynet", "charades", "ava"]

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        temporal_segments: int = 8,
        task_name: str = "action_recognition",
    ):
        """
        Initialize action recognition head.

        Args:
            input_dim: Input feature dimension
            num_actions: Number of action classes
            temporal_segments: Number of temporal segments
            task_name: Task identifier
        """
        super().__init__(input_dim, task_name)
        self.num_actions = num_actions
        self.temporal_segments = temporal_segments

        # Temporal pooling and classification
        self.action_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_head(x)

    def get_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.cross_entropy(predictions, targets)


TaskRegistry.register("action_recognition", ActionRecognition)
