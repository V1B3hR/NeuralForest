"""
Video Classification task for NeuralForest.
"""

import torch
import torch.nn as nn
from ..base import TaskHead, TaskRegistry


class VideoClassification(TaskHead):
    """
    Video classification task head.
    """
    
    SUPPORTED_DATASETS = ["ucf101", "kinetics", "hmdb51", "something_something"]
    
    def __init__(self, input_dim: int, num_classes: int,
                 dropout: float = 0.3,
                 task_name: str = "video_classification"):
        """
        Initialize video classification head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of video classes
            dropout: Dropout probability
            task_name: Task identifier
        """
        super().__init__(input_dim, task_name)
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
    
    def get_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(predictions, targets)


TaskRegistry.register("video_classification", VideoClassification)
