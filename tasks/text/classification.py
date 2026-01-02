"""
Text Classification tasks for NeuralForest.
"""

import torch
import torch.nn as nn
from ..base import TaskHead, TaskRegistry


class TextClassification(TaskHead):
    """
    Text classification task head.
    Supports sentiment analysis, topic classification, intent detection, etc.
    """
    
    SUPPORTED_TASKS = [
        "sentiment_analysis", "topic_classification",
        "intent_detection", "spam_detection"
    ]
    
    def __init__(self, input_dim: int, num_classes: int,
                 dropout: float = 0.3,
                 task_name: str = "text_classification"):
        """
        Initialize text classification head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of classes
            dropout: Dropout probability
            task_name: Task identifier
        """
        super().__init__(input_dim, task_name)
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
    
    def get_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(predictions, targets)


TaskRegistry.register("text_classification", TextClassification)
TaskRegistry.register("sentiment_analysis", TextClassification)
