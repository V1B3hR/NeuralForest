"""
Named Entity Recognition task for NeuralForest.
"""

import torch
import torch.nn as nn
from ..base import TaskHead, TaskRegistry


class NamedEntityRecognition(TaskHead):
    """
    Named Entity Recognition task head.
    Token-level classification for entity identification.
    """
    
    ENTITY_TYPES = ["PERSON", "ORG", "LOC", "DATE", "MISC", "O"]
    
    def __init__(self, input_dim: int, num_entity_types: int,
                 max_length: int = 128,
                 task_name: str = "named_entity_recognition"):
        """
        Initialize NER head.
        
        Args:
            input_dim: Input feature dimension
            num_entity_types: Number of entity types
            max_length: Maximum sequence length
            task_name: Task identifier
        """
        super().__init__(input_dim, task_name)
        self.num_entity_types = num_entity_types
        self.max_length = max_length
        
        # Token-level classifier
        self.ner_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, max_length * num_entity_types)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for NER.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Entity predictions [batch_size, max_length, num_entity_types]
        """
        batch_size = x.size(0)
        output = self.ner_head(x)
        return output.view(batch_size, self.max_length, self.num_entity_types)
    
    def get_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute NER loss."""
        batch_size, seq_len, num_types = predictions.shape
        predictions_flat = predictions.reshape(-1, num_types)
        targets_flat = targets.reshape(-1)
        return nn.functional.cross_entropy(predictions_flat, targets_flat, ignore_index=-1)


TaskRegistry.register("named_entity_recognition", NamedEntityRecognition)
TaskRegistry.register("ner", NamedEntityRecognition)
