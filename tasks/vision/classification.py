"""
Image Classification task for NeuralForest.

Supports various image classification scenarios.
"""

import torch
import torch.nn as nn
from ..base import TaskHead, TaskRegistry


class ImageClassification(TaskHead):
    """
    Image classification task head.

    Supports:
    - Multi-class classification
    - Fine-grained classification
    - Scene recognition
    """

    SUPPORTED_DATASETS = [
        "imagenet",
        "cifar10",
        "cifar100",
        "mnist",
        "fashion_mnist",
        "svhn",
        "places365",
        "caltech101",
    ]

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        task_name: str = "image_classification",
    ):
        """
        Initialize image classification head.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            dropout: Dropout probability
            task_name: Task identifier
        """
        super().__init__(input_dim, task_name)
        self.num_classes = num_classes
        self.dropout = dropout

        # Multi-layer classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.BatchNorm1d(input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(input_dim // 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Class logits [batch_size, num_classes]
        """
        return self.classifier(x)

    def get_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss.

        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Cross-entropy loss
        """
        return nn.functional.cross_entropy(predictions, targets)

    def predict(self, x: torch.Tensor, return_probs: bool = False):
        """
        Make predictions on input.

        Args:
            x: Input features
            return_probs: If True, return probabilities instead of class indices

        Returns:
            Predicted classes or probabilities
        """
        logits = self.forward(x)
        if return_probs:
            return torch.softmax(logits, dim=-1)
        return torch.argmax(logits, dim=-1)


# Register with task registry
TaskRegistry.register("image_classification", ImageClassification)
