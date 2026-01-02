"""
Semantic Segmentation task for NeuralForest.
"""

import torch
import torch.nn as nn
from ..base import TaskHead, TaskRegistry


class SemanticSegmentation(TaskHead):
    """
    Semantic segmentation task head.

    Outputs per-pixel class predictions.
    """

    SUPPORTED_DATASETS = ["cityscapes", "ade20k", "pascal_voc", "coco_stuff"]

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        spatial_size: int = 32,
        task_name: str = "semantic_segmentation",
    ):
        """
        Initialize segmentation head.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of segmentation classes
            spatial_size: Spatial dimension of output (e.g., 32x32)
            task_name: Task identifier
        """
        super().__init__(input_dim, task_name)
        self.num_classes = num_classes
        self.spatial_size = spatial_size

        # Upsampling layers
        hidden_dim = input_dim // 2
        self.segmenter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, spatial_size * spatial_size * num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for segmentation.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Segmentation map [batch_size, num_classes, H, W]
        """
        batch_size = x.size(0)
        output = self.segmenter(x)

        # Reshape to spatial format
        return output.view(
            batch_size, self.num_classes, self.spatial_size, self.spatial_size
        )

    def get_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute segmentation loss.

        Args:
            predictions: Predicted segmentation map [B, C, H, W]
            targets: Ground truth segmentation map [B, H, W]

        Returns:
            Cross-entropy loss
        """
        # Flatten spatial dimensions
        batch_size, num_classes, H, W = predictions.shape
        predictions_flat = predictions.permute(0, 2, 3, 1).reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)

        return nn.functional.cross_entropy(predictions_flat, targets_flat)


TaskRegistry.register("semantic_segmentation", SemanticSegmentation)
