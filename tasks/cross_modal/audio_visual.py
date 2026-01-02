"""
Audio-Visual cross-modal tasks for NeuralForest.
"""

import torch
import torch.nn as nn
from ..base import TaskHead, TaskRegistry


class AudioVisualCorrespondence(TaskHead):
    """
    Audio-visual correspondence task.
    Determines if audio and video are aligned/synchronized.
    """

    SUPPORTED_DATASETS = ["audioset", "vggsound", "kinetics_sound"]

    def __init__(self, input_dim: int, task_name: str = "audio_visual_correspondence"):
        """
        Initialize audio-visual correspondence head.

        Args:
            input_dim: Input feature dimension
            task_name: Task identifier
        """
        super().__init__(input_dim, task_name)

        # Binary classification: aligned or not
        self.correspondence_head = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),  # Concatenated audio+visual
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 2),  # Binary: aligned or not
        )

    def forward(
        self, audio_features: torch.Tensor, visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict audio-visual correspondence.

        Args:
            audio_features: Audio feature embeddings
            visual_features: Visual feature embeddings

        Returns:
            Correspondence logits [batch_size, 2]
        """
        # Concatenate audio and visual features
        combined = torch.cat([audio_features, visual_features], dim=-1)
        return self.correspondence_head(combined)

    def get_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute correspondence loss.

        Args:
            predictions: Predicted correspondence [batch_size, 2]
            targets: Ground truth labels [batch_size] (0 or 1)

        Returns:
            Binary cross-entropy loss
        """
        return nn.functional.cross_entropy(predictions, targets)


TaskRegistry.register("audio_visual_correspondence", AudioVisualCorrespondence)
