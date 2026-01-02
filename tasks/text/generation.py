"""
Text Generation task for NeuralForest.
"""

import torch
import torch.nn as nn
from ..base import TaskHead, TaskRegistry


class TextGeneration(TaskHead):
    """
    Text generation task head.
    Supports summarization, translation, question answering, etc.
    """

    SUPPORTED_TASKS = ["summarization", "translation", "question_answering", "dialogue"]

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        max_length: int = 100,
        task_name: str = "text_generation",
    ):
        """
        Initialize text generation head.

        Args:
            input_dim: Input feature dimension
            vocab_size: Vocabulary size
            max_length: Maximum generation length
            task_name: Task identifier
        """
        super().__init__(input_dim, task_name)
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.generator = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, max_length * vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for generation.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Generation logits [batch_size, max_length, vocab_size]
        """
        batch_size = x.size(0)
        output = self.generator(x)
        return output.view(batch_size, self.max_length, self.vocab_size)

    def get_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute generation loss."""
        batch_size, seq_len, vocab_size = predictions.shape
        predictions_flat = predictions.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        return nn.functional.cross_entropy(
            predictions_flat, targets_flat, ignore_index=-1
        )


TaskRegistry.register("text_generation", TextGeneration)
TaskRegistry.register("summarization", TextGeneration)
