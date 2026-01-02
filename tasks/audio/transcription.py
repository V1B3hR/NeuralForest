"""
Speech Recognition task for NeuralForest.
"""

import torch
import torch.nn as nn
from ..base import TaskHead, TaskRegistry


class SpeechRecognition(TaskHead):
    """
    Speech recognition / transcription task head.
    """

    SUPPORTED_DATASETS = ["librispeech", "common_voice", "tedlium"]

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        max_length: int = 100,
        task_name: str = "speech_recognition",
    ):
        """
        Initialize speech recognition head.

        Args:
            input_dim: Input feature dimension
            vocab_size: Size of character/token vocabulary
            max_length: Maximum sequence length
            task_name: Task identifier
        """
        super().__init__(input_dim, task_name)
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Sequence generation layers
        self.transcriber = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, max_length * vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transcription.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Transcription logits [batch_size, max_length, vocab_size]
        """
        batch_size = x.size(0)
        output = self.transcriber(x)
        return output.view(batch_size, self.max_length, self.vocab_size)

    def get_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute CTC or cross-entropy loss for transcription."""
        batch_size, seq_len, vocab_size = predictions.shape
        predictions_flat = predictions.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        return nn.functional.cross_entropy(
            predictions_flat, targets_flat, ignore_index=-1
        )


TaskRegistry.register("speech_recognition", SpeechRecognition)
