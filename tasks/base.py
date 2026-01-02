"""
Base classes for NeuralForest task implementations.

Provides foundation for all task-specific heads and implementations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


class TaskHead(nn.Module, ABC):
    """
    Base class for task-specific output heads.

    Each task (classification, detection, etc.) should subclass this
    and implement the forward method.
    """

    def __init__(self, input_dim: int, task_name: str):
        """
        Initialize task head.

        Args:
            input_dim: Dimension of input features
            task_name: Name of the task
        """
        super().__init__()
        self.input_dim = input_dim
        self.task_name = task_name

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through task head.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Task-specific output tensor
        """
        pass

    def get_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute task-specific loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Loss tensor
        """
        raise NotImplementedError("Subclass must implement get_loss()")


class ClassificationHead(TaskHead):
    """Head for classification tasks."""

    def __init__(
        self, input_dim: int, num_classes: int, task_name: str = "classification"
    ):
        super().__init__(input_dim, task_name)
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def get_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.cross_entropy(predictions, targets)


class DetectionHead(TaskHead):
    """Head for object detection tasks."""

    def __init__(self, input_dim: int, num_classes: int, task_name: str = "detection"):
        super().__init__(input_dim, task_name)
        self.num_classes = num_classes

        # Simplified detection head (bbox coords + class)
        self.detector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, num_classes + 4),  # 4 for bbox coords
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.detector(x)
        # Split into bbox and class predictions
        return output


class SegmentationHead(TaskHead):
    """Head for segmentation tasks."""

    def __init__(
        self, input_dim: int, num_classes: int, task_name: str = "segmentation"
    ):
        super().__init__(input_dim, task_name)
        self.num_classes = num_classes

        self.segmenter = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segmenter(x)


class RegressionHead(TaskHead):
    """Head for regression tasks."""

    def __init__(
        self, input_dim: int, output_dim: int = 1, task_name: str = "regression"
    ):
        super().__init__(input_dim, task_name)
        self.output_dim = output_dim

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)

    def get_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.mse_loss(predictions, targets)


class GenerationHead(TaskHead):
    """Head for sequence generation tasks."""

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        max_length: int = 50,
        task_name: str = "generation",
    ):
        super().__init__(input_dim, task_name)
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.generator = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, vocab_size * max_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        output = self.generator(x)
        # Reshape to [batch, max_length, vocab_size]
        return output.view(batch_size, self.max_length, self.vocab_size)


class TaskRegistry:
    """
    Central registry for all available tasks.
    Allows dynamic task lookup and instantiation.
    """

    _tasks = {}

    @classmethod
    def register(cls, task_name: str, task_class: type):
        """Register a task class."""
        cls._tasks[task_name] = task_class

    @classmethod
    def get(cls, task_name: str) -> Optional[type]:
        """Get a registered task class."""
        return cls._tasks.get(task_name)

    @classmethod
    def list_tasks(cls) -> List[str]:
        """List all registered tasks."""
        return list(cls._tasks.keys())

    @classmethod
    def create_head(
        cls, task_name: str, input_dim: int, **kwargs
    ) -> Optional[TaskHead]:
        """
        Create a task head instance.

        Args:
            task_name: Name of the task
            input_dim: Input dimension for the head
            **kwargs: Additional task-specific parameters

        Returns:
            TaskHead instance or None if task not found
        """
        task_class = cls.get(task_name)
        if task_class is None:
            return None
        return task_class(input_dim=input_dim, **kwargs)


# Register base task heads
TaskRegistry.register("classification", ClassificationHead)
TaskRegistry.register("detection", DetectionHead)
TaskRegistry.register("segmentation", SegmentationHead)
TaskRegistry.register("regression", RegressionHead)
TaskRegistry.register("generation", GenerationHead)


class TaskConfig:
    """Configuration for a specific task."""

    def __init__(
        self,
        task_name: str,
        modality: str,
        head_type: str,
        input_dim: int,
        output_params: Dict[str, Any],
    ):
        """
        Initialize task configuration.

        Args:
            task_name: Name of the task
            modality: Input modality (image, audio, text, video, cross_modal)
            head_type: Type of head to use
            input_dim: Input feature dimension
            output_params: Parameters for output head (e.g., num_classes)
        """
        self.task_name = task_name
        self.modality = modality
        self.head_type = head_type
        self.input_dim = input_dim
        self.output_params = output_params

    def create_head(self) -> TaskHead:
        """Create task head from configuration."""
        # Get the task class
        task_class = TaskRegistry.get(self.head_type)
        if task_class is None:
            raise ValueError(f"Unknown head type: {self.head_type}")

        # Create instance with proper parameters
        return task_class(
            input_dim=self.input_dim, task_name=self.task_name, **self.output_params
        )
