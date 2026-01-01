"""
Base SoilProcessor class for media-specific preprocessing.
Transforms raw input into nutrient-rich embeddings for trees.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SoilProcessor(nn.Module, ABC):
    """
    Base class for modality-specific preprocessing.
    Each soil processor transforms raw media input into embeddings.
    """
    
    def __init__(self, modality: str, output_dim: int = 512):
        super().__init__()
        self.modality = modality
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process raw input into embeddings.
        
        Args:
            x: Raw input tensor (format depends on modality)
            
        Returns:
            embeddings: [B, output_dim] tensor
        """
        pass
    
    def get_modality(self) -> str:
        """Return the modality this processor handles."""
        return self.modality
    
    def get_output_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim
