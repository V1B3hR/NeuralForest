"""
RootNetwork: Unified Representation Layer
Combines modality-specific embeddings into unified representation.
Enables cross-modal understanding and knowledge transfer.
"""

import torch
import torch.nn as nn
import warnings
from typing import Dict, Optional


class RootNetwork(nn.Module):
    """
    Combines modality-specific embeddings into unified representation.
    Uses projectors and cross-modal attention for multi-input scenarios.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        modality_dims: Optional[Dict[str, int]] = None,
        num_heads: int = 8,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Default dimensions for each modality
        if modality_dims is None:
            modality_dims = {
                "image": 512,
                "audio": 512,
                "text": 512,
                "video": 512,
            }

        # Projectors for each modality to common embedding space
        self.projectors = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                )
                for modality, dim in modality_dims.items()
            }
        )

        # Cross-modal attention for multi-input scenarios
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine multiple modality inputs into unified representation.

        Args:
            inputs: Dictionary mapping modality names to tensors
                   e.g., {"image": [B, 512], "audio": [B, 512]}

        Returns:
            unified_embedding: [B, embedding_dim]
        """
        if not inputs:
            raise ValueError("At least one modality input is required")

        # Project each modality to common space
        embeddings = []
        modalities = []

        for modality, tensor in inputs.items():
            if modality not in self.projectors:
                raise ValueError(f"Unknown modality: {modality}")

            proj = self.projectors[modality](tensor)  # [B, embedding_dim]
            embeddings.append(proj)
            modalities.append(modality)

        # Single modality: just return projection
        if len(embeddings) == 1:
            return self.output_projection(embeddings[0])

        # Multiple modalities: fuse with cross-attention
        # Stack embeddings: [B, num_modalities, embedding_dim]
        stacked = torch.stack(embeddings, dim=1)

        # Self-attention across modalities
        fused, attention_weights = self.cross_attention(
            stacked, stacked, stacked
        )  # fused: [B, num_modalities, embedding_dim]

        # Aggregate: mean pooling across modalities
        unified = fused.mean(dim=1)  # [B, embedding_dim]

        # Final output projection
        output = self.output_projection(unified)  # [B, embedding_dim]

        return output

    def get_attention_weights(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Get cross-modal attention weights (for analysis/visualization).

        Args:
            inputs: Dictionary mapping modality names to tensors

        Returns:
            attention_weights: [B, num_heads, num_modalities, num_modalities] or None
        """
        if len(inputs) <= 1:
            return None

        # Project each modality to common space
        embeddings = []
        for modality, tensor in inputs.items():
            if modality in self.projectors:
                proj = self.projectors[modality](tensor)
                embeddings.append(proj)

        if len(embeddings) <= 1:
            return None

        # Stack and get attention weights
        stacked = torch.stack(embeddings, dim=1)
        _, attention_weights = self.cross_attention(stacked, stacked, stacked)

        return attention_weights


class SimpleRootNetwork(nn.Module):
    """
    Simplified root network without cross-attention.
    Uses mean-pooling and MLP for multi-modal fusion.
    """

    def __init__(
        self, embedding_dim: int = 512, modality_dims: Optional[Dict[str, int]] = None
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Default dimensions
        if modality_dims is None:
            modality_dims = {
                "image": 512,
                "audio": 512,
                "text": 512,
                "video": 512,
            }

        # Simple projectors
        self.projectors = nn.ModuleDict(
            {
                modality: nn.Linear(dim, embedding_dim)
                for modality, dim in modality_dims.items()
            }
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine multiple modality inputs via mean-pooling + MLP.

        Args:
            inputs: Dictionary mapping modality names to tensors

        Returns:
            unified_embedding: [B, embedding_dim]
        """
        if not inputs:
            raise ValueError("At least one modality input is required")

        # Project each modality
        embeddings = []
        for modality, tensor in inputs.items():
            if modality not in self.projectors:
                warnings.warn(f"Unknown modality '{modality}' will be ignored.")
                continue
            proj = self.projectors[modality](tensor)
            embeddings.append(proj)

        if not embeddings:
            raise ValueError("No valid modality inputs found")

        # Stack and mean-pool modality embeddings
        stacked = torch.stack(embeddings, dim=1)  # [B, N, embedding_dim]
        pooled = stacked.mean(dim=1)  # [B, embedding_dim]

        return self.fusion_mlp(pooled)
