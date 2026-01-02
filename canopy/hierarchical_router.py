"""
Hierarchical Router: Multi-level routing system for the forest canopy.
1. Detect modality (image/audio/text/video/mixed)
2. Select appropriate grove(s)
3. Route to specific trees within grove
4. Aggregate outputs with learned weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .modality_detector import ModalityDetector
from .attention_aggregator import CrossGroveAttention, AdaptiveAggregator


class GroveRouter(nn.Module):
    """
    Routes inputs to appropriate groves based on modality and task.
    """

    def __init__(
        self, num_groves: int = 4, embedding_dim: int = 512, hidden_dim: int = 128
    ):
        super().__init__()
        self.num_groves = num_groves
        self.embedding_dim = embedding_dim

        # Router network
        self.router = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_groves),
        )

    def forward(
        self, x: torch.Tensor, active_groves: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute routing scores for each grove.

        Args:
            x: Input features [B, embedding_dim]
            active_groves: Optional list of active grove names

        Returns:
            scores: Routing scores [B, num_groves]
        """
        scores = self.router(x)  # [B, num_groves]
        return scores

    def get_top_groves(
        self, x: torch.Tensor, top_k: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k groves for the input.

        Args:
            x: Input features [B, embedding_dim]
            top_k: Number of groves to select

        Returns:
            indices: Top grove indices [B, top_k]
            weights: Softmax weights for top groves [B, top_k]
        """
        scores = self.forward(x)  # [B, num_groves]

        # Get top-k
        topk_values, topk_indices = torch.topk(scores, k=top_k, dim=1)

        # Apply softmax to top-k scores
        weights = F.softmax(topk_values, dim=1)

        return topk_indices, weights


class ForestCanopy(nn.Module):
    """
    Multi-level routing system for the forest.
    Handles modality detection, grove selection, and output aggregation.
    """

    def __init__(
        self,
        grove_dict: Optional[Dict[str, nn.Module]] = None,
        embedding_dim: int = 512,
        num_heads: int = 8,
    ):
        """
        Initialize Forest Canopy.

        Args:
            grove_dict: Dictionary mapping grove names to grove modules
            embedding_dim: Embedding dimension
            num_heads: Number of attention heads for aggregation
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Initialize groves (can be empty initially)
        if grove_dict is None:
            grove_dict = {}
        self.groves = nn.ModuleDict(grove_dict)

        # Level 1: Modality detection
        self.modality_detector = ModalityDetector(embedding_dim=embedding_dim)

        # Level 2: Grove selection
        num_groves = len(grove_dict) if grove_dict else 4
        self.grove_router = GroveRouter(
            num_groves=num_groves, embedding_dim=embedding_dim
        )

        # Level 3: Cross-grove attention for final aggregation
        self.cross_grove_attention = CrossGroveAttention(
            embed_dim=embedding_dim, num_heads=num_heads
        )

        # Adaptive aggregator as alternative
        self.adaptive_aggregator = AdaptiveAggregator(
            embed_dim=embedding_dim,
            num_groves=num_groves,
            num_heads=num_heads,
            use_attention=True,
        )

        self.modality_names = ["image", "audio", "text", "video"]

    def add_grove(self, name: str, grove: nn.Module):
        """
        Add a grove to the canopy.

        Args:
            name: Name of the grove (e.g., "image", "audio")
            grove: Grove module
        """
        self.groves[name] = grove

    def forward(
        self,
        x: torch.Tensor,
        modality_hint: Optional[str] = None,
        top_k_groves: int = 2,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Route input through the canopy to appropriate groves.

        Args:
            x: Input features [B, embedding_dim]
            modality_hint: Optional modality hint to skip detection
            top_k_groves: Number of groves to activate

        Returns:
            output: Final aggregated output [B, 1]
            routing_info: Dictionary with routing information
        """
        batch_size = x.shape[0]

        # Level 1: Detect modality if not provided
        if modality_hint is None:
            modality_probs = self.modality_detector(x)
            detected_modalities = self._select_modalities(modality_probs, threshold=0.3)
        else:
            detected_modalities = [modality_hint]
            modality_probs = {modality_hint: torch.ones(batch_size, device=x.device)}

        # Level 2: Route to selected groves
        if not self.groves:
            # No groves available, return zeros
            return torch.zeros(batch_size, 1, device=x.device), {
                "modalities": detected_modalities,
                "groves_used": [],
                "error": "No groves available",
            }

        grove_outputs = []
        grove_weights_list = []
        groves_used = []

        # Get grove routing scores
        grove_scores = self.grove_router(x)

        # Activate top-k groves
        top_k = min(top_k_groves, len(self.groves))
        topk_values, topk_indices = torch.topk(grove_scores, k=top_k, dim=1)
        grove_routing_weights = F.softmax(topk_values, dim=1)

        # Route to selected groves
        grove_names = list(self.groves.keys())
        for i in range(top_k):
            grove_idx = topk_indices[0, i].item()  # Use first sample's routing
            if grove_idx < len(grove_names):
                grove_name = grove_names[grove_idx]
                grove = self.groves[grove_name]

                try:
                    # Forward through grove
                    grove_output, tree_weights = grove(x, top_k=3)
                    grove_outputs.append(grove_output)
                    grove_weights_list.append(grove_routing_weights[:, i : i + 1])
                    groves_used.append(grove_name)
                except Exception as e:
                    # Skip this grove if error occurs
                    print(f"Warning: Grove {grove_name} failed: {e}")
                    continue

        # Level 3: Aggregate outputs across groves
        if not grove_outputs:
            return torch.zeros(batch_size, 1, device=x.device), {
                "modalities": detected_modalities,
                "groves_used": [],
                "error": "No valid grove outputs",
            }

        if len(grove_outputs) == 1:
            # Single grove, no aggregation needed
            final_output = grove_outputs[0]
        else:
            # Multiple groves, use weighted aggregation
            stacked_outputs = torch.stack(grove_outputs, dim=1)  # [B, num_groves, 1]
            stacked_weights = torch.stack(
                grove_weights_list, dim=1
            )  # [B, num_groves, 1]

            # Normalize weights
            stacked_weights = stacked_weights / stacked_weights.sum(dim=1, keepdim=True)

            # Weighted sum
            final_output = (stacked_outputs * stacked_weights).sum(dim=1)

        # Routing information
        routing_info = {
            "modalities": detected_modalities,
            "modality_probs": {k: v[0].item() for k, v in modality_probs.items()},
            "groves_used": groves_used,
            "grove_weights": grove_routing_weights[0].tolist()[:top_k],
        }

        return final_output, routing_info

    def _select_modalities(
        self, modality_probs: Dict[str, torch.Tensor], threshold: float = 0.3
    ) -> List[str]:
        """
        Select modalities based on probability threshold.

        Args:
            modality_probs: Dictionary of modality probabilities
            threshold: Minimum probability to select a modality

        Returns:
            List of selected modality names
        """
        selected = []

        for modality, probs in modality_probs.items():
            # Use first sample's probability
            if probs[0].item() > threshold:
                selected.append(modality)

        # If nothing selected, pick the top one
        if not selected:
            max_modality = max(modality_probs.items(), key=lambda x: x[1][0].item())
            selected = [max_modality[0]]

        return selected

    def get_canopy_stats(self) -> Dict:
        """Get statistics about the canopy."""
        return {
            "num_groves": len(self.groves),
            "grove_names": list(self.groves.keys()),
            "embedding_dim": self.embedding_dim,
        }

    def route_summary(self, x: torch.Tensor) -> Dict:
        """
        Get a detailed routing summary without actually forwarding.

        Args:
            x: Input features [B, embedding_dim]

        Returns:
            Dictionary with routing analysis
        """
        # Detect modality
        modality_probs = self.modality_detector(x)
        top_modality, confidence = self.modality_detector.get_modality_confidence(x)

        # Get grove routing scores
        grove_scores = self.grove_router(x)
        grove_probs = F.softmax(grove_scores[0], dim=0)

        grove_names = list(self.groves.keys())
        grove_routing = {
            grove_names[i]: grove_probs[i].item()
            for i in range(min(len(grove_names), len(grove_probs)))
        }

        return {
            "detected_modality": top_modality,
            "modality_confidence": confidence,
            "modality_probs": {k: v[0].item() for k, v in modality_probs.items()},
            "grove_routing_probs": grove_routing,
        }
