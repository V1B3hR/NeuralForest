"""
Cross-Grove Attention: Aggregates outputs from multiple groves using attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class CrossGroveAttention(nn.Module):
    """
    Cross-attention mechanism for aggregating outputs from multiple groves.
    Allows groves to attend to each other's outputs for better fusion.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize cross-grove attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, grove_outputs: torch.Tensor, 
                grove_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-grove attention to aggregate grove outputs.
        
        Args:
            grove_outputs: Outputs from groves [B, num_groves, embed_dim]
            grove_mask: Optional mask for inactive groves [B, num_groves]
            
        Returns:
            aggregated: Aggregated output [B, embed_dim]
            attention_weights: Attention weights [B, num_heads, num_groves, num_groves]
        """
        # Self-attention across groves
        attn_output, attn_weights = self.attention(
            grove_outputs,
            grove_outputs,
            grove_outputs,
            key_padding_mask=grove_mask
        )
        
        # Residual connection and normalization
        grove_outputs = self.norm1(grove_outputs + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(grove_outputs)
        grove_outputs = self.norm2(grove_outputs + self.dropout(ffn_output))
        
        # Aggregate across groves (mean pooling)
        aggregated = grove_outputs.mean(dim=1)  # [B, embed_dim]
        
        return aggregated, attn_weights
    
    def forward_with_weights(self, 
                            grove_outputs: torch.Tensor,
                            grove_weights: torch.Tensor) -> torch.Tensor:
        """
        Aggregate grove outputs using provided weights (no attention).
        
        Args:
            grove_outputs: Outputs from groves [B, num_groves, embed_dim]
            grove_weights: Weights for each grove [B, num_groves]
            
        Returns:
            aggregated: Weighted aggregation [B, embed_dim]
        """
        # Normalize weights
        weights = F.softmax(grove_weights, dim=-1)  # [B, num_groves]
        weights = weights.unsqueeze(-1)  # [B, num_groves, 1]
        
        # Weighted sum
        aggregated = (grove_outputs * weights).sum(dim=1)  # [B, embed_dim]
        
        return aggregated


class AdaptiveAggregator(nn.Module):
    """
    Adaptive aggregator that learns to combine grove outputs.
    Can use attention or learned weights.
    """
    def __init__(self, embed_dim: int = 512, num_groves: int = 4, 
                 num_heads: int = 8, use_attention: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groves = num_groves
        self.use_attention = use_attention
        
        if use_attention:
            # Use cross-attention for aggregation
            self.aggregator = CrossGroveAttention(embed_dim, num_heads)
        else:
            # Use learned linear weights
            self.weight_predictor = nn.Sequential(
                nn.Linear(embed_dim * num_groves, 256),
                nn.ReLU(),
                nn.Linear(256, num_groves),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, grove_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate outputs from multiple groves.
        
        Args:
            grove_outputs: List of grove outputs, each [B, embed_dim]
            
        Returns:
            aggregated: Combined output [B, embed_dim]
        """
        if not grove_outputs:
            raise ValueError("No grove outputs to aggregate")
        
        # Stack grove outputs
        stacked = torch.stack(grove_outputs, dim=1)  # [B, num_groves, embed_dim]
        
        if self.use_attention:
            # Use attention-based aggregation
            aggregated, _ = self.aggregator(stacked)
        else:
            # Use learned weights
            flattened = stacked.view(stacked.shape[0], -1)  # [B, num_groves * embed_dim]
            weights = self.weight_predictor(flattened)  # [B, num_groves]
            
            # Weighted combination
            weights_expanded = weights.unsqueeze(-1)  # [B, num_groves, 1]
            aggregated = (stacked * weights_expanded).sum(dim=1)  # [B, embed_dim]
        
        return aggregated
    
    def get_aggregation_weights(self, grove_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Get the aggregation weights used for combining groves.
        
        Args:
            grove_outputs: List of grove outputs, each [B, embed_dim]
            
        Returns:
            weights: Aggregation weights [B, num_groves]
        """
        stacked = torch.stack(grove_outputs, dim=1)
        
        if self.use_attention:
            # For attention, return uniform weights (actual attention is more complex)
            batch_size = stacked.shape[0]
            return torch.ones(batch_size, len(grove_outputs), device=stacked.device) / len(grove_outputs)
        else:
            # Return learned weights
            flattened = stacked.view(stacked.shape[0], -1)
            weights = self.weight_predictor(flattened)
            return weights
