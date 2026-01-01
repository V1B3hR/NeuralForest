"""
TextSoil: Processes text using tokenization + transformer layers.
- Subword tokenization
- Positional encoding
- Contextual embeddings
"""

import torch
import torch.nn as nn
from .base import SoilProcessor


class TextSoil(SoilProcessor):
    """
    Processes text tokens into embeddings.
    Uses transformer-based encoding for contextual understanding.
    """
    
    def __init__(self, vocab_size: int = 30000, output_dim: int = 512,
                 max_seq_length: int = 512):
        super().__init__(modality="text", output_dim=output_dim)
        
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, output_dim, padding_idx=0)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(max_seq_length, output_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=8,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Process text tokens into embeddings.
        
        Args:
            x: Token IDs tensor [B, seq_len]
            attention_mask: Optional mask [B, seq_len] (1 for real tokens, 0 for padding)
            
        Returns:
            embeddings: [B, output_dim] - mean pooling over sequence
        """
        B, seq_len = x.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(x)  # [B, seq_len, output_dim]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)
        position_embeds = self.position_embedding(positions)  # [B, seq_len, output_dim]
        
        # Combined embeddings
        embeds = token_embeds + position_embeds  # [B, seq_len, output_dim]
        embeds = self.layer_norm(embeds)
        
        # Create attention mask for transformer (inverted: 0 = attend, -inf = ignore)
        if attention_mask is not None:
            # Convert binary mask to float mask for transformer
            # attention_mask: [B, seq_len] with 1 for real tokens, 0 for padding
            mask = (attention_mask == 0)  # [B, seq_len]
        else:
            mask = None
        
        # Transformer encoding
        encoded = self.transformer(embeds, src_key_padding_mask=mask)  # [B, seq_len, output_dim]
        
        # Mean pooling (considering attention mask)
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            sum_embeddings = (encoded * mask_expanded).sum(dim=1)  # [B, output_dim]
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1]
            embeddings = sum_embeddings / sum_mask  # [B, output_dim]
        else:
            # Simple mean pooling
            embeddings = encoded.mean(dim=1)  # [B, output_dim]
        
        return embeddings


class SimpleTextSoil(SoilProcessor):
    """
    Simplified text processor using BiLSTM.
    More lightweight alternative to transformer-based TextSoil.
    """
    
    def __init__(self, vocab_size: int = 30000, output_dim: int = 512,
                 embedding_dim: int = 256, hidden_dim: int = 256):
        super().__init__(modality="text", output_dim=output_dim)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Projection layer
        self.projector = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Process text tokens into embeddings using LSTM.
        
        Args:
            x: Token IDs tensor [B, seq_len]
            attention_mask: Optional mask [B, seq_len] (1 for real tokens, 0 for padding)
            
        Returns:
            embeddings: [B, output_dim]
        """
        # Embeddings
        embeds = self.embedding(x)  # [B, seq_len, embedding_dim]
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(embeds)  # lstm_out: [B, seq_len, hidden_dim*2]
        
        # Use final hidden states (concatenate forward and backward)
        # hidden shape: [4, B, hidden_dim] (2 layers * 2 directions)
        forward_hidden = hidden[-2]  # [B, hidden_dim]
        backward_hidden = hidden[-1]  # [B, hidden_dim]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)  # [B, hidden_dim*2]
        
        # Project to output dimension
        embeddings = self.projector(combined)  # [B, output_dim]
        
        return embeddings
