"""
ImageSoil: Processes images using CNN/ViT backbone.
- Patch embedding (like ViT) or convolutional feature extraction
- Spatial awareness preservation
"""

import torch
import torch.nn as nn
from .base import SoilProcessor


class ImageSoil(SoilProcessor):
    """
    Processes images into embeddings.
    Uses a simple CNN backbone for feature extraction.
    """
    
    def __init__(self, input_channels: int = 3, output_dim: int = 512, 
                 image_size: int = 224):
        super().__init__(modality="image", output_dim=output_dim)
        
        self.input_channels = input_channels
        self.image_size = image_size
        
        # Simple CNN backbone
        self.conv_layers = nn.Sequential(
            # Layer 1: 3x224x224 -> 64x112x112
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # -> 64x56x56
            
            # Layer 2: 64x56x56 -> 128x28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 3: 128x28x28 -> 256x14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4: 256x14x14 -> 512x7x7
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling + projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Linear(512, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process image tensor into embeddings.
        
        Args:
            x: Image tensor [B, C, H, W]
            
        Returns:
            embeddings: [B, output_dim]
        """
        # Convolutional feature extraction
        features = self.conv_layers(x)  # [B, 512, H', W']
        
        # Global pooling
        pooled = self.global_pool(features)  # [B, 512, 1, 1]
        pooled = pooled.flatten(1)  # [B, 512]
        
        # Project to output dimension
        embeddings = self.projector(pooled)  # [B, output_dim]
        
        return embeddings


class PatchImageSoil(SoilProcessor):
    """
    Processes images using patch embedding (ViT-style).
    Divides image into patches and embeds them.
    """
    
    def __init__(self, input_channels: int = 3, output_dim: int = 512,
                 image_size: int = 224, patch_size: int = 16):
        super().__init__(modality="image", output_dim=output_dim)
        
        self.input_channels = input_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding via convolution
        self.patch_embedding = nn.Conv2d(
            input_channels, output_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, output_dim)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=8,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process image tensor into embeddings using patches.
        
        Args:
            x: Image tensor [B, C, H, W]
            
        Returns:
            embeddings: [B, output_dim]
        """
        B = x.shape[0]
        
        # Patch embedding
        patches = self.patch_embedding(x)  # [B, output_dim, H/P, W/P]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, output_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, output_dim]
        x = torch.cat([cls_tokens, patches], dim=1)  # [B, num_patches+1, output_dim]
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        # Transformer encoding
        x = self.transformer(x)  # [B, num_patches+1, output_dim]
        
        # Return CLS token embedding
        embeddings = x[:, 0]  # [B, output_dim]
        
        return embeddings
