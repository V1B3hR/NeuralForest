"""
AudioSoil: Processes audio using spectrogram + temporal modeling.
- Mel-spectrogram conversion
- Temporal convolutions
- Frequency-aware embeddings
"""

import torch
import torch.nn as nn
from .base import SoilProcessor


class AudioSoil(SoilProcessor):
    """
    Processes audio waveforms or spectrograms into embeddings.
    Uses temporal convolutions for audio feature extraction.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_dim: int = 512,
        sample_rate: int = 16000,
        n_mels: int = 128,
    ):
        super().__init__(modality="audio", output_dim=output_dim)

        self.input_channels = input_channels
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Temporal convolution layers (1D convolutions for audio)
        self.temporal_conv = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            # Layer 2
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            # Layer 3
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # Layer 4
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # Global temporal pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process audio tensor into embeddings.

        Args:
            x: Audio tensor [B, C, T] where T is time dimension

        Returns:
            embeddings: [B, output_dim]
        """
        # Temporal feature extraction
        features = self.temporal_conv(x)  # [B, 512, T']

        # Global temporal pooling
        pooled = self.global_pool(features)  # [B, 512, 1]
        pooled = pooled.squeeze(-1)  # [B, 512]

        # Project to output dimension
        embeddings = self.projector(pooled)  # [B, output_dim]

        return embeddings


class SpectrogramAudioSoil(SoilProcessor):
    """
    Processes audio spectrograms (2D) into embeddings.
    Treats spectrograms like images with frequency and time dimensions.
    """

    def __init__(self, n_mels: int = 128, output_dim: int = 512):
        super().__init__(modality="audio", output_dim=output_dim)

        self.n_mels = n_mels

        # 2D convolutions for spectrogram processing
        self.conv_layers = nn.Sequential(
            # Input: [B, 1, n_mels, T]
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process spectrogram tensor into embeddings.

        Args:
            x: Spectrogram tensor [B, 1, n_mels, T] or [B, n_mels, T]

        Returns:
            embeddings: [B, output_dim]
        """
        # Ensure 4D input [B, C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        # Convolutional feature extraction
        features = self.conv_layers(x)  # [B, 512, H', W']

        # Global pooling
        pooled = self.global_pool(features)  # [B, 512, 1, 1]
        pooled = pooled.flatten(1)  # [B, 512]

        # Project to output dimension
        embeddings = self.projector(pooled)  # [B, output_dim]

        return embeddings
