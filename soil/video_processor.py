"""
VideoSoil: Processes video as temporal sequence of image frames.
- Frame sampling
- Spatial features (per frame)
- Temporal modeling (across frames)
"""

import torch
import torch.nn as nn
from .base import SoilProcessor


class VideoSoil(SoilProcessor):
    """
    Processes video (sequence of frames) into embeddings.
    Uses 3D convolutions to capture spatio-temporal features.
    """

    def __init__(
        self, input_channels: int = 3, output_dim: int = 512, num_frames: int = 16
    ):
        super().__init__(modality="video", output_dim=output_dim)

        self.input_channels = input_channels
        self.num_frames = num_frames

        # 3D convolutions for spatio-temporal feature extraction
        self.conv3d_layers = nn.Sequential(
            # Input: [B, C, T, H, W]
            nn.Conv3d(
                input_channels,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(
                64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

        # Global spatio-temporal pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.projector = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process video tensor into embeddings.

        Args:
            x: Video tensor [B, C, T, H, W] where T is temporal (frames)

        Returns:
            embeddings: [B, output_dim]
        """
        # 3D convolutional feature extraction
        features = self.conv3d_layers(x)  # [B, 512, T', H', W']

        # Global pooling over spatial and temporal dimensions
        pooled = self.global_pool(features)  # [B, 512, 1, 1, 1]
        pooled = pooled.flatten(1)  # [B, 512]

        # Project to output dimension
        embeddings = self.projector(pooled)  # [B, output_dim]

        return embeddings


class Frame2DVideoSoil(SoilProcessor):
    """
    Processes video by extracting 2D features per frame and aggregating temporally.
    More memory-efficient than full 3D convolutions.
    """

    def __init__(
        self, input_channels: int = 3, output_dim: int = 512, num_frames: int = 16
    ):
        super().__init__(modality="video", output_dim=output_dim)

        self.input_channels = input_channels
        self.num_frames = num_frames

        # 2D CNN for per-frame feature extraction
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Spatial pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Temporal aggregation via LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=512,
            hidden_size=output_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.projector = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process video tensor into embeddings.

        Args:
            x: Video tensor [B, C, T, H, W]

        Returns:
            embeddings: [B, output_dim]
        """
        B, C, T, H, W = x.shape

        # Reshape to process all frames at once: [B*T, C, H, W]
        x_frames = x.transpose(1, 2).contiguous().view(B * T, C, H, W)

        # Extract per-frame features
        frame_features = self.frame_encoder(x_frames)  # [B*T, 512, H', W']

        # Spatial pooling
        frame_features = self.spatial_pool(frame_features)  # [B*T, 512, 1, 1]
        frame_features = frame_features.flatten(1)  # [B*T, 512]

        # Reshape back to [B, T, 512]
        frame_features = frame_features.view(B, T, 512)

        # Temporal aggregation with LSTM
        lstm_out, (hidden, cell) = self.temporal_lstm(
            frame_features
        )  # lstm_out: [B, T, output_dim]

        # Use final hidden states
        forward_hidden = hidden[-2]  # [B, output_dim//2]
        backward_hidden = hidden[-1]  # [B, output_dim//2]
        combined = torch.cat(
            [forward_hidden, backward_hidden], dim=1
        )  # [B, output_dim]

        # Final projection
        embeddings = self.projector(combined)  # [B, output_dim]

        return embeddings
