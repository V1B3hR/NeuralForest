"""
Modality Detector: Automatically detects input modality from raw data or features.
Uses learned signatures for each media type.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SignatureAnalyzer(nn.Module):
    """
    Base class for analyzing modality-specific signatures.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.analyzer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract signature features."""
        return self.analyzer(x)


class ImageSignatureAnalyzer(SignatureAnalyzer):
    """Analyzes image-specific signatures."""
    pass


class AudioSignatureAnalyzer(SignatureAnalyzer):
    """Analyzes audio-specific signatures."""
    pass


class TextSignatureAnalyzer(SignatureAnalyzer):
    """Analyzes text-specific signatures."""
    pass


class VideoSignatureAnalyzer(SignatureAnalyzer):
    """Analyzes video-specific signatures."""
    pass


class ModalityDetector(nn.Module):
    """
    Automatically detects input modality from features.
    Uses learned signatures for each media type.
    """
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Signature analyzers for each modality
        self.analyzers = nn.ModuleDict({
            "image": ImageSignatureAnalyzer(embedding_dim, hidden_dim),
            "audio": AudioSignatureAnalyzer(embedding_dim, hidden_dim),
            "text": TextSignatureAnalyzer(embedding_dim, hidden_dim),
            "video": VideoSignatureAnalyzer(embedding_dim, hidden_dim),
        })
        
        # Classifier to determine modality
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )
        
        self.modality_names = ["image", "audio", "text", "video"]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect modality from input features.
        
        Args:
            x: Input features [B, embedding_dim]
            
        Returns:
            Dictionary mapping modality names to probabilities
        """
        # Extract signatures from each analyzer
        signatures = []
        for name in self.modality_names:
            analyzer = self.analyzers[name]
            sig = analyzer(x)
            signatures.append(sig)
        
        # Concatenate all signatures
        combined = torch.cat(signatures, dim=-1)  # [B, hidden_dim * 4]
        
        # Classify modality
        logits = self.classifier(combined)  # [B, 4]
        probs = F.softmax(logits, dim=-1)
        
        # Return as dictionary
        result = {
            name: probs[:, i]
            for i, name in enumerate(self.modality_names)
        }
        
        return result
    
    def get_top_modality(self, x: torch.Tensor, threshold: float = 0.5) -> List[str]:
        """
        Get the most likely modality/modalities.
        
        Args:
            x: Input features [B, embedding_dim]
            threshold: Probability threshold for multi-modal detection
            
        Returns:
            List of modality names (can be multiple for multi-modal input)
        """
        probs_dict = self.forward(x)
        
        # Get probabilities as tensor
        probs = torch.stack([probs_dict[name] for name in self.modality_names], dim=-1)
        
        # For batch, return modalities for first sample
        probs_first = probs[0]
        
        # Get all modalities above threshold
        selected = []
        for i, name in enumerate(self.modality_names):
            if probs_first[i] > threshold:
                selected.append(name)
        
        # If nothing above threshold, return the top one
        if not selected:
            top_idx = torch.argmax(probs_first).item()
            selected = [self.modality_names[top_idx]]
        
        return selected
    
    def get_modality_confidence(self, x: torch.Tensor) -> Tuple[str, float]:
        """
        Get the top modality and its confidence score.
        
        Args:
            x: Input features [B, embedding_dim]
            
        Returns:
            Tuple of (modality_name, confidence)
        """
        probs_dict = self.forward(x)
        
        # Get probabilities as tensor
        probs = torch.stack([probs_dict[name] for name in self.modality_names], dim=-1)
        
        # For batch, use first sample
        probs_first = probs[0]
        
        top_idx = torch.argmax(probs_first).item()
        confidence = probs_first[top_idx].item()
        
        return self.modality_names[top_idx], confidence
