"""
Image-Text cross-modal tasks for NeuralForest.
"""

import torch
import torch.nn as nn
from ..base import TaskHead, TaskRegistry


class ImageTextMatching(TaskHead):
    """
    Image-text matching task.
    Learns alignment between images and text descriptions.
    """
    
    SUPPORTED_DATASETS = ["coco_captions", "flickr30k", "conceptual_captions"]
    
    def __init__(self, input_dim: int,
                 task_name: str = "image_text_matching"):
        """
        Initialize image-text matching head.
        
        Args:
            input_dim: Input feature dimension
            task_name: Task identifier
        """
        super().__init__(input_dim, task_name)
        
        # Contrastive learning head
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to common space."""
        return self.projection(x)
    
    def get_loss(self, image_features: torch.Tensor, 
                 text_features: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
        """
        Compute contrastive loss between image and text features.
        
        Args:
            image_features: Image feature embeddings
            text_features: Text feature embeddings
            temperature: Temperature for softmax
            
        Returns:
            Contrastive loss
        """
        # Normalize features
        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) / temperature
        
        # Targets are diagonal (matching pairs)
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        
        # Bidirectional loss
        loss_i2t = nn.functional.cross_entropy(logits, labels)
        loss_t2i = nn.functional.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2


class ImageCaptioning(TaskHead):
    """
    Image captioning task.
    Generates textual descriptions of images.
    """
    
    def __init__(self, input_dim: int, vocab_size: int,
                 max_length: int = 50,
                 task_name: str = "image_captioning"):
        """
        Initialize image captioning head.
        
        Args:
            input_dim: Input feature dimension
            vocab_size: Vocabulary size
            max_length: Maximum caption length
            task_name: Task identifier
        """
        super().__init__(input_dim, task_name)
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        self.caption_generator = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, max_length * vocab_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate caption from image features.
        
        Args:
            x: Image features [batch_size, input_dim]
            
        Returns:
            Caption logits [batch_size, max_length, vocab_size]
        """
        batch_size = x.size(0)
        output = self.caption_generator(x)
        return output.view(batch_size, self.max_length, self.vocab_size)
    
    def get_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute captioning loss."""
        batch_size, seq_len, vocab_size = predictions.shape
        predictions_flat = predictions.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        return nn.functional.cross_entropy(predictions_flat, targets_flat, ignore_index=-1)


TaskRegistry.register("image_text_matching", ImageTextMatching)
TaskRegistry.register("image_captioning", ImageCaptioning)
