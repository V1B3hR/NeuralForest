"""
Object Detection task for NeuralForest.
"""

import torch
import torch.nn as nn
from ..base import TaskHead, TaskRegistry


class ObjectDetection(TaskHead):
    """
    Object detection task head.
    
    Simplified detection head that outputs:
    - Bounding box coordinates (x, y, w, h)
    - Class probabilities
    - Confidence score
    """
    
    SUPPORTED_DATASETS = [
        "coco", "pascal_voc", "open_images", "objects365"
    ]
    
    def __init__(self, input_dim: int, num_classes: int,
                 task_name: str = "object_detection"):
        """
        Initialize object detection head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of object classes
            task_name: Task identifier
        """
        super().__init__(input_dim, task_name)
        self.num_classes = num_classes
        
        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 4)  # x, y, w, h
        )
        
        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_classes)
        )
        
        # Confidence head
        self.conf_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass for detection.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with bbox, class_logits, and confidence
        """
        bbox = self.bbox_head(x)
        class_logits = self.class_head(x)
        confidence = self.conf_head(x)
        
        return {
            'bbox': bbox,
            'class_logits': class_logits,
            'confidence': confidence
        }
    
    def get_loss(self, predictions: dict, targets: dict) -> torch.Tensor:
        """
        Compute detection loss.
        
        Args:
            predictions: Dict with bbox, class_logits, confidence
            targets: Dict with target_bbox, target_classes
            
        Returns:
            Combined detection loss
        """
        # Bbox regression loss (L1)
        bbox_loss = nn.functional.l1_loss(
            predictions['bbox'], 
            targets['target_bbox']
        )
        
        # Classification loss
        class_loss = nn.functional.cross_entropy(
            predictions['class_logits'],
            targets['target_classes']
        )
        
        # Confidence loss (binary cross entropy)
        # Confidence should be high for correct predictions
        conf_targets = (predictions['class_logits'].argmax(dim=-1) == targets['target_classes']).float().unsqueeze(1)
        conf_loss = nn.functional.binary_cross_entropy(
            predictions['confidence'],
            conf_targets
        )
        
        # Weighted combination
        return bbox_loss + class_loss + 0.5 * conf_loss


TaskRegistry.register("object_detection", ObjectDetection)
