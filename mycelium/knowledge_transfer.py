"""
Mycelium Network: Underground network connecting trees for knowledge transfer.
Inspired by real forest mycorrhizal networks.

Functions:
- Share useful gradients between related trees
- Transfer knowledge from mature to young trees
- Enable cross-grove communication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Optional, Dict, List, Tuple


class MyceliumNetwork(nn.Module):
    """
    Underground network connecting trees for knowledge transfer.
    Manages connections and knowledge flow between trees.
    """
    def __init__(self, num_groves: int = 4):
        super().__init__()
        self.num_groves = num_groves
        self.connections = defaultdict(list)
        self.transfer_strength = nn.ParameterDict()
    
    def connect(self, tree_a_id: int, tree_b_id: int, strength: float = 1.0):
        """
        Establish mycelium connection between trees.
        
        Args:
            tree_a_id: ID of first tree
            tree_b_id: ID of second tree
            strength: Connection strength (0.0 to 1.0)
        """
        key_ab = f"{tree_a_id}_{tree_b_id}"
        key_ba = f"{tree_b_id}_{tree_a_id}"
        
        # Bidirectional connection
        if tree_b_id not in self.connections[tree_a_id]:
            self.connections[tree_a_id].append(tree_b_id)
        if tree_a_id not in self.connections[tree_b_id]:
            self.connections[tree_b_id].append(tree_a_id)
        
        # Store connection strengths as learnable parameters
        self.transfer_strength[key_ab] = nn.Parameter(
            torch.tensor(strength, dtype=torch.float32)
        )
        self.transfer_strength[key_ba] = nn.Parameter(
            torch.tensor(strength, dtype=torch.float32)
        )
    
    def get_connections(self, tree_id: int) -> List[int]:
        """Get all trees connected to the given tree."""
        return self.connections.get(tree_id, [])
    
    def get_strength(self, source_id: int, target_id: int) -> float:
        """Get connection strength between two trees."""
        key = f"{source_id}_{target_id}"
        if key in self.transfer_strength:
            return self.transfer_strength[key].item()
        return 0.0
    
    def transfer_knowledge(self, source_tree, target_tree, x: torch.Tensor) -> torch.Tensor:
        """
        Soft knowledge transfer via feature alignment.
        Mature trees help young trees learn faster.
        
        Args:
            source_tree: Tree providing knowledge
            target_tree: Tree receiving knowledge
            x: Input data
            
        Returns:
            alignment_loss: Loss encouraging similar representations
        """
        # Get features from both trees
        with torch.no_grad():
            source_features = source_tree.get_features(x) if hasattr(source_tree, 'get_features') else source_tree.trunk(source_tree.act(x))
        
        target_features = target_tree.get_features(x) if hasattr(target_tree, 'get_features') else target_tree.trunk(target_tree.act(x))
        
        # Alignment loss encourages similar representations
        alignment_loss = F.mse_loss(target_features, source_features.detach())
        
        # Weight by connection strength and age difference
        age_factor = max(0, source_tree.age - target_tree.age) / 100.0
        age_factor = min(age_factor, 1.0)  # Cap at 1.0
        
        key = f"{source_tree.id}_{target_tree.id}"
        strength = self.transfer_strength.get(key, torch.tensor(1.0)).item()
        
        # Combined weighting
        weighted_loss = alignment_loss * strength * (0.5 + 0.5 * age_factor)
        
        return weighted_loss
    
    def get_network_stats(self) -> Dict:
        """Return statistics about the mycelium network."""
        total_connections = sum(len(conns) for conns in self.connections.values()) // 2
        
        return {
            "num_trees": len(self.connections),
            "total_connections": total_connections,
            "avg_connections_per_tree": total_connections * 2 / max(len(self.connections), 1),
            "connection_details": dict(self.connections),
        }


class KnowledgeTransfer:
    """
    Utilities for knowledge transfer between trees.
    Supports various transfer strategies.
    """
    
    @staticmethod
    def distillation_loss(teacher_output: torch.Tensor, 
                          student_output: torch.Tensor,
                          temperature: float = 2.0) -> torch.Tensor:
        """
        Knowledge distillation loss.
        
        Args:
            teacher_output: Logits from teacher tree
            student_output: Logits from student tree
            temperature: Temperature for softening distributions
            
        Returns:
            Distillation loss
        """
        teacher_soft = F.softmax(teacher_output / temperature, dim=-1)
        student_log_soft = F.log_softmax(student_output / temperature, dim=-1)
        
        loss = F.kl_div(
            student_log_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return loss
    
    @staticmethod
    def feature_alignment_loss(source_features: torch.Tensor,
                               target_features: torch.Tensor,
                               margin: float = 0.5) -> torch.Tensor:
        """
        Feature alignment loss with margin.
        
        Args:
            source_features: Features from source tree
            target_features: Features from target tree
            margin: Margin for contrastive loss
            
        Returns:
            Alignment loss
        """
        # Normalize features
        source_norm = F.normalize(source_features, dim=-1)
        target_norm = F.normalize(target_features, dim=-1)
        
        # Cosine similarity
        similarity = (source_norm * target_norm).sum(dim=-1)
        
        # Encourage similarity above margin
        loss = F.relu(margin - similarity).mean()
        
        return loss
    
    @staticmethod
    def gradient_sharing(source_tree, target_tree, share_ratio: float = 0.3):
        """
        Share gradients between connected trees.
        
        Args:
            source_tree: Source tree with gradients
            target_tree: Target tree to receive gradients
            share_ratio: Proportion of gradients to share (0.0 to 1.0)
        """
        with torch.no_grad():
            # Share trunk gradients
            if hasattr(source_tree, 'trunk') and hasattr(target_tree, 'trunk'):
                for src_param, tgt_param in zip(
                    source_tree.trunk.parameters(),
                    target_tree.trunk.parameters()
                ):
                    if src_param.grad is not None and tgt_param.grad is not None:
                        # Blend gradients
                        tgt_param.grad = (1 - share_ratio) * tgt_param.grad + share_ratio * src_param.grad
    
    @staticmethod
    def progressive_knowledge_transfer(
        teacher_trees: List,
        student_tree,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Progressive knowledge transfer from multiple teachers.
        
        Args:
            teacher_trees: List of teacher trees
            student_tree: Student tree to train
            x: Input data
            weights: Optional weights for each teacher (if None, uniform)
            
        Returns:
            Combined transfer loss
        """
        if not teacher_trees:
            return torch.tensor(0.0)
        
        if weights is None:
            weights = torch.ones(len(teacher_trees)) / len(teacher_trees)
        
        total_loss = 0.0
        
        with torch.no_grad():
            teacher_features = []
            for teacher in teacher_trees:
                if hasattr(teacher, 'get_features'):
                    features = teacher.get_features(x)
                else:
                    features = teacher.trunk(teacher.act(x))
                teacher_features.append(features)
        
        # Student features
        student_features = student_tree.get_features(x) if hasattr(student_tree, 'get_features') else student_tree.trunk(student_tree.act(x))
        
        # Weighted alignment loss
        for teacher_feat, weight in zip(teacher_features, weights):
            loss = F.mse_loss(student_features, teacher_feat.detach())
            total_loss += weight * loss
        
        return total_loss
