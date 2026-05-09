"""
Forest Litter: Pasywny, zdecentralizowany transfer wiedzy przez ściółkę cech.

Drzewa "opadają" reprezentacjami do PrioritizedMulch (ściółka).
Młode drzewa wchłaniają je pasywnie podczas treningu — bez jawnych połączeń.

KnowledgeTransfer: statyczne narzędzia matematyczne dla transferu wiedzy.
  - distillation_loss: KL-divergence distillation (Hinton et al.)
  - feature_alignment_loss: cosine similarity z marginesem
  - gradient_sharing: blending gradientów między drzewami
  - progressive_knowledge_transfer: transfer od wielu nauczycieli
  - litter_absorption_loss: wchłanianie ściółki z PrioritizedMulch
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List


class KnowledgeTransfer:
    """
    Utilities for knowledge transfer between trees.
    Supports various transfer strategies.
    """

    @staticmethod
    def distillation_loss(
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        temperature: float = 2.0,
    ) -> torch.Tensor:
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

        loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (
            temperature**2
        )

        return loss

    @staticmethod
    def feature_alignment_loss(
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        margin: float = 0.5,
    ) -> torch.Tensor:
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
            if hasattr(source_tree, "trunk") and hasattr(target_tree, "trunk"):
                for src_param, tgt_param in zip(
                    source_tree.trunk.parameters(), target_tree.trunk.parameters()
                ):
                    if src_param.grad is not None and tgt_param.grad is not None:
                        # Blend gradients
                        tgt_param.grad = (
                            1 - share_ratio
                        ) * tgt_param.grad + share_ratio * src_param.grad

    @staticmethod
    def progressive_knowledge_transfer(
        teacher_trees: List,
        student_tree,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
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
                if hasattr(teacher, "get_features"):
                    features = teacher.get_features(x)
                else:
                    features = teacher.trunk(teacher.act(x))
                teacher_features.append(features)

        # Student features
        student_features = (
            student_tree.get_features(x)
            if hasattr(student_tree, "get_features")
            else student_tree.trunk(student_tree.act(x))
        )

        # Weighted alignment loss
        for teacher_feat, weight in zip(teacher_features, weights):
            loss = F.mse_loss(student_features, teacher_feat.detach())
            total_loss += weight * loss

        return total_loss

    @staticmethod
    def litter_absorption_loss(
        student_features: torch.Tensor,
        mulch,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Wchłanianie ściółki — student uczy się od zakumulowanych reprezentacji w mulch.
        Fasada nad PrioritizedMulch.sample_features().
        """
        if not hasattr(mulch, "sample_features"):
            return torch.tensor(0.0)

        litter = mulch.sample_features(batch_size)
        if litter is None:
            return torch.tensor(0.0)

        litter = litter.to(student_features.device)
        min_len = min(len(student_features), len(litter))
        return F.mse_loss(student_features[:min_len], litter[:min_len].detach())
