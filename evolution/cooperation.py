"""
Tree Cooperation System for NeuralForest Phase 6.

Implements cooperation mechanisms between trees including:
- Communication channels between trees
- Information exchange protocols
- Federated learning across trees
- Collaborative knowledge sharing
- Transfer learning across tree species
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CommunicationMessage:
    """Message passed between trees."""
    
    sender_id: int
    receiver_id: int
    message_type: str  # 'gradient', 'feature', 'knowledge', 'alert'
    content: Any
    timestamp: float
    priority: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CommunicationChannel:
    """
    Communication channel for information exchange between trees.
    
    Features:
    - Message passing between trees
    - Priority-based message queuing
    - Broadcast and unicast messaging
    - Message history tracking
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize communication channel.
        
        Args:
            max_history: Maximum number of messages to keep in history
        """
        self.messages: Dict[int, List[CommunicationMessage]] = defaultdict(list)
        self.message_history = []
        self.max_history = max_history
        self.stats = {
            'total_messages': 0,
            'messages_by_type': defaultdict(int),
        }
    
    def send_message(
        self,
        sender_id: int,
        receiver_id: int,
        message_type: str,
        content: Any,
        priority: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send a message from one tree to another.
        
        Args:
            sender_id: ID of sending tree
            receiver_id: ID of receiving tree
            message_type: Type of message
            content: Message content
            priority: Message priority (higher = more important)
            metadata: Optional metadata
        """
        import time
        
        message = CommunicationMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            priority=priority,
            metadata=metadata or {}
        )
        
        # Add to receiver's queue
        self.messages[receiver_id].append(message)
        
        # Sort by priority (descending)
        self.messages[receiver_id].sort(key=lambda m: m.priority, reverse=True)
        
        # Add to history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        # Update stats
        self.stats['total_messages'] += 1
        self.stats['messages_by_type'][message_type] += 1
    
    def broadcast_message(
        self,
        sender_id: int,
        receiver_ids: List[int],
        message_type: str,
        content: Any,
        priority: float = 1.0
    ) -> None:
        """
        Broadcast a message to multiple trees.
        
        Args:
            sender_id: ID of sending tree
            receiver_ids: IDs of receiving trees
            message_type: Type of message
            content: Message content
            priority: Message priority
        """
        for receiver_id in receiver_ids:
            self.send_message(sender_id, receiver_id, message_type, content, priority)
    
    def receive_messages(
        self,
        receiver_id: int,
        message_type: Optional[str] = None,
        max_messages: Optional[int] = None
    ) -> List[CommunicationMessage]:
        """
        Receive messages for a tree.
        
        Args:
            receiver_id: ID of receiving tree
            message_type: Optional filter by message type
            max_messages: Maximum number of messages to retrieve
            
        Returns:
            List of messages
        """
        messages = self.messages.get(receiver_id, [])
        
        # Filter by type if specified
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        # Limit number of messages
        if max_messages:
            messages = messages[:max_messages]
        
        # Remove retrieved messages from queue
        if message_type:
            self.messages[receiver_id] = [
                m for m in self.messages[receiver_id] 
                if m.message_type != message_type or m not in messages
            ]
        else:
            if max_messages:
                self.messages[receiver_id] = self.messages[receiver_id][max_messages:]
            else:
                self.messages[receiver_id] = []
        
        return messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            'total_messages': self.stats['total_messages'],
            'messages_by_type': dict(self.stats['messages_by_type']),
            'pending_messages': sum(len(msgs) for msgs in self.messages.values()),
        }


class FederatedLearning:
    """
    Federated learning system for collaborative training across trees.
    
    Features:
    - Gradient aggregation from multiple trees
    - Parameter averaging with weighted contributions
    - Knowledge distillation across trees
    - Privacy-preserving learning
    """
    
    def __init__(
        self,
        aggregation_method: str = 'weighted_average',
        min_participants: int = 2
    ):
        """
        Initialize federated learning system.
        
        Args:
            aggregation_method: Method for aggregating updates ('weighted_average', 'median')
            min_participants: Minimum number of trees required for aggregation
        """
        self.aggregation_method = aggregation_method
        self.min_participants = min_participants
        self.round_history = []
    
    def aggregate_gradients(
        self,
        gradients: Dict[int, Dict[str, torch.Tensor]],
        weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients from multiple trees.
        
        Args:
            gradients: Dictionary mapping tree_id to gradient dictionary
            weights: Optional weights for each tree (e.g., based on fitness)
            
        Returns:
            Aggregated gradients
        """
        if len(gradients) < self.min_participants:
            logger.warning(f"Not enough participants ({len(gradients)} < {self.min_participants})")
            return {}
        
        # Default to equal weights
        if weights is None:
            weights = {tree_id: 1.0 / len(gradients) for tree_id in gradients.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Aggregate gradients
        aggregated = {}
        
        # Get common parameter names
        param_names = set(gradients[list(gradients.keys())[0]].keys())
        
        for param_name in param_names:
            if self.aggregation_method == 'weighted_average':
                # Weighted average
                aggregated[param_name] = sum(
                    gradients[tree_id][param_name] * weights[tree_id]
                    for tree_id in gradients.keys()
                    if param_name in gradients[tree_id]
                )
            elif self.aggregation_method == 'median':
                # Median aggregation (more robust to outliers)
                stacked = torch.stack([
                    gradients[tree_id][param_name] 
                    for tree_id in gradients.keys()
                    if param_name in gradients[tree_id]
                ])
                aggregated[param_name] = torch.median(stacked, dim=0)[0]
        
        return aggregated
    
    def aggregate_parameters(
        self,
        parameters: Dict[int, Dict[str, torch.Tensor]],
        weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model parameters from multiple trees.
        
        Args:
            parameters: Dictionary mapping tree_id to parameter dictionary
            weights: Optional weights for each tree
            
        Returns:
            Aggregated parameters
        """
        # Use same aggregation logic as gradients
        return self.aggregate_gradients(parameters, weights)
    
    def federated_averaging_round(
        self,
        tree_parameters: Dict[int, Dict[str, torch.Tensor]],
        tree_fitness: Dict[int, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one round of federated averaging.
        
        Args:
            tree_parameters: Parameters from each participating tree
            tree_fitness: Fitness scores for weighting
            
        Returns:
            Averaged parameters
        """
        # Weight by fitness
        weights = {
            tree_id: max(0.1, tree_fitness.get(tree_id, 1.0))
            for tree_id in tree_parameters.keys()
        }
        
        averaged = self.aggregate_parameters(tree_parameters, weights)
        
        # Record round
        self.round_history.append({
            'num_participants': len(tree_parameters),
            'avg_fitness': np.mean(list(tree_fitness.values())),
            'participants': list(tree_parameters.keys()),
        })
        
        return averaged


class TransferLearning:
    """
    Transfer learning system for knowledge transfer across tree species.
    
    Features:
    - Knowledge distillation from teacher to student trees
    - Feature-based transfer learning
    - Architecture adaptation
    - Progressive transfer with curriculum
    """
    
    def __init__(self):
        """Initialize transfer learning system."""
        self.transfer_history = []
        self.species_knowledge = defaultdict(dict)
    
    def distill_knowledge(
        self,
        teacher_outputs: torch.Tensor,
        student_outputs: torch.Tensor,
        temperature: float = 2.0,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Knowledge distillation loss.
        
        Args:
            teacher_outputs: Output logits from teacher tree
            student_outputs: Output logits from student tree
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss (1-alpha for hard targets)
            
        Returns:
            Distillation loss
        """
        # Soft targets from teacher
        soft_teacher = torch.nn.functional.softmax(teacher_outputs / temperature, dim=-1)
        soft_student = torch.nn.functional.log_softmax(student_outputs / temperature, dim=-1)
        
        # KL divergence loss
        distillation_loss = torch.nn.functional.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return distillation_loss
    
    def transfer_features(
        self,
        source_tree_id: int,
        target_tree_id: int,
        source_features: torch.Tensor,
        feature_type: str = 'intermediate'
    ) -> Dict[str, Any]:
        """
        Transfer learned features from source to target tree.
        
        Args:
            source_tree_id: ID of source tree
            target_tree_id: ID of target tree
            source_features: Features to transfer
            feature_type: Type of features ('intermediate', 'final', 'representation')
            
        Returns:
            Transfer result information
        """
        # Store features in species knowledge base
        if source_tree_id not in self.species_knowledge:
            self.species_knowledge[source_tree_id] = {}
        
        self.species_knowledge[source_tree_id][feature_type] = {
            'features': source_features.detach().clone(),
            'target_tree': target_tree_id,
        }
        
        # Record transfer
        transfer_record = {
            'source_tree': source_tree_id,
            'target_tree': target_tree_id,
            'feature_type': feature_type,
            'feature_shape': source_features.shape,
        }
        
        self.transfer_history.append(transfer_record)
        
        return transfer_record
    
    def cross_species_transfer(
        self,
        source_species: List[int],
        target_tree_id: int,
        transfer_method: str = 'ensemble'
    ) -> Dict[str, Any]:
        """
        Transfer knowledge from multiple source species to target tree.
        
        Args:
            source_species: List of source tree IDs
            target_tree_id: Target tree ID
            transfer_method: Method for combining knowledge ('ensemble', 'weighted', 'best')
            
        Returns:
            Transfer summary
        """
        summary = {
            'num_sources': len(source_species),
            'target_tree': target_tree_id,
            'method': transfer_method,
            'transferred_knowledge': [],
        }
        
        for source_id in source_species:
            if source_id in self.species_knowledge:
                for feature_type, data in self.species_knowledge[source_id].items():
                    summary['transferred_knowledge'].append({
                        'source': source_id,
                        'feature_type': feature_type,
                        'shape': data['features'].shape,
                    })
        
        return summary


class CooperationSystem:
    """
    Main cooperation system coordinating all cooperation mechanisms.
    
    Features:
    - Communication between trees
    - Federated learning
    - Transfer learning
    - Collaborative optimization
    """
    
    def __init__(self):
        """Initialize cooperation system."""
        self.communication = CommunicationChannel()
        self.federated_learning = FederatedLearning()
        self.transfer_learning = TransferLearning()
        self.collaboration_stats = defaultdict(int)
    
    def enable_tree_communication(
        self,
        tree_id: int,
        can_send: bool = True,
        can_receive: bool = True
    ) -> None:
        """
        Enable communication capabilities for a tree.
        
        Args:
            tree_id: Tree ID
            can_send: Whether tree can send messages
            can_receive: Whether tree can receive messages
        """
        self.collaboration_stats[f'tree_{tree_id}_communication_enabled'] = 1
        logger.info(f"Communication enabled for tree {tree_id} (send={can_send}, receive={can_receive})")
    
    def coordinate_learning(
        self,
        participating_trees: Dict[int, Any],
        coordination_type: str = 'federated'
    ) -> Dict[str, Any]:
        """
        Coordinate collaborative learning among trees.
        
        Args:
            participating_trees: Dictionary of tree_id to tree data
            coordination_type: Type of coordination ('federated', 'distillation', 'ensemble')
            
        Returns:
            Coordination results
        """
        results = {
            'coordination_type': coordination_type,
            'num_participants': len(participating_trees),
            'success': False,
        }
        
        if coordination_type == 'federated':
            # Federated learning round
            if len(participating_trees) >= self.federated_learning.min_participants:
                # Extract parameters and fitness
                tree_params = {
                    tree_id: data.get('parameters', {})
                    for tree_id, data in participating_trees.items()
                }
                tree_fitness = {
                    tree_id: data.get('fitness', 1.0)
                    for tree_id, data in participating_trees.items()
                }
                
                # Perform federated averaging
                averaged_params = self.federated_learning.federated_averaging_round(
                    tree_params, tree_fitness
                )
                
                results['averaged_parameters'] = averaged_params
                results['success'] = True
                self.collaboration_stats['federated_rounds'] += 1
        
        elif coordination_type == 'distillation':
            # Knowledge distillation
            # Find best tree as teacher
            best_tree_id = max(
                participating_trees.keys(),
                key=lambda tid: participating_trees[tid].get('fitness', 0)
            )
            
            results['teacher_tree'] = best_tree_id
            results['student_trees'] = [
                tid for tid in participating_trees.keys() if tid != best_tree_id
            ]
            results['success'] = True
            self.collaboration_stats['distillation_sessions'] += 1
        
        return results
    
    def get_cooperation_summary(self) -> Dict[str, Any]:
        """Get summary of cooperation activities."""
        return {
            'communication_stats': self.communication.get_stats(),
            'federated_rounds': len(self.federated_learning.round_history),
            'transfer_operations': len(self.transfer_learning.transfer_history),
            'collaboration_events': dict(self.collaboration_stats),
        }
