"""
Load Balancer: Ensures healthy forest by balancing load across trees.
Prevents overuse of popular experts and underuse of specialists.
"""

import torch
import torch.nn.functional as F
from collections import defaultdict, deque
from typing import Dict, List, Optional


class CanopyBalancer:
    """
    Ensures healthy forest by balancing load across trees.
    Prevents overuse of popular experts and underuse of specialists.
    """
    def __init__(self, target_utilization: float = 0.7, history_size: int = 1000):
        """
        Initialize the load balancer.
        
        Args:
            target_utilization: Target average utilization per tree (0.0 to 1.0)
            history_size: Number of recent routing decisions to track
        """
        self.target = target_utilization
        self.history_size = history_size
        self.usage_stats = defaultdict(lambda: deque(maxlen=history_size))
        self.total_calls = 0
    
    def record_usage(self, routing_weights: torch.Tensor, tree_ids: List[int]):
        """
        Record tree usage from routing decision.
        
        Args:
            routing_weights: Routing weights [B, num_trees]
            tree_ids: List of tree IDs corresponding to weight columns
        """
        self.total_calls += routing_weights.shape[0]
        
        # Average weights across batch
        avg_weights = routing_weights.mean(dim=0)
        
        # Track usage for each tree
        for tree_id, weight in zip(tree_ids, avg_weights.tolist()):
            self.usage_stats[tree_id].append(weight)
    
    def get_utilization(self, tree_id: int) -> float:
        """
        Get current utilization for a specific tree.
        
        Args:
            tree_id: ID of the tree
            
        Returns:
            Average utilization (0.0 to 1.0)
        """
        if tree_id not in self.usage_stats or not self.usage_stats[tree_id]:
            return 0.0
        
        stats = self.usage_stats[tree_id]
        return sum(stats) / len(stats)
    
    def get_all_utilizations(self) -> Dict[int, float]:
        """
        Get utilization for all tracked trees.
        
        Returns:
            Dictionary mapping tree_id to utilization
        """
        return {
            tree_id: self.get_utilization(tree_id)
            for tree_id in self.usage_stats.keys()
        }
    
    def compute_balance_loss(self, routing_weights: torch.Tensor, tree_ids: List[int]) -> torch.Tensor:
        """
        Auxiliary loss to encourage balanced expert usage.
        
        Args:
            routing_weights: Current routing weights [B, num_trees]
            tree_ids: List of tree IDs
            
        Returns:
            Balance loss encouraging uniform distribution
        """
        # Record this usage
        self.record_usage(routing_weights, tree_ids)
        
        # Get current utilizations
        utilizations = []
        for tree_id in tree_ids:
            util = self.get_utilization(tree_id)
            utilizations.append(util)
        
        if not utilizations:
            return torch.tensor(0.0, device=routing_weights.device)
        
        # Target: uniform distribution
        util_tensor = torch.tensor(utilizations, device=routing_weights.device)
        target_tensor = torch.ones_like(util_tensor) / len(util_tensor)
        
        # KL divergence to encourage balanced usage
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        util_normalized = util_tensor / (util_tensor.sum() + eps)
        
        balance_loss = F.kl_div(
            (util_normalized + eps).log(),
            target_tensor,
            reduction='batchmean'
        )
        
        return balance_loss
    
    def get_underutilized_trees(self, threshold: float = 0.3) -> List[int]:
        """
        Find trees that are underutilized.
        
        Args:
            threshold: Utilization threshold below which a tree is underutilized
            
        Returns:
            List of underutilized tree IDs
        """
        underutilized = []
        for tree_id, util in self.get_all_utilizations().items():
            if util < threshold:
                underutilized.append(tree_id)
        
        return underutilized
    
    def get_overutilized_trees(self, threshold: float = 0.8) -> List[int]:
        """
        Find trees that are overutilized.
        
        Args:
            threshold: Utilization threshold above which a tree is overutilized
            
        Returns:
            List of overutilized tree IDs
        """
        overutilized = []
        for tree_id, util in self.get_all_utilizations().items():
            if util > threshold:
                overutilized.append(tree_id)
        
        return overutilized
    
    def get_balance_stats(self) -> Dict:
        """
        Get statistics about load balance.
        
        Returns:
            Dictionary with balance statistics
        """
        utilizations = list(self.get_all_utilizations().values())
        
        if not utilizations:
            return {
                "num_trees": 0,
                "avg_utilization": 0.0,
                "min_utilization": 0.0,
                "max_utilization": 0.0,
                "std_utilization": 0.0,
                "balance_score": 1.0,  # Perfect balance when no trees
            }
        
        util_tensor = torch.tensor(utilizations)
        
        # Compute statistics
        avg_util = util_tensor.mean().item()
        min_util = util_tensor.min().item()
        max_util = util_tensor.max().item()
        std_util = util_tensor.std().item()
        
        # Balance score: 1.0 is perfect, 0.0 is very imbalanced
        # Based on coefficient of variation
        balance_score = 1.0 - min(std_util / (avg_util + 1e-8), 1.0)
        
        return {
            "num_trees": len(utilizations),
            "avg_utilization": avg_util,
            "min_utilization": min_util,
            "max_utilization": max_util,
            "std_utilization": std_util,
            "balance_score": balance_score,
            "target_utilization": self.target,
            "total_routing_calls": self.total_calls,
        }
    
    def suggest_actions(self) -> List[Dict]:
        """
        Suggest actions to improve load balance.
        
        Returns:
            List of suggested actions
        """
        suggestions = []
        
        # Check for underutilized trees
        underutilized = self.get_underutilized_trees(threshold=0.2)
        if underutilized:
            suggestions.append({
                "action": "boost_underutilized",
                "tree_ids": underutilized,
                "reason": f"{len(underutilized)} trees are underutilized"
            })
        
        # Check for overutilized trees
        overutilized = self.get_overutilized_trees(threshold=0.9)
        if overutilized:
            suggestions.append({
                "action": "relieve_overutilized",
                "tree_ids": overutilized,
                "reason": f"{len(overutilized)} trees are overutilized"
            })
        
        # Check overall balance
        stats = self.get_balance_stats()
        if stats["balance_score"] < 0.7:
            suggestions.append({
                "action": "improve_balance",
                "balance_score": stats["balance_score"],
                "reason": "Overall load balance is poor"
            })
        
        return suggestions
