"""
Phase 2: Forest Ecosystem Simulation (roadmap2.md)

Implements:
- Competition for resources (data batches)
- Robustness tests (drought, flood scenarios)
- Statistics logging for ecosystem evolution
- Fitness-based selection and pruning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import time

import torch
import numpy as np


@dataclass
class EcosystemStats:
    """Statistics tracker for the forest ecosystem."""
    
    generation: int = 0
    timestamp: float = 0.0
    
    # Tree population stats
    num_trees: int = 0
    avg_fitness: float = 0.0
    max_fitness: float = 0.0
    min_fitness: float = 0.0
    fitness_std: float = 0.0
    
    # Architecture diversity
    unique_architectures: int = 0
    avg_tree_age: float = 0.0
    
    # Resource allocation
    total_data_allocated: int = 0
    competition_events: int = 0
    
    # Selection pressure
    trees_pruned: int = 0
    trees_planted: int = 0
    selection_rate: float = 0.0
    
    # Robustness
    disruption_type: Optional[str] = None
    disruption_severity: float = 0.0
    survival_rate: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'generation': self.generation,
            'timestamp': self.timestamp,
            'num_trees': self.num_trees,
            'avg_fitness': self.avg_fitness,
            'max_fitness': self.max_fitness,
            'min_fitness': self.min_fitness,
            'fitness_std': self.fitness_std,
            'unique_architectures': self.unique_architectures,
            'avg_tree_age': self.avg_tree_age,
            'total_data_allocated': self.total_data_allocated,
            'competition_events': self.competition_events,
            'trees_pruned': self.trees_pruned,
            'trees_planted': self.trees_planted,
            'selection_rate': self.selection_rate,
            'disruption_type': self.disruption_type,
            'disruption_severity': self.disruption_severity,
            'survival_rate': self.survival_rate,
        }


class CompetitionSystem:
    """
    Manages competition for resources (data batches) among trees.
    
    Trees with higher fitness get better access to quality data.
    """
    
    def __init__(self, fairness_factor: float = 0.3):
        """
        Args:
            fairness_factor: How much to balance between fitness-based and equal allocation.
                            0.0 = pure fitness-based, 1.0 = pure equal distribution
        """
        self.fairness_factor = fairness_factor
        self.allocation_history = deque(maxlen=100)
    
    def allocate_data(self, forest, batch_x, batch_y) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Allocate data batches to trees based on fitness competition.
        
        Args:
            forest: ForestEcosystem instance
            batch_x: Input batch [B, D]
            batch_y: Target batch [B, 1]
            
        Returns:
            Dict mapping tree_id to (data_x, data_y) allocation
        """
        num_trees = forest.num_trees()
        if num_trees == 0:
            return {}
        
        batch_size = batch_x.shape[0]
        
        # Calculate fitness-based weights
        fitnesses = torch.tensor([t.fitness for t in forest.trees], dtype=torch.float32)
        
        # Add small epsilon to avoid division by zero
        fitnesses = fitnesses + 1e-6
        
        # Normalize fitnesses to probabilities
        fitness_weights = fitnesses / fitnesses.sum()
        
        # Mix with uniform distribution for fairness
        uniform_weights = torch.ones(num_trees) / num_trees
        allocation_weights = (1 - self.fairness_factor) * fitness_weights + self.fairness_factor * uniform_weights
        
        # Allocate samples to trees
        samples_per_tree = (allocation_weights * batch_size).int()
        
        # Ensure at least 1 sample per tree and sum equals batch_size
        samples_per_tree = torch.clamp(samples_per_tree, min=1)
        
        # Adjust to match exact batch size
        diff = batch_size - samples_per_tree.sum().item()
        if diff > 0:
            # Give extra samples to highest fitness trees
            top_indices = torch.argsort(fitnesses, descending=True)[:diff]
            for idx in top_indices:
                samples_per_tree[idx] += 1
        elif diff < 0:
            # Remove samples from lowest fitness trees
            low_indices = torch.argsort(fitnesses)[:abs(diff)]
            for idx in low_indices:
                if samples_per_tree[idx] > 1:
                    samples_per_tree[idx] -= 1
        
        # Create data allocations
        allocations = {}
        start_idx = 0
        
        for i, tree in enumerate(forest.trees):
            n_samples = samples_per_tree[i].item()
            if n_samples > 0:
                end_idx = start_idx + n_samples
                allocations[tree.id] = (
                    batch_x[start_idx:end_idx],
                    batch_y[start_idx:end_idx]
                )
                start_idx = end_idx
        
        self.allocation_history.append({
            'timestamp': time.time(),
            'allocations': {k: v[0].shape[0] for k, v in allocations.items()},
        })
        
        return allocations
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get statistics about resource allocation."""
        if not self.allocation_history:
            return {}
        
        recent = list(self.allocation_history)[-10:]
        
        return {
            'total_allocations': len(self.allocation_history),
            'recent_competitions': len(recent),
            'avg_trees_per_allocation': np.mean([len(a['allocations']) for a in recent]),
        }


class RobustnessTester:
    """
    Tests forest robustness by introducing environmental disruptions.
    
    Simulates "drought" (data scarcity) and "flood" (data overload/noise).
    """
    
    @staticmethod
    def apply_drought(batch_x, batch_y, severity: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate drought by reducing available data.
        
        Args:
            batch_x: Input batch [B, D]
            batch_y: Target batch [B, 1]
            severity: Fraction of data to remove (0.0 = no drought, 1.0 = total drought)
            
        Returns:
            Reduced batch
        """
        severity = np.clip(severity, 0.0, 0.95)  # Never remove all data
        keep_ratio = 1.0 - severity
        
        batch_size = batch_x.shape[0]
        keep_size = max(1, int(batch_size * keep_ratio))
        
        # Randomly select samples to keep
        indices = torch.randperm(batch_size)[:keep_size]
        
        return batch_x[indices], batch_y[indices]
    
    @staticmethod
    def apply_flood(batch_x, batch_y, severity: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate flood by adding noise and corrupting data.
        
        Args:
            batch_x: Input batch [B, D]
            batch_y: Target batch [B, 1]
            severity: Amount of noise to add (0.0 = no noise, 1.0 = high noise)
            
        Returns:
            Corrupted batch
        """
        severity = np.clip(severity, 0.0, 1.0)
        
        # Add Gaussian noise to inputs
        noise_x = torch.randn_like(batch_x) * severity
        corrupted_x = batch_x + noise_x
        
        # Add noise to targets
        noise_y = torch.randn_like(batch_y) * severity * 0.5  # Less noise on targets
        corrupted_y = batch_y + noise_y
        
        return corrupted_x, corrupted_y
    
    @staticmethod
    def apply_disruption(batch_x, batch_y, disruption_type: str = "drought", 
                         severity: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply a disruption to the data.
        
        Args:
            batch_x: Input batch
            batch_y: Target batch
            disruption_type: 'drought' or 'flood'
            severity: Disruption severity (0.0 to 1.0)
            
        Returns:
            Disrupted batch
        """
        if disruption_type == "drought":
            return RobustnessTester.apply_drought(batch_x, batch_y, severity)
        elif disruption_type == "flood":
            return RobustnessTester.apply_flood(batch_x, batch_y, severity)
        else:
            # No disruption
            return batch_x, batch_y


class EcosystemSimulator:
    """
    Main simulator for Phase 2: Forest Ecosystem.
    
    Orchestrates competition, selection, and robustness testing.
    """
    
    def __init__(self, forest, 
                 competition_fairness: float = 0.3,
                 selection_threshold: float = 0.3,
                 max_history: int = 1000):
        """
        Args:
            forest: ForestEcosystem instance
            competition_fairness: Fairness in resource allocation (0=fitness-based, 1=equal)
            selection_threshold: Fitness percentile below which trees are pruned
            max_history: Maximum number of stats to keep in history
        """
        self.forest = forest
        self.competition = CompetitionSystem(fairness_factor=competition_fairness)
        self.robustness = RobustnessTester()
        
        self.selection_threshold = selection_threshold
        self.generation = 0
        
        self.stats_history = deque(maxlen=max_history)
        self.current_stats = EcosystemStats()
    
    def simulate_generation(self, batch_x, batch_y, 
                           disruption_type: Optional[str] = None,
                           disruption_severity: float = 0.0) -> EcosystemStats:
        """
        Simulate one generation of the ecosystem.
        
        Args:
            batch_x: Input data batch
            batch_y: Target data batch
            disruption_type: Optional disruption ('drought' or 'flood')
            disruption_severity: Severity of disruption (0.0 to 1.0)
            
        Returns:
            Statistics for this generation
        """
        # Apply disruption if specified
        if disruption_type:
            batch_x, batch_y = self.robustness.apply_disruption(
                batch_x, batch_y, disruption_type, disruption_severity
            )
        
        # Allocate resources through competition
        allocations = self.competition.allocate_data(self.forest, batch_x, batch_y)
        
        # Update statistics
        self.generation += 1
        stats = self._compute_stats(allocations, disruption_type, disruption_severity)
        
        self.stats_history.append(stats)
        self.current_stats = stats
        
        return stats
    
    def _compute_stats(self, allocations: Dict, 
                       disruption_type: Optional[str],
                       disruption_severity: float) -> EcosystemStats:
        """Compute ecosystem statistics."""
        stats = EcosystemStats()
        
        stats.generation = self.generation
        stats.timestamp = time.time()
        
        # Tree population
        stats.num_trees = self.forest.num_trees()
        
        if stats.num_trees > 0:
            fitnesses = [t.fitness for t in self.forest.trees]
            stats.avg_fitness = np.mean(fitnesses)
            stats.max_fitness = np.max(fitnesses)
            stats.min_fitness = np.min(fitnesses)
            stats.fitness_std = np.std(fitnesses)
            
            # Architecture diversity (count unique architectures)
            arch_signatures = set()
            ages = []
            for t in self.forest.trees:
                arch_sig = (
                    t.arch.num_layers,
                    t.arch.hidden_dim,
                    t.arch.activation,
                    t.arch.dropout,
                )
                arch_signatures.add(arch_sig)
                ages.append(t.age)
            
            stats.unique_architectures = len(arch_signatures)
            stats.avg_tree_age = np.mean(ages)
        
        # Resource allocation
        stats.total_data_allocated = sum(v[0].shape[0] for v in allocations.values())
        stats.competition_events = len(self.competition.allocation_history)
        
        # Disruption info
        stats.disruption_type = disruption_type
        stats.disruption_severity = disruption_severity
        
        return stats
    
    def selection_pressure(self, min_keep: int = 2) -> Tuple[List[int], float]:
        """
        Apply selection pressure: identify weak trees for pruning.
        
        Args:
            min_keep: Minimum number of trees to keep
            
        Returns:
            (ids_to_remove, selection_rate)
        """
        num_trees = self.forest.num_trees()
        
        if num_trees <= min_keep:
            return [], 0.0
        
        # Sort trees by fitness
        sorted_trees = sorted(self.forest.trees, key=lambda t: t.fitness)
        
        # Calculate threshold
        threshold_idx = max(0, int(num_trees * self.selection_threshold))
        threshold_idx = min(threshold_idx, num_trees - min_keep)
        
        # Select weak trees for removal
        to_remove = [t.id for t in sorted_trees[:threshold_idx]]
        
        selection_rate = len(to_remove) / num_trees if num_trees > 0 else 0.0
        
        return to_remove, selection_rate
    
    def prune_weak_trees(self, min_keep: int = 2) -> int:
        """
        Prune weak trees based on fitness.
        
        Args:
            min_keep: Minimum trees to keep
            
        Returns:
            Number of trees pruned
        """
        to_remove, selection_rate = self.selection_pressure(min_keep)
        
        if to_remove:
            before = self.forest.num_trees()
            self.forest._prune_trees(to_remove, min_keep=min_keep)
            after = self.forest.num_trees()
            
            pruned = before - after
            
            # Update stats
            self.current_stats.trees_pruned = pruned
            self.current_stats.selection_rate = selection_rate
            
            return pruned
        
        return 0
    
    def plant_trees(self, count: int = 1) -> int:
        """
        Plant new trees in the forest.
        
        Args:
            count: Number of trees to plant
            
        Returns:
            Number of trees actually planted
        """
        before = self.forest.num_trees()
        
        for _ in range(count):
            if self.forest.num_trees() >= self.forest.max_trees:
                break
            self.forest._plant_tree()
        
        after = self.forest.num_trees()
        planted = after - before
        
        # Update stats
        self.current_stats.trees_planted = planted
        
        return planted
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive ecosystem summary."""
        if not self.stats_history:
            return {}
        
        recent_stats = list(self.stats_history)[-10:]
        
        summary = {
            'current_generation': self.generation,
            'total_trees': self.forest.num_trees(),
            'current_fitness': {
                'avg': self.current_stats.avg_fitness,
                'max': self.current_stats.max_fitness,
                'min': self.current_stats.min_fitness,
                'std': self.current_stats.fitness_std,
            },
            'architecture_diversity': self.current_stats.unique_architectures,
            'avg_tree_age': self.current_stats.avg_tree_age,
            'recent_pruned': sum(s.trees_pruned for s in recent_stats),
            'recent_planted': sum(s.trees_planted for s in recent_stats),
            'competition_stats': self.competition.get_allocation_stats(),
        }
        
        return summary
    
    def get_stats_history(self) -> List[Dict[str, Any]]:
        """Get full statistics history."""
        return [stats.to_dict() for stats in self.stats_history]
