"""
Layer-wise Learning Rate Optimizer for NeuralForest.

Implements a complete optimizer system with:
- Layer-wise learning rates (early layers slow, late layers fast)
- Exponential age decay (older trees → lower LR)
- Fitness-aware LR adjustment (high fitness → lower LR)
- Per-layer warmup
- Cosine annealing schedule support
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class LayerWiseConfig:
    """Configuration for layer-wise optimizer."""
    
    # Base learning rates
    base_lr: float = 0.01
    min_lr: float = 0.0001
    
    # Age decay parameters
    half_life: float = 60.0  # Age decay half-life in epochs
    
    # Fitness parameters
    fitness_scale: float = 5.0  # Target fitness for scaling
    fitness_aware: bool = True
    
    # Layer-wise multipliers
    early_trunk_multiplier: float = 0.1  # Early layers learn slowly
    middle_trunk_multiplier: float = 0.5  # Middle layers moderate
    late_trunk_multiplier: float = 1.0   # Late layers learn fast
    tree_head_multiplier: float = 1.0    # Tree heads same as late trunk
    task_head_multiplier: float = 2.0    # Task head learns fastest
    
    # Warmup parameters
    warmup_epochs: int = 5
    warmup_early_multiplier: float = 0.1
    warmup_late_multiplier: float = 0.5
    
    # Schedule
    schedule: str = 'cosine'  # 'cosine', 'step', or 'none'
    total_epochs: int = 100
    
    # Optimizer parameters
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer_type: str = 'adam'  # 'adam' or 'sgd'


class ImprovedTreeAgeSystem:
    """
    Age and fitness-aware learning rate computation system.
    
    Implements:
    - Exponential age decay: age_factor = exp(-epoch_age / half_life)
    - Fitness-aware: fitness_factor = target_fitness / current_fitness
    - Combined LR adjustment
    """
    
    def __init__(self, config: LayerWiseConfig):
        self.config = config
        self.tree_ages = {}  # tree_id -> epoch_age
        self.tree_fitness = {}  # tree_id -> fitness_value
    
    def register_tree(self, tree_id: int, epoch_age: int = 0, fitness: float = None):
        """Register a tree with its age and fitness."""
        if fitness is None:
            fitness = self.config.fitness_scale
        self.tree_ages[tree_id] = epoch_age
        self.tree_fitness[tree_id] = fitness
    
    def update_tree_age(self, tree_id: int):
        """Increment tree age by 1 epoch."""
        if tree_id in self.tree_ages:
            self.tree_ages[tree_id] += 1
        else:
            self.tree_ages[tree_id] = 0
    
    def update_tree_fitness(self, tree_id: int, fitness: float):
        """Update tree fitness value."""
        self.tree_fitness[tree_id] = fitness
    
    def compute_age_factor(self, tree_id: int) -> float:
        """
        Compute exponential age decay factor.
        
        age_factor = exp(-epoch_age / half_life)
        
        Returns:
            Age factor in range (0, 1], older trees have lower values
        """
        epoch_age = self.tree_ages.get(tree_id, 0)
        age_factor = math.exp(-epoch_age / self.config.half_life)
        return age_factor
    
    def compute_fitness_factor(self, tree_id: int) -> float:
        """
        Compute fitness-aware adjustment factor.
        
        fitness_factor = target_fitness / current_fitness
        
        Returns:
            Fitness factor, high fitness trees have lower values (learn slower)
        """
        if not self.config.fitness_aware:
            return 1.0
        
        fitness = self.tree_fitness.get(tree_id, self.config.fitness_scale)
        # Prevent division by zero
        fitness = max(fitness, 0.1)
        
        # Higher fitness → lower learning rate (more refined)
        fitness_factor = self.config.fitness_scale / fitness
        
        # Clamp to reasonable range
        fitness_factor = max(0.2, min(fitness_factor, 5.0))
        
        return fitness_factor
    
    def compute_combined_factor(self, tree_id: int) -> float:
        """
        Compute combined age and fitness adjustment factor.
        
        Returns:
            Combined factor for LR adjustment
        """
        age_factor = self.compute_age_factor(tree_id)
        fitness_factor = self.compute_fitness_factor(tree_id)
        
        # Multiply factors (both reduce LR for old/high-fitness trees)
        combined = age_factor * fitness_factor
        
        return combined
    
    def get_tree_lr_scale(self, tree_id: int, base_scale: float = 1.0) -> float:
        """
        Get learning rate scale for a specific tree.
        
        Args:
            tree_id: Tree identifier
            base_scale: Base layer-wise multiplier
            
        Returns:
            Final LR scale combining base, age, and fitness factors
        """
        combined_factor = self.compute_combined_factor(tree_id)
        return base_scale * combined_factor


# Layer categorization thresholds
EARLY_LAYER_THRESHOLD = 0.33
LATE_LAYER_THRESHOLD = 0.67


class LayerWiseOptimizer:
    """
    Optimizer factory with layer-wise learning rates.
    
    Creates optimizers with different learning rates for:
    - Early trunk layers (slow learning)
    - Middle trunk layers (moderate learning)
    - Late trunk layers (fast learning)
    - Task head (fastest learning)
    
    Supports:
    - Age-based LR decay
    - Fitness-aware LR adjustment
    - Per-layer warmup
    - Cosine annealing scheduler
    """
    
    def __init__(self, config: LayerWiseConfig):
        self.config = config
        self.age_system = ImprovedTreeAgeSystem(config)
        self.current_epoch = 0
    
    def register_forest(self, forest):
        """Register all trees in the forest with age system."""
        for tree in forest.trees:
            epoch_age = getattr(tree, 'epoch_age', 0)
            fitness = getattr(tree, 'fitness', 5.0)
            self.age_system.register_tree(tree.id, epoch_age, fitness)
    
    def update_tree_ages(self, forest):
        """Update ages for all trees in the forest."""
        for tree in forest.trees:
            # Update internal age tracking
            if not hasattr(tree, 'epoch_age'):
                tree.epoch_age = 0
            tree.epoch_age += 1
            
            # Update age system
            self.age_system.update_tree_age(tree.id)
            
            # Update fitness
            fitness = getattr(tree, 'fitness', 5.0)
            self.age_system.update_tree_fitness(tree.id, fitness)
    
    def _get_warmup_factor(self, epoch: int) -> float:
        """
        Compute warmup factor for current epoch.
        
        Returns:
            Warmup multiplier in range [0.1, 1.0]
        """
        if epoch >= self.config.warmup_epochs:
            return 1.0
        
        # Linear warmup from 0.1 to 1.0
        progress = epoch / self.config.warmup_epochs
        return 0.1 + 0.9 * progress
    
    def _get_schedule_factor(self, epoch: int) -> float:
        """
        Compute learning rate schedule factor.
        
        Returns:
            Schedule multiplier based on epoch
        """
        if self.config.schedule == 'none':
            return 1.0
        
        # After warmup, apply schedule
        if epoch < self.config.warmup_epochs:
            return 1.0
        
        adjusted_epoch = epoch - self.config.warmup_epochs
        adjusted_total = self.config.total_epochs - self.config.warmup_epochs
        
        if self.config.schedule == 'cosine':
            # Cosine annealing
            factor = 0.5 * (1 + math.cos(math.pi * adjusted_epoch / adjusted_total))
            # Scale between min_lr and base_lr
            min_factor = self.config.min_lr / self.config.base_lr
            factor = min_factor + (1.0 - min_factor) * factor
            return factor
        
        elif self.config.schedule == 'step':
            # Step decay at 30%, 60%, 90% of training
            if adjusted_epoch >= 0.9 * adjusted_total:
                return 0.01
            elif adjusted_epoch >= 0.6 * adjusted_total:
                return 0.1
            elif adjusted_epoch >= 0.3 * adjusted_total:
                return 0.5
            return 1.0
        
        return 1.0
    
    def _categorize_tree_layers(self, tree) -> Dict[str, List[nn.Parameter]]:
        """
        Categorize tree layers into early, middle, late trunk layers.
        
        Returns:
            Dictionary mapping category to parameters
        """
        categories = {
            'early_trunk': [],
            'middle_trunk': [],
            'late_trunk': [],
            'head': []
        }
        
        # Get trunk layers
        if hasattr(tree, 'trunk') and isinstance(tree.trunk, nn.Sequential):
            trunk_layers = list(tree.trunk.children())
            num_layers = len(trunk_layers)
            
            for idx, layer in enumerate(trunk_layers):
                if hasattr(layer, 'weight'):
                    # Determine category based on position
                    position = idx / max(num_layers - 1, 1)
                    
                    if position < EARLY_LAYER_THRESHOLD:
                        categories['early_trunk'].extend([p for p in layer.parameters()])
                    elif position < LATE_LAYER_THRESHOLD:
                        categories['middle_trunk'].extend([p for p in layer.parameters()])
                    else:
                        categories['late_trunk'].extend([p for p in layer.parameters()])
        
        # Get head parameters
        if hasattr(tree, 'head'):
            categories['head'].extend(tree.head.parameters())
        
        # Skip projection (residual)
        if hasattr(tree, 'skip_proj') and tree.skip_proj is not None:
            categories['early_trunk'].extend(tree.skip_proj.parameters())
        
        return categories
    
    def create_param_groups(
        self, 
        forest, 
        task_head, 
        epoch: int
    ) -> List[Dict]:
        """
        Create parameter groups with layer-wise learning rates.
        
        Args:
            forest: ForestEcosystem instance
            task_head: Task head module
            epoch: Current epoch number
            
        Returns:
            List of parameter groups for optimizer
        """
        self.current_epoch = epoch
        param_groups = []
        
        # Update forest registration
        self.register_forest(forest)
        
        # Get warmup and schedule factors
        warmup_factor = self._get_warmup_factor(epoch)
        schedule_factor = self._get_schedule_factor(epoch)
        
        # Process each tree
        for tree in forest.trees:
            tree_id = tree.id
            
            # Get layer categories
            layer_categories = self._categorize_tree_layers(tree)
            
            # Early trunk layers
            if layer_categories['early_trunk']:
                base_scale = self.config.early_trunk_multiplier
                tree_scale = self.age_system.get_tree_lr_scale(tree_id, base_scale)
                lr = self.config.base_lr * tree_scale * warmup_factor * schedule_factor
                lr = max(lr, self.config.min_lr)
                
                param_groups.append({
                    'params': layer_categories['early_trunk'],
                    'lr': lr,
                    'name': f'tree_{tree_id}_early_trunk'
                })
            
            # Middle trunk layers
            if layer_categories['middle_trunk']:
                base_scale = self.config.middle_trunk_multiplier
                tree_scale = self.age_system.get_tree_lr_scale(tree_id, base_scale)
                lr = self.config.base_lr * tree_scale * warmup_factor * schedule_factor
                lr = max(lr, self.config.min_lr)
                
                param_groups.append({
                    'params': layer_categories['middle_trunk'],
                    'lr': lr,
                    'name': f'tree_{tree_id}_middle_trunk'
                })
            
            # Late trunk layers
            if layer_categories['late_trunk']:
                base_scale = self.config.late_trunk_multiplier
                tree_scale = self.age_system.get_tree_lr_scale(tree_id, base_scale)
                lr = self.config.base_lr * tree_scale * warmup_factor * schedule_factor
                lr = max(lr, self.config.min_lr)
                
                param_groups.append({
                    'params': layer_categories['late_trunk'],
                    'lr': lr,
                    'name': f'tree_{tree_id}_late_trunk'
                })
            
            # Tree head
            if layer_categories['head']:
                base_scale = self.config.tree_head_multiplier
                tree_scale = self.age_system.get_tree_lr_scale(tree_id, base_scale)
                lr = self.config.base_lr * tree_scale * warmup_factor * schedule_factor
                lr = max(lr, self.config.min_lr)
                
                param_groups.append({
                    'params': layer_categories['head'],
                    'lr': lr,
                    'name': f'tree_{tree_id}_head'
                })
        
        # Router parameters
        if hasattr(forest, 'router'):
            lr = self.config.base_lr * warmup_factor * schedule_factor
            lr = max(lr, self.config.min_lr)
            param_groups.append({
                'params': forest.router.parameters(),
                'lr': lr,
                'name': 'router'
            })
        
        # Task head parameters (learn fastest)
        if task_head is not None:
            lr = self.config.base_lr * self.config.task_head_multiplier * warmup_factor * schedule_factor
            lr = max(lr, self.config.min_lr)
            param_groups.append({
                'params': task_head.parameters(),
                'lr': lr,
                'name': 'task_head'
            })
        
        return param_groups
    
    def create_optimizer(
        self, 
        forest, 
        task_head, 
        epoch: int
    ) -> torch.optim.Optimizer:
        """
        Create optimizer with layer-wise learning rates.
        
        Args:
            forest: ForestEcosystem instance
            task_head: Task head module
            epoch: Current epoch number
            
        Returns:
            Configured optimizer
        """
        param_groups = self.create_param_groups(forest, task_head, epoch)
        
        if self.config.optimizer_type == 'adam':
            optimizer = optim.Adam(
                param_groups,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
        
        return optimizer
    
    def print_lr_summary(self, optimizer: torch.optim.Optimizer):
        """Print summary of learning rates for all parameter groups."""
        print("\n📊 Learning Rate Summary:")
        print("=" * 70)
        
        for i, group in enumerate(optimizer.param_groups):
            name = group.get('name', f'group_{i}')
            lr = group['lr']
            num_params = sum(p.numel() for p in group['params'])
            print(f"  {name:30s}: lr={lr:.6f}, params={num_params:,}")
        
        print("=" * 70)
    
    def log_tree_lr_factors(self, forest):
        """Log age and fitness factors for all trees."""
        print("\n🌲 Tree Learning Rate Factors:")
        print("=" * 70)
        print(f"{'Tree ID':>8} {'Age':>6} {'Fitness':>8} {'Age Factor':>12} {'Fitness Factor':>15} {'Combined':>10}")
        print("-" * 70)
        
        for tree in forest.trees:
            tree_id = tree.id
            age = self.age_system.tree_ages.get(tree_id, 0)
            fitness = self.age_system.tree_fitness.get(tree_id, 5.0)
            age_factor = self.age_system.compute_age_factor(tree_id)
            fitness_factor = self.age_system.compute_fitness_factor(tree_id)
            combined = age_factor * fitness_factor
            
            print(f"{tree_id:8d} {age:6d} {fitness:8.2f} {age_factor:12.4f} {fitness_factor:15.4f} {combined:10.4f}")
        
        print("=" * 70)


if __name__ == "__main__":
    """Example usage and testing."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from NeuralForest import ForestEcosystem, DEVICE
    from tasks.vision.classification import ImageClassification
    
    print("Testing Layer-Wise Optimizer System")
    print("=" * 70)
    
    # Create config
    config = LayerWiseConfig(
        base_lr=0.01,
        half_life=60.0,
        fitness_aware=True,
        warmup_epochs=5,
        total_epochs=100
    )
    
    print("\nConfiguration:")
    print(f"  Base LR: {config.base_lr}")
    print(f"  Half-life: {config.half_life}")
    print(f"  Warmup epochs: {config.warmup_epochs}")
    print(f"  Schedule: {config.schedule}")
    
    # Create forest and task head
    print("\nCreating forest...")
    forest = ForestEcosystem(
        input_dim=3072,
        hidden_dim=128,
        max_trees=6
    ).to(DEVICE)
    
    # Plant some trees
    for _ in range(5):
        forest._plant_tree()
    
    print(f"  Trees: {forest.num_trees()}")
    
    # Create task head
    task_head = ImageClassification(
        input_dim=128,
        num_classes=10
    ).to(DEVICE)
    
    # Create optimizer factory
    print("\nCreating optimizer factory...")
    opt_factory = LayerWiseOptimizer(config)
    
    # Test different epochs
    for epoch in [0, 5, 30, 60, 100]:
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}")
        print(f"{'='*70}")
        
        # Set tree ages for demonstration
        for i, tree in enumerate(forest.trees):
            tree.epoch_age = min(epoch, epoch - i * 10)
            tree.fitness = 5.0 + i * 0.5
        
        # Create optimizer
        optimizer = opt_factory.create_optimizer(forest, task_head, epoch)
        
        # Print summaries
        opt_factory.print_lr_summary(optimizer)
        opt_factory.log_tree_lr_factors(forest)
    
    print("\n✅ Layer-wise optimizer testing complete!")
