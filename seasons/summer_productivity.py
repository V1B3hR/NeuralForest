"""
Summer Productivity phase for NeuralForest.

Implements summer-specific behaviors: maximum learning and expertise building.
"""

from typing import Dict, List, Optional
import torch


class SummerProductivity:
    """
    Summer-specific behaviors: maximum learning and building expertise.
    
    In summer, the forest:
    - Maximizes learning with optimal hyperparameters
    - Builds expertise through intensive training
    - Balances exploration and exploitation
    - Focuses on improving tree performance
    """
    
    def __init__(self, forest):
        """
        Initialize summer productivity manager.
        
        Args:
            forest: The ForestEcosystem instance to manage
        """
        self.forest = forest
        self._performance_history = []
        self._training_stats = {
            'steps': 0,
            'total_loss': 0.0,
            'best_fitness': 0.0
        }
        
    def optimize_learning(self, config: Dict) -> Dict:
        """
        Get optimized learning parameters for summer productivity.
        
        Args:
            config: Base training configuration
            
        Returns:
            Enhanced configuration for summer
        """
        summer_config = config.copy()
        
        # Summer emphasizes productive learning
        summer_config['batch_size'] = summer_config.get('batch_size', 32)
        summer_config['gradient_accumulation_steps'] = 1
        summer_config['warmup_steps'] = 100
        
        return summer_config
    
    def build_expertise(self, tree_id: int, task_performance: Dict) -> float:
        """
        Track and build expertise for a specific tree.
        
        Args:
            tree_id: ID of the tree
            task_performance: Dictionary of task names and performance scores
            
        Returns:
            Updated expertise score
        """
        if not hasattr(self.forest, 'trees'):
            return 0.0
        
        # Find the tree
        tree = None
        for t in self.forest.trees:
            if t.id == tree_id:
                tree = t
                break
        
        if tree is None:
            return 0.0
        
        # Calculate average performance
        if task_performance:
            avg_performance = sum(task_performance.values()) / len(task_performance)
            
            # Update expertise (weighted moving average)
            if hasattr(tree, 'expertise_score'):
                tree.expertise_score = 0.9 * tree.expertise_score + 0.1 * avg_performance
            else:
                tree.expertise_score = avg_performance
            
            return tree.expertise_score
        
        return getattr(tree, 'expertise_score', 0.0)
    
    def intensive_training_pass(self, batch_x, batch_y, optimizer, criterion) -> Dict:
        """
        Perform an intensive training pass focused on productivity.
        
        Args:
            batch_x: Input batch
            batch_y: Target batch
            optimizer: PyTorch optimizer
            criterion: Loss criterion
            
        Returns:
            Dictionary with training metrics
        """
        try:
            # Forward pass
            if hasattr(self.forest, 'forward_forest'):
                output, weights, tree_outputs = self.forest.forward_forest(batch_x, top_k=3)
            else:
                output, weights = self.forest(batch_x)
                tree_outputs = [output]
            
            # Compute loss
            loss = criterion(output, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            if hasattr(self.forest, 'parameters'):
                torch.nn.utils.clip_grad_norm_(self.forest.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update stats
            self._training_stats['steps'] += 1
            self._training_stats['total_loss'] += loss.item()
            
            return {
                'loss': loss.item(),
                'avg_loss': self._training_stats['total_loss'] / self._training_stats['steps'],
                'active_trees': weights.nonzero().size(0) if isinstance(weights, torch.Tensor) else len(weights)
            }
            
        except Exception as e:
            return {
                'loss': float('inf'),
                'error': str(e)
            }
    
    def balance_workload(self) -> Dict:
        """
        Analyze and balance workload across trees.
        
        Returns:
            Dictionary with workload analysis
        """
        if not hasattr(self.forest, 'trees'):
            return {'status': 'no_trees'}
        
        # Collect tree usage statistics
        tree_stats = []
        for tree in self.forest.trees:
            stats = {
                'id': tree.id,
                'age': tree.age,
                'fitness': tree.fitness,
                'expertise': getattr(tree, 'expertise_score', 0.0)
            }
            tree_stats.append(stats)
        
        if not tree_stats:
            return {'status': 'empty'}
        
        # Calculate balance metrics
        fitnesses = [s['fitness'] for s in tree_stats]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)
        min_fitness = min(fitnesses)
        
        return {
            'status': 'balanced' if (max_fitness - min_fitness) < avg_fitness * 0.5 else 'imbalanced',
            'num_trees': len(tree_stats),
            'avg_fitness': avg_fitness,
            'fitness_range': max_fitness - min_fitness,
            'tree_stats': tree_stats
        }
    
    def get_productivity_metrics(self) -> Dict:
        """
        Get comprehensive productivity metrics for summer.
        
        Returns:
            Dictionary with productivity metrics
        """
        metrics = {
            'training_steps': self._training_stats['steps'],
            'avg_loss': self._training_stats['total_loss'] / max(1, self._training_stats['steps']),
        }
        
        if hasattr(self.forest, 'trees') and self.forest.trees:
            metrics['num_trees'] = len(self.forest.trees)
            metrics['avg_tree_age'] = sum(t.age for t in self.forest.trees) / len(self.forest.trees)
            metrics['avg_tree_fitness'] = sum(t.fitness for t in self.forest.trees) / len(self.forest.trees)
            metrics['max_tree_fitness'] = max(t.fitness for t in self.forest.trees)
            
            # Track expertise if available
            expertise_scores = [getattr(t, 'expertise_score', 0.0) for t in self.forest.trees]
            if expertise_scores:
                metrics['avg_expertise'] = sum(expertise_scores) / len(expertise_scores)
        
        return metrics
    
    def recommend_optimizations(self) -> List[str]:
        """
        Provide recommendations for improving summer productivity.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        metrics = self.get_productivity_metrics()
        
        # Check training progress
        if metrics.get('avg_loss', float('inf')) > 1.0:
            recommendations.append("Loss is high. Consider increasing training intensity or adjusting learning rate.")
        
        # Check tree performance
        if 'avg_tree_fitness' in metrics:
            if metrics['avg_tree_fitness'] < 5.0:
                recommendations.append("Tree fitness is low. Focus on more diverse training examples.")
            
            # Check for imbalance
            workload = self.balance_workload()
            if workload.get('status') == 'imbalanced':
                recommendations.append("Tree workload is imbalanced. Consider load balancing strategies.")
        
        # Check tree count
        if metrics.get('num_trees', 0) < 3:
            recommendations.append("Few trees available. Consider planting more specialists.")
        
        if not recommendations:
            recommendations.append("Productivity is optimal. Continue with current strategy.")
        
        return recommendations
    
    def reset_stats(self):
        """Reset training statistics."""
        self._training_stats = {
            'steps': 0,
            'total_loss': 0.0,
            'best_fitness': 0.0
        }
        self._performance_history = []
