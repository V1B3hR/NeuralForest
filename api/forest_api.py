"""
Production-Ready API for NeuralForest

Provides a clean, high-level interface for deploying and using NeuralForest
in production environments. Supports prediction, online learning, monitoring,
and checkpoint management.
"""

import time
import os
from typing import Dict, List, Any, Optional, Union
import torch
import torch.nn as nn


class ForestCheckpoint:
    """
    Utility class for managing forest checkpoints.
    
    Provides static methods for saving and loading forest state,
    validating checkpoints, and managing checkpoint directories.
    """
    
    @staticmethod
    def save(forest, path: str, metadata: Optional[Dict] = None):
        """
        Save forest to checkpoint file.
        
        Args:
            forest: ForestEcosystem instance to save
            path: Path to save checkpoint
            metadata: Optional metadata dictionary to include
        """
        forest.save_checkpoint(path)
        
        # Add metadata if provided
        if metadata:
            checkpoint = torch.load(path)
            checkpoint['metadata'] = metadata
            torch.save(checkpoint, path)
    
    @staticmethod
    def load(path: str, device=None):
        """
        Load forest from checkpoint file.
        
        Args:
            path: Path to checkpoint file
            device: Device to load to (default: auto-detect)
            
        Returns:
            Loaded ForestEcosystem instance
        """
        # Import here to avoid circular dependency
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from NeuralForest import ForestEcosystem
        
        return ForestEcosystem.load_checkpoint(path, device=device)
    
    @staticmethod
    def validate(path: str) -> bool:
        """
        Validate that a checkpoint file is valid and loadable.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            True if checkpoint is valid
        """
        try:
            checkpoint = torch.load(path, map_location='cpu')
            required_keys = ['input_dim', 'hidden_dim', 'tree_states', 'router_state_dict']
            return all(key in checkpoint for key in required_keys)
        except Exception:
            return False
    
    @staticmethod
    def get_info(path: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint without fully loading it.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        return {
            'valid': True,
            'num_trees': len(checkpoint.get('tree_states', [])),
            'input_dim': checkpoint.get('input_dim'),
            'hidden_dim': checkpoint.get('hidden_dim'),
            'max_trees': checkpoint.get('max_trees'),
            'memory_size': len(checkpoint.get('mulch_data', [])),
            'anchor_size': len(checkpoint.get('anchor_data', [])),
            'metadata': checkpoint.get('metadata', {}),
        }


class NeuralForestAPI:
    """
    Production-ready API for NeuralForest.
    
    Provides a high-level interface for:
    - Making predictions
    - Online learning
    - Performance monitoring
    - Resource management
    - Health checks
    
    Example usage:
        api = NeuralForestAPI(checkpoint_path='models/forest.pt')
        result = api.predict({'input': data})
        api.train_online({'input': data}, target)
        status = api.get_forest_status()
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, forest=None, device=None):
        """
        Initialize the API.
        
        Args:
            checkpoint_path: Path to checkpoint file (if loading from file)
            forest: ForestEcosystem instance (if using existing forest)
            device: Device to use (default: auto-detect)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        if checkpoint_path and forest:
            raise ValueError("Provide either checkpoint_path or forest, not both")
        
        if checkpoint_path:
            self.forest = ForestCheckpoint.load(checkpoint_path, device=self.device)
        elif forest:
            self.forest = forest.to(self.device)
        else:
            raise ValueError("Must provide either checkpoint_path or forest")
        
        self.forest.eval()
        
        # Performance tracking
        self.prediction_count = 0
        self.prediction_times = []
        self.training_count = 0
        
        # Health tracking
        self.last_health_check = None
        self.health_history = []
    
    def predict(
        self,
        inputs: Dict[str, torch.Tensor],
        top_k: int = 3,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Universal prediction endpoint.
        
        Args:
            inputs: Dictionary with input tensors
            top_k: Number of trees to use for prediction
            return_details: If True, return detailed information
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Extract input tensor
        if 'input' in inputs:
            x = inputs['input']
        elif 'x' in inputs:
            x = inputs['x']
        else:
            # Use first value
            x = next(iter(inputs.values()))
        
        # Ensure correct device and shape
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output, weights, tree_outputs = self.forest.forward_forest(x, top_k=top_k)
        
        # Compute confidence
        confidence = self._compute_confidence(weights, tree_outputs)
        
        # Track performance
        processing_time = time.time() - start_time
        self.prediction_times.append(processing_time)
        self.prediction_count += 1
        
        result = {
            'prediction': output.cpu().numpy(),
            'confidence': confidence,
            'processing_time_ms': processing_time * 1000,
        }
        
        if return_details:
            result['trees_used'] = self._get_active_trees(weights)
            result['routing_weights'] = weights.cpu().numpy()
            result['tree_outputs'] = [out.cpu().numpy() for out in tree_outputs]
        
        return result
    
    def train_online(
        self,
        inputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        feedback: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Online learning from new examples.
        
        Args:
            inputs: Input dictionary
            target: Target values
            feedback: Optional feedback score (0-1)
            
        Returns:
            Dictionary with training results
        """
        # Extract input
        if 'input' in inputs:
            x = inputs['input']
        elif 'x' in inputs:
            x = inputs['x']
        else:
            x = next(iter(inputs.values()))
        
        # Ensure tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
        
        x = x.to(self.device)
        target = target.to(self.device)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        
        # Compute priority based on feedback or prediction error
        with torch.no_grad():
            pred, _, _ = self.forest.forward_forest(x)
            error = (pred - target).abs().mean().item()
        
        priority = feedback if feedback is not None else error
        
        # Add to memory
        self.forest.mulch.add(x.squeeze(0), target.squeeze(0), priority)
        self.training_count += 1
        
        return {
            'success': True,
            'priority': priority,
            'memory_size': len(self.forest.mulch),
        }
    
    def get_forest_status(self) -> Dict[str, Any]:
        """
        Return current forest health and statistics.
        
        Returns:
            Dictionary with comprehensive status information
        """
        trees = self.forest.trees
        
        if not trees:
            return {
                'status': 'empty',
                'num_trees': 0,
            }
        
        fitnesses = [t.fitness for t in trees]
        ages = [t.age for t in trees]
        barks = [t.bark for t in trees]
        
        status = {
            'status': 'operational',
            'num_trees': len(trees),
            'num_groves': 1,  # Simplified for base forest
            'memory_usage': {
                'mulch_size': len(self.forest.mulch),
                'mulch_capacity': self.forest.mulch.capacity,
                'utilization': len(self.forest.mulch) / self.forest.mulch.capacity,
                'anchor_size': len(self.forest.anchors),
                'anchor_capacity': self.forest.anchors.capacity,
            },
            'performance': {
                'total_predictions': self.prediction_count,
                'total_training': self.training_count,
                'avg_prediction_time_ms': (
                    sum(self.prediction_times) / len(self.prediction_times) * 1000
                    if self.prediction_times else 0
                ),
            },
            'tree_health': {
                'average_fitness': sum(fitnesses) / len(fitnesses),
                'min_fitness': min(fitnesses),
                'max_fitness': max(fitnesses),
                'average_age': sum(ages) / len(ages),
                'average_bark': sum(barks) / len(barks),
            },
            'tree_details': [
                {
                    'id': t.id,
                    'age': t.age,
                    'fitness': t.fitness,
                    'bark': t.bark,
                }
                for t in trees
            ]
        }
        
        self.last_health_check = time.time()
        self.health_history.append(status)
        
        return status
    
    def health_check(self) -> Dict[str, Any]:
        """
        Quick health check for monitoring/alerting.
        
        Returns:
            Dictionary with health status
        """
        status = self.get_forest_status()
        
        # Determine health
        issues = []
        
        if status['num_trees'] == 0:
            issues.append('no_trees')
        elif status['num_trees'] < 2:
            issues.append('insufficient_trees')
        
        mem_util = status['memory_usage']['utilization']
        if mem_util > 0.95:
            issues.append('memory_critical')
        elif mem_util > 0.85:
            issues.append('memory_high')
        
        avg_fitness = status['tree_health']['average_fitness']
        if avg_fitness < 2.0:
            issues.append('low_fitness')
        
        health = 'healthy' if not issues else ('degraded' if len(issues) < 2 else 'unhealthy')
        
        return {
            'health': health,
            'issues': issues,
            'status': status,
            'timestamp': time.time(),
        }
    
    def save(self, path: str, metadata: Optional[Dict] = None):
        """
        Save current forest state.
        
        Args:
            path: Path to save checkpoint
            metadata: Optional metadata to include
        """
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'predictions_served': self.prediction_count,
            'training_samples': self.training_count,
            'save_timestamp': time.time(),
        })
        
        ForestCheckpoint.save(self.forest, path, metadata)
    
    def _compute_confidence(
        self,
        weights: torch.Tensor,
        tree_outputs: List[torch.Tensor]
    ) -> float:
        """
        Compute confidence score for predictions.
        
        Higher confidence when:
        - Routing weights are concentrated (not uniform)
        - Tree outputs agree (low variance)
        """
        # Concentration of routing weights
        weight_entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item()
        max_entropy = torch.log(torch.tensor(max(weights.size(-1), 2), dtype=torch.float32)).item()
        concentration = 1.0 - (weight_entropy / max_entropy) if max_entropy > 0 else 1.0
        
        # Agreement between trees
        if len(tree_outputs) > 1:
            outputs_stacked = torch.stack(tree_outputs, dim=1).squeeze(-1)
            variance = outputs_stacked.var(dim=1).mean().item()
            agreement = 1.0 / (1.0 + variance)
        else:
            agreement = 1.0
        
        # Combined confidence
        confidence = (concentration + agreement) / 2.0
        
        return float(confidence)
    
    def _get_active_trees(self, weights: torch.Tensor) -> List[int]:
        """Get IDs of trees with non-zero weight."""
        mean_weights = weights.mean(dim=0)
        active_indices = (mean_weights > 0).nonzero(as_tuple=True)[0].tolist()
        active_ids = [self.forest.trees[i].id for i in active_indices if i < len(self.forest.trees)]
        return active_ids
    
    def _timing(self) -> float:
        """Get average processing time in milliseconds."""
        if not self.prediction_times:
            return 0.0
        return sum(self.prediction_times) / len(self.prediction_times) * 1000
    
    def __repr__(self):
        return (
            f"NeuralForestAPI(trees={self.forest.num_trees()}, "
            f"predictions={self.prediction_count}, "
            f"device={self.device})"
        )
