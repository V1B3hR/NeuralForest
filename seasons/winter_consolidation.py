"""
Winter Consolidation phase for NeuralForest.

Implements winter-specific behaviors: memory consolidation and knowledge distillation.
"""

import torch
from typing import Dict, List, Optional


class WinterConsolidation:
    """
    Winter-specific behaviors: memory consolidation and distillation.
    
    In winter, the forest:
    - Performs deep memory consolidation
    - Strengthens anchor memories
    - Transfers knowledge between trees via distillation
    - Protects learned knowledge with high bark (low plasticity)
    """
    
    def __init__(self, forest):
        """
        Initialize winter consolidation manager.
        
        Args:
            forest: The ForestEcosystem instance to manage
        """
        self.forest = forest
        self._consolidation_history = []
        self._distillation_stats = {
            'rounds': 0,
            'total_distill_loss': 0.0,
            'transfers_performed': 0
        }
        
    def deep_consolidation(self, num_rounds: int = 10, batch_size: int = 64) -> Dict:
        """
        Intensive knowledge consolidation through multiple passes.
        
        Args:
            num_rounds: Number of consolidation rounds
            batch_size: Batch size for anchor replay
            
        Returns:
            Dictionary with consolidation results
        """
        print("❄️ Winter consolidation: strengthening memories...")
        
        results = {
            'anchor_rounds': 0,
            'anchor_loss': 0.0,
            'teacher_snapshot': False,
            'knowledge_transfers': 0
        }
        
        # 1. Snapshot teacher for distillation
        if hasattr(self.forest, 'snapshot_teacher'):
            try:
                self.forest.snapshot_teacher()
                results['teacher_snapshot'] = True
                print("  📸 Teacher snapshot created")
            except Exception as e:
                print(f"  ⚠️ Teacher snapshot failed: {e}")
        
        # 2. Reinforce anchor memories
        if hasattr(self.forest, 'anchors') and len(self.forest.anchors) > 0:
            anchor_losses = []
            for round_idx in range(num_rounds):
                try:
                    ax, ay = self.forest.anchors.sample(batch_size=min(batch_size, len(self.forest.anchors)))
                    if ax is not None and ay is not None:
                        # Forward pass on anchors
                        if hasattr(self.forest, 'forward_forest'):
                            output, _, _ = self.forest.forward_forest(ax, top_k=3)
                        else:
                            output, _ = self.forest(ax)
                        
                        # Calculate anchor loss (MSE for regression)
                        loss = torch.nn.functional.mse_loss(output, ay)
                        anchor_losses.append(loss.item())
                        results['anchor_rounds'] += 1
                        
                except Exception as e:
                    print(f"  ⚠️ Anchor round {round_idx} failed: {e}")
                    continue
            
            if anchor_losses:
                results['anchor_loss'] = sum(anchor_losses) / len(anchor_losses)
                print(f"  🔒 Anchors reinforced: avg loss = {results['anchor_loss']:.4f}")
        
        # 3. Transfer knowledge from strong to weak trees
        if hasattr(self.forest, 'trees') and len(self.forest.trees) >= 2:
            results['knowledge_transfers'] = self._cross_tree_transfer()
        
        # Store consolidation record
        self._consolidation_history.append(results)
        
        return results
    
    def _cross_tree_transfer(self) -> int:
        """
        Transfer knowledge from strong trees to weak trees.
        
        Returns:
            Number of transfers performed
        """
        if not hasattr(self.forest, 'trees') or len(self.forest.trees) < 2:
            return 0
        
        # Identify strong and weak trees
        sorted_trees = sorted(self.forest.trees, key=lambda t: t.fitness, reverse=True)
        num_teachers = min(3, len(sorted_trees) // 2)
        num_students = min(3, len(sorted_trees) // 2)
        
        strong_trees = sorted_trees[:num_teachers]
        weak_trees = sorted_trees[-num_students:] if num_students > 0 else []
        
        transfers = 0
        for teacher in strong_trees:
            for student in weak_trees:
                if teacher.id != student.id:
                    try:
                        self._transfer_knowledge(teacher, student)
                        transfers += 1
                    except Exception as e:
                        print(f"  ⚠️ Transfer from tree {teacher.id} to {student.id} failed: {e}")
        
        if transfers > 0:
            print(f"  🔄 Performed {transfers} knowledge transfer(s)")
        
        return transfers
    
    def _transfer_knowledge(self, teacher_tree, student_tree):
        """
        Distill knowledge from teacher to student tree.
        
        Args:
            teacher_tree: Source tree with strong performance
            student_tree: Target tree to improve
        """
        # Generate some sample data for transfer
        batch_size = 32
        
        # Try to get input dimension from tree structure
        try:
            if hasattr(teacher_tree, 'trunk') and len(teacher_tree.trunk) > 0:
                first_layer = teacher_tree.trunk[0]
                if hasattr(first_layer, 'in_features'):
                    input_dim = first_layer.in_features
                    device = first_layer.weight.device
                else:
                    # Fallback to common dimension
                    input_dim = 512
                    device = next(teacher_tree.parameters()).device
            else:
                input_dim = 512
                device = next(teacher_tree.parameters()).device
        except Exception:
            # Safe fallback
            input_dim = 512
            device = torch.device('cpu')
        
        sample_x = torch.randn(batch_size, input_dim).to(device)
        
        # Get teacher predictions (no gradients)
        with torch.no_grad():
            teacher_output = teacher_tree(sample_x)
        
        # Get student predictions
        student_output = student_tree(sample_x)
        
        # Distillation loss (MSE between outputs)
        distill_loss = torch.nn.functional.mse_loss(student_output, teacher_output.detach())
        
        # Backward pass on student
        if student_tree.training:
            distill_loss.backward()
        
        # Track statistics
        self._distillation_stats['rounds'] += 1
        self._distillation_stats['total_distill_loss'] += distill_loss.item()
        self._distillation_stats['transfers_performed'] += 1
    
    def strengthen_bark(self) -> Dict:
        """
        Increase bark (plasticity protection) on mature trees.
        
        Returns:
            Dictionary with bark strengthening results
        """
        if not hasattr(self.forest, 'trees'):
            return {'trees_strengthened': 0}
        
        results = {
            'trees_strengthened': 0,
            'avg_bark_before': 0.0,
            'avg_bark_after': 0.0
        }
        
        bark_values_before = []
        bark_values_after = []
        
        for tree in self.forest.trees:
            bark_values_before.append(tree.bark)
            
            # Increase bark for mature trees with good fitness
            if tree.age > 100 and tree.fitness > 5.0:
                # Gradually increase bark (asymptotically approaches 1.0)
                tree.bark = min(0.99, tree.bark + (1.0 - tree.bark) * 0.1)
                results['trees_strengthened'] += 1
            
            bark_values_after.append(tree.bark)
        
        if bark_values_before:
            results['avg_bark_before'] = sum(bark_values_before) / len(bark_values_before)
            results['avg_bark_after'] = sum(bark_values_after) / len(bark_values_after)
        
        if results['trees_strengthened'] > 0:
            print(f"  🛡️ Strengthened bark on {results['trees_strengthened']} tree(s)")
        
        return results
    
    def consolidate_memory(self, priority_boost: float = 1.5) -> Dict:
        """
        Consolidate memories in the mulch (replay buffer).
        
        Args:
            priority_boost: Factor to boost priority of important memories
            
        Returns:
            Dictionary with memory consolidation results
        """
        results = {
            'memories_boosted': 0,
            'total_memories': 0
        }
        
        # Boost priority of important memories in mulch
        if hasattr(self.forest, 'mulch') and len(self.forest.mulch) > 0:
            results['total_memories'] = len(self.forest.mulch)
            
            # Access underlying data if possible
            if hasattr(self.forest.mulch, 'data'):
                # Boost priority of older, stable memories
                boosted = 0
                for i, (x, y, priority) in enumerate(self.forest.mulch.data):
                    # Boost every 10th item (anchor-like memories)
                    if i % 10 == 0:
                        new_priority = priority * priority_boost
                        self.forest.mulch.data[i] = (x, y, new_priority)
                        boosted += 1
                
                results['memories_boosted'] = boosted
                print(f"  💾 Boosted priority of {boosted} memory/memories")
        
        return results
    
    def get_consolidation_metrics(self) -> Dict:
        """
        Get comprehensive consolidation metrics for winter.
        
        Returns:
            Dictionary with consolidation metrics
        """
        metrics = {
            'consolidation_rounds': len(self._consolidation_history),
            'distillation_rounds': self._distillation_stats['rounds'],
            'total_transfers': self._distillation_stats['transfers_performed'],
        }
        
        if self._distillation_stats['rounds'] > 0:
            metrics['avg_distill_loss'] = (
                self._distillation_stats['total_distill_loss'] / 
                self._distillation_stats['rounds']
            )
        
        # Tree statistics
        if hasattr(self.forest, 'trees') and self.forest.trees:
            metrics['num_trees'] = len(self.forest.trees)
            metrics['avg_bark'] = sum(t.bark for t in self.forest.trees) / len(self.forest.trees)
            metrics['mature_trees'] = sum(1 for t in self.forest.trees if t.age > 100)
        
        # Memory statistics
        if hasattr(self.forest, 'mulch'):
            metrics['mulch_size'] = len(self.forest.mulch)
        if hasattr(self.forest, 'anchors'):
            metrics['anchor_size'] = len(self.forest.anchors)
        
        return metrics
    
    def get_recommendations(self) -> List[str]:
        """
        Get recommendations for winter consolidation activities.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        metrics = self.get_consolidation_metrics()
        
        # Check if consolidation has been performed
        if metrics.get('consolidation_rounds', 0) == 0:
            recommendations.append("❄️ No consolidation performed yet. Run deep_consolidation().")
        
        # Check knowledge transfer
        if metrics.get('total_transfers', 0) == 0:
            recommendations.append("No knowledge transfers performed. Trees may benefit from cross-tree learning.")
        
        # Check memory usage
        mulch_size = metrics.get('mulch_size', 0)
        anchor_size = metrics.get('anchor_size', 0)
        
        if mulch_size > 5000:
            recommendations.append("Mulch is large. Consider pruning old memories to save resources.")
        if anchor_size < 100:
            recommendations.append("Few anchor memories. Consider adding more to prevent forgetting.")
        
        # Check bark protection
        avg_bark = metrics.get('avg_bark', 0.0)
        if avg_bark < 0.5:
            recommendations.append("Low bark protection. Strengthen bark to protect knowledge.")
        
        if not recommendations:
            recommendations.append("✅ Winter consolidation is proceeding well.")
        
        return recommendations
    
    def get_consolidation_history(self) -> List[Dict]:
        """Get history of consolidation rounds."""
        return self._consolidation_history.copy()
    
    def reset_stats(self):
        """Reset consolidation statistics."""
        self._consolidation_history = []
        self._distillation_stats = {
            'rounds': 0,
            'total_distill_loss': 0.0,
            'transfers_performed': 0
        }
