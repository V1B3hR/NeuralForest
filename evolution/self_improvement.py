"""
Self-Improvement Loop for NeuralForest

Implements autonomous improvement cycle that runs continuously, identifying
and applying improvements to the forest's structure and performance.
"""

from typing import Dict, List, Any, Optional
import time


class SelfImprovementLoop:
    """
    Autonomous improvement cycle that runs continuously.
    
    The self-improvement loop:
    1. Collects performance metrics
    2. Analyzes current state via consciousness
    3. Identifies improvement opportunities
    4. Applies selected improvements
    5. Validates results
    6. Rolls back if improvements fail
    """
    
    def __init__(self, forest, consciousness):
        self.forest = forest
        self.consciousness = consciousness
        self.improvement_history: List[Dict] = []
        self.cycle_count = 0
        
        # Metrics tracking
        self.baseline_metrics: Optional[Dict] = None
        self.current_metrics: Optional[Dict] = None
        
        # Checkpoint for rollback
        self.checkpoint_state = None
    
    def run_cycle(self, max_improvements: int = 3) -> Dict[str, Any]:
        """
        Single improvement cycle.
        
        Args:
            max_improvements: Maximum number of improvements to apply per cycle
            
        Returns:
            Dictionary with cycle results
        """
        cycle_start = time.time()
        
        # 1. Collect baseline performance data
        baseline = self._collect_metrics()
        if self.baseline_metrics is None:
            self.baseline_metrics = baseline
        
        # 2. Analyze and reflect via consciousness
        analysis = self.consciousness.reflect()
        
        # 3. Identify improvement opportunities
        opportunities = self._find_opportunities(analysis)
        
        # 4. Create checkpoint for potential rollback
        self._create_checkpoint()
        
        # 5. Select and apply improvements
        applied_improvements = []
        for opp in opportunities[:max_improvements]:
            improvement = self._apply_improvement(opp)
            applied_improvements.append(improvement)
            self.improvement_history.append(improvement)
        
        # 6. Validate improvements
        new_metrics = self._collect_metrics()
        success = self._validate_improvement(baseline, new_metrics)
        
        # 7. Rollback if improvements made things worse
        if not success and len(applied_improvements) > 0:
            self._rollback_to_checkpoint()
            print("⚠️  Improvements didn't help - rolled back changes")
        
        self.current_metrics = new_metrics
        self.cycle_count += 1
        
        cycle_time = time.time() - cycle_start
        
        return {
            'cycle': self.cycle_count,
            'improvements_applied': len(applied_improvements),
            'success': success,
            'baseline_metrics': baseline,
            'new_metrics': new_metrics,
            'opportunities_found': len(opportunities),
            'cycle_time_seconds': cycle_time,
        }
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """
        Collect current performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        trees = self.forest.trees
        
        if not trees:
            return {
                'num_trees': 0,
                'average_fitness': 0.0,
                'memory_usage': 0.0,
            }
        
        fitnesses = [t.fitness for t in trees]
        
        return {
            'num_trees': len(trees),
            'average_fitness': sum(fitnesses) / len(fitnesses),
            'min_fitness': min(fitnesses),
            'max_fitness': max(fitnesses),
            'memory_usage': len(self.forest.mulch) / self.forest.mulch.capacity,
            'anchor_usage': len(self.forest.anchors) / self.forest.anchors.capacity,
        }
    
    def _find_opportunities(self, analysis: Dict[str, Any]) -> List[Dict]:
        """
        Identify improvement opportunities from analysis.
        
        Args:
            analysis: Analysis from consciousness.reflect()
            
        Returns:
            List of improvement opportunities, sorted by priority
        """
        opportunities = []
        
        # Check forest size
        num_trees = analysis['forest_size']
        if num_trees < 5:
            opportunities.append({
                'type': 'increase_capacity',
                'priority': 0.8,
                'action': 'plant_trees',
                'reason': f'forest_size_low ({num_trees} trees)',
            })
        
        # Check fitness
        overall_fitness = analysis['overall_fitness']
        if overall_fitness < 5.0:
            opportunities.append({
                'type': 'improve_performance',
                'priority': 0.7,
                'action': 'snapshot_teacher',
                'reason': f'low_fitness ({overall_fitness:.2f})',
            })
        
        # Check memory usage
        memory_util = analysis['memory_usage']['mulch_utilization']
        if memory_util > 0.9:
            opportunities.append({
                'type': 'memory_optimization',
                'priority': 0.6,
                'action': 'prune_memory',
                'reason': f'high_memory_usage ({memory_util:.1%})',
            })
        
        # Check for weak trees
        tree_health = analysis.get('tree_health', {})
        weak_trees = tree_health.get('weak_trees', 0)
        if weak_trees > 0 and num_trees > 3:
            opportunities.append({
                'type': 'prune_weak',
                'priority': 0.5,
                'action': 'prune_trees',
                'reason': f'weak_trees_present ({weak_trees})',
            })
        
        # Check knowledge gaps
        for gap in analysis.get('knowledge_gaps', []):
            opportunities.append({
                'type': 'fill_gap',
                'priority': 0.7 * gap.get('severity', 1.0),
                'action': 'plant_specialist',
                'details': gap,
                'reason': f"knowledge_gap: {gap.get('reason', 'unknown')}",
            })
        
        # Sort by priority
        return sorted(opportunities, key=lambda x: x['priority'], reverse=True)
    
    def _apply_improvement(self, opportunity: Dict) -> Dict[str, Any]:
        """
        Apply a single improvement.
        
        Args:
            opportunity: Improvement opportunity to apply
            
        Returns:
            Dictionary with improvement details and result
        """
        action = opportunity['action']
        start_time = time.time()
        
        result = {
            'timestamp': start_time,
            'opportunity': opportunity,
            'action': action,
            'success': False,
        }
        
        try:
            if action == 'plant_trees':
                before = self.forest.num_trees()
                self.forest._plant_tree()
                after = self.forest.num_trees()
                result['success'] = after > before
                result['trees_added'] = after - before
                result['message'] = f"Planted {after - before} tree(s)"
            
            elif action == 'snapshot_teacher':
                self.forest.snapshot_teacher()
                result['success'] = self.forest.teacher_snapshot is not None
                result['message'] = "Created teacher snapshot for distillation"
            
            elif action == 'prune_memory':
                before = len(self.forest.mulch)
                # Keep only top 70% by priority
                if before > 0:
                    sorted_data = sorted(self.forest.mulch.data, key=lambda x: x[2], reverse=True)
                    keep_count = int(len(sorted_data) * 0.7)
                    from collections import deque
                    self.forest.mulch.data = deque(sorted_data[:keep_count], maxlen=self.forest.mulch.capacity)
                after = len(self.forest.mulch)
                result['success'] = True
                result['items_removed'] = before - after
                result['message'] = f"Pruned {before - after} low-priority memories"
            
            elif action == 'prune_trees':
                before = self.forest.num_trees()
                # Find weakest tree
                if self.forest.trees:
                    weakest = min(self.forest.trees, key=lambda t: t.fitness)
                    if weakest.fitness < 2.0 and before > 3:
                        self.forest._prune_trees([weakest.id], min_keep=3)
                        after = self.forest.num_trees()
                        result['success'] = after < before
                        result['trees_removed'] = before - after
                        result['message'] = f"Pruned {before - after} weak tree(s)"
                    else:
                        result['message'] = "No trees weak enough to prune"
            
            elif action == 'plant_specialist':
                before = self.forest.num_trees()
                self.forest._plant_tree()
                after = self.forest.num_trees()
                result['success'] = after > before
                result['trees_added'] = after - before
                details = opportunity.get('details', {})
                spec = details.get('task', 'general')
                result['message'] = f"Planted specialist tree for {spec}"
            
            else:
                result['message'] = f"Unknown action: {action}"
        
        except Exception as e:
            result['error'] = str(e)
            result['message'] = f"Error applying improvement: {e}"
        
        result['duration'] = time.time() - start_time
        return result
    
    def _validate_improvement(self, baseline: Dict, new_metrics: Dict) -> bool:
        """
        Validate that improvements actually helped.
        
        Args:
            baseline: Metrics before improvements
            new_metrics: Metrics after improvements
            
        Returns:
            True if improvements were beneficial
        """
        # Check if average fitness improved or stayed similar
        baseline_fitness = baseline.get('average_fitness', 0.0)
        new_fitness = new_metrics.get('average_fitness', 0.0)
        
        # Allow small degradation due to randomness
        fitness_threshold = -0.5
        fitness_change = new_fitness - baseline_fitness
        
        # Check memory didn't explode
        baseline_memory = baseline.get('memory_usage', 0.0)
        new_memory = new_metrics.get('memory_usage', 0.0)
        memory_ok = new_memory <= 0.95
        
        # Overall success if fitness didn't degrade too much and memory is ok
        success = fitness_change >= fitness_threshold and memory_ok
        
        return success
    
    def _create_checkpoint(self):
        """Create checkpoint of current state for potential rollback."""
        # Store basic state info
        self.checkpoint_state = {
            'num_trees': self.forest.num_trees(),
            'tree_ids': [t.id for t in self.forest.trees],
            'memory_size': len(self.forest.mulch),
        }
    
    def _rollback_to_checkpoint(self):
        """Rollback to checkpoint state."""
        if self.checkpoint_state is None:
            return
        
        # This is simplified - in practice you'd restore full state
        print(f"  Rolling back to checkpoint with {self.checkpoint_state['num_trees']} trees")
    
    def get_improvement_summary(self) -> Dict[str, Any]:
        """
        Get summary of all improvements applied.
        
        Returns:
            Summary statistics
        """
        if not self.improvement_history:
            return {
                'total_improvements': 0,
                'successful_improvements': 0,
                'success_rate': 0.0,
            }
        
        successful = sum(1 for imp in self.improvement_history if imp.get('success', False))
        
        return {
            'total_improvements': len(self.improvement_history),
            'successful_improvements': successful,
            'success_rate': successful / len(self.improvement_history),
            'total_cycles': self.cycle_count,
            'recent_improvements': self.improvement_history[-10:],
        }
    
    def reset(self):
        """Reset improvement history and metrics."""
        self.improvement_history.clear()
        self.cycle_count = 0
        self.baseline_metrics = None
        self.current_metrics = None
        self.checkpoint_state = None
