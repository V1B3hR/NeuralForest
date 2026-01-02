"""
Autumn Pruning phase for NeuralForest.

Implements autumn-specific behaviors: fitness analysis and pruning weak trees.
"""

import random
from typing import Dict, List, Optional


class AutumnPruning:
    """
    Autumn-specific behaviors: fitness analysis and pruning.
    
    In autumn, the forest:
    - Evaluates health and fitness of all trees
    - Identifies weak or underperforming trees
    - Prunes trees that are not contributing
    - Provides recommendations for forest health
    """
    
    def __init__(self, forest):
        """
        Initialize autumn pruning manager.
        
        Args:
            forest: The ForestEcosystem instance to manage
        """
        self.forest = forest
        self._pruning_history = []
        self._health_reports = []
        
    def evaluate_forest_health(self) -> Dict:
        """
        Comprehensive health check of all trees.
        
        Returns:
            Dictionary with forest health report
        """
        if not hasattr(self.forest, 'trees') or not self.forest.trees:
            return {
                "total_trees": 0,
                "trees": [],
                "recommendations": ["Forest is empty. Plant initial trees."],
                "overall_health": "poor"
            }
        
        report = {
            "total_trees": len(self.forest.trees),
            "trees": [],
            "recommendations": []
        }
        
        weak_count = 0
        thriving_count = 0
        ancient_count = 0
        
        for tree in self.forest.trees:
            health = {
                "id": tree.id,
                "age": tree.age,
                "fitness": tree.fitness,
                "bark": tree.bark,
                "specialization": getattr(tree, 'specialization', 'general'),
                "status": self._assess_status(tree)
            }
            report["trees"].append(health)
            
            # Count status types
            if health["status"] == "weak":
                weak_count += 1
                report["recommendations"].append(
                    f"🍂 Consider pruning tree {tree.id} (fitness: {tree.fitness:.2f}, age: {tree.age})"
                )
            elif health["status"] == "thriving":
                thriving_count += 1
            elif health["status"] == "ancient":
                ancient_count += 1
        
        # Overall health assessment
        if thriving_count > len(self.forest.trees) * 0.6:
            report["overall_health"] = "excellent"
        elif weak_count > len(self.forest.trees) * 0.4:
            report["overall_health"] = "poor"
        else:
            report["overall_health"] = "good"
        
        # Add summary recommendations
        if weak_count > 0:
            report["recommendations"].insert(0, 
                f"Found {weak_count} weak tree(s). Pruning recommended.")
        if ancient_count > 0:
            report["recommendations"].append(
                f"Found {ancient_count} ancient tree(s). They have high bark protection.")
        if thriving_count == 0:
            report["recommendations"].append(
                "No thriving trees found. Consider adjusting training strategy.")
        
        # Store report
        self._health_reports.append(report)
        
        return report
    
    def _assess_status(self, tree) -> str:
        """
        Assess the health status of a single tree.
        
        Args:
            tree: Tree to assess
            
        Returns:
            Status string: "weak", "thriving", "ancient", or "healthy"
        """
        # Ancient trees have high bark protection
        if tree.bark > 0.8:
            return "ancient"
        
        # Weak trees have low fitness and are mature
        if tree.fitness < 2.0 and tree.age > 50:
            return "weak"
        
        # Thriving trees have high fitness
        if tree.fitness > 8.0:
            return "thriving"
        
        # Default to healthy
        return "healthy"
    
    def prune_weakest(self, config: Dict, min_keep: int = 3) -> List[int]:
        """
        Prune the weakest trees based on configuration.
        
        Args:
            config: Training configuration with prune_probability
            min_keep: Minimum number of trees to keep
            
        Returns:
            List of pruned tree IDs
        """
        if not hasattr(self.forest, 'trees') or len(self.forest.trees) <= min_keep:
            return []
        
        # Check if we should prune this step
        if random.random() >= config.get("prune_probability", 0.0):
            return []
        
        # Find weak trees
        weak_trees = [
            t for t in self.forest.trees
            if t.fitness < 2.0 and t.age > 40
        ]
        
        if not weak_trees or len(self.forest.trees) <= min_keep:
            return []
        
        # Determine how many to remove (at most half of weak trees, respecting min_keep)
        max_to_remove = min(
            len(weak_trees) // 2 + 1,
            len(self.forest.trees) - min_keep
        )
        
        if max_to_remove <= 0:
            return []
        
        # Sort by fitness and select weakest
        weakest = sorted(weak_trees, key=lambda t: t.fitness)[:max_to_remove]
        tree_ids_to_prune = [t.id for t in weakest]
        
        # Log pruning
        for tree in weakest:
            print(f"🍂 Autumn pruning: removing tree {tree.id} (fitness: {tree.fitness:.2f}, age: {tree.age})")
            self._pruning_history.append({
                'tree_id': tree.id,
                'fitness': tree.fitness,
                'age': tree.age,
                'reason': 'low_fitness'
            })
        
        # Perform pruning
        if hasattr(self.forest, '_prune_trees'):
            try:
                self.forest._prune_trees(tree_ids_to_prune, min_keep=min_keep)
            except Exception as e:
                print(f"⚠️ Warning: Pruning failed: {e}")
                return []
        
        return tree_ids_to_prune
    
    def identify_redundant_trees(self, similarity_threshold: float = 0.9) -> List[tuple]:
        """
        Identify trees that are too similar and potentially redundant.
        
        Args:
            similarity_threshold: Cosine similarity threshold for redundancy
            
        Returns:
            List of tuples (tree_id_1, tree_id_2, similarity_score)
        """
        if not hasattr(self.forest, 'trees') or len(self.forest.trees) < 2:
            return []
        
        import torch
        redundant_pairs = []
        
        try:
            # Compare tree parameters for similarity
            for i, tree_a in enumerate(self.forest.trees):
                for tree_b in self.forest.trees[i+1:]:
                    # Get tree parameters as vectors
                    params_a = torch.cat([p.flatten() for p in tree_a.parameters()])
                    params_b = torch.cat([p.flatten() for p in tree_b.parameters()])
                    
                    # Compute cosine similarity
                    similarity = torch.nn.functional.cosine_similarity(
                        params_a.unsqueeze(0), 
                        params_b.unsqueeze(0)
                    ).item()
                    
                    if similarity > similarity_threshold:
                        redundant_pairs.append((tree_a.id, tree_b.id, similarity))
        
        except Exception as e:
            # Silently fail if parameter comparison doesn't work
            pass
        
        return redundant_pairs
    
    def get_pruning_recommendations(self) -> List[str]:
        """
        Get recommendations for autumn pruning activities.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not hasattr(self.forest, 'trees'):
            return ["No trees to analyze."]
        
        # Evaluate current health
        health = self.evaluate_forest_health()
        
        if health["overall_health"] == "poor":
            recommendations.append("⚠️ Forest health is poor. Aggressive pruning recommended.")
        elif health["overall_health"] == "excellent":
            recommendations.append("✅ Forest health is excellent. Light maintenance pruning only.")
        
        # Check for redundancy
        redundant = self.identify_redundant_trees(similarity_threshold=0.85)
        if redundant:
            recommendations.append(
                f"Found {len(redundant)} redundant tree pair(s). Consider pruning for efficiency."
            )
        
        # Check tree age distribution
        if self.forest.trees:
            ages = [t.age for t in self.forest.trees]
            avg_age = sum(ages) / len(ages)
            
            if avg_age > 200:
                recommendations.append("Trees are mature. Consider planting younger specialists.")
            elif avg_age < 50:
                recommendations.append("Trees are young. Give them time to mature before pruning.")
        
        if not recommendations:
            recommendations.append("Forest is well-balanced. No immediate actions needed.")
        
        return recommendations
    
    def get_pruning_history(self) -> List[Dict]:
        """Get history of pruned trees."""
        return self._pruning_history.copy()
    
    def get_health_history(self) -> List[Dict]:
        """Get history of health evaluations."""
        return self._health_reports.copy()
    
    def reset_history(self):
        """Reset pruning and health history."""
        self._pruning_history = []
        self._health_reports = []
