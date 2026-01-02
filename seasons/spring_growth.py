"""
Spring Growth phase for NeuralForest.

Implements spring-specific behaviors: planting new trees and exploration.
"""

import random
from typing import Dict, Optional, List


class SpringGrowth:
    """
    Spring-specific behaviors: planting and exploration.
    
    In spring, the forest:
    - Plants new specialized trees based on weaknesses
    - Encourages high exploration and plasticity
    - Focuses on growth rather than pruning
    """
    
    def __init__(self, forest):
        """
        Initialize spring growth manager.
        
        Args:
            forest: The ForestEcosystem instance to manage
        """
        self.forest = forest
        self._planting_history = []
        self._weakness_analysis = {}
        
    def maybe_plant_trees(self, loss_trend: Optional[List[float]], config: Dict) -> bool:
        """
        Decide whether to plant new trees based on configuration and forest state.
        
        Args:
            loss_trend: Recent loss values (optional)
            config: Training configuration from SeasonalCycle
            
        Returns:
            True if a tree was planted
        """
        if random.random() < config["growth_probability"]:
            # Determine what kind of tree to plant based on current weaknesses
            weakness = self._identify_weakness(loss_trend)
            
            # Only plant if we have identified a clear weakness
            if weakness:
                self._plant_specialist(weakness)
                print(f"🌱 Spring planting: new {weakness} specialist tree")
                return True
        
        return False
    
    def _identify_weakness(self, loss_trend: Optional[List[float]] = None) -> Optional[str]:
        """
        Analyze which modality/task has highest loss.
        Plant tree specialized for that area.
        
        Args:
            loss_trend: Recent loss values for analysis
            
        Returns:
            Specialization type string, or None if no clear weakness
        """
        # If we don't have enough trees, plant a general tree
        if hasattr(self.forest, 'trees') and len(self.forest.trees) < 3:
            return "general"
        
        # Analyze recent performance if available
        if loss_trend and len(loss_trend) > 5:
            recent_avg = sum(loss_trend[-5:]) / 5
            earlier_avg = sum(loss_trend[-10:-5]) / 5 if len(loss_trend) >= 10 else recent_avg
            
            # If loss is increasing, we need more capacity
            if recent_avg > earlier_avg * 1.1:
                return self._suggest_specialization()
        
        # Check if forest needs diversity
        if hasattr(self.forest, 'trees'):
            specializations = {}
            for tree in self.forest.trees:
                spec = getattr(tree, 'specialization', 'general')
                specializations[spec] = specializations.get(spec, 0) + 1
            
            # If we have too many of one type, diversify
            if specializations:
                most_common = max(specializations.values())
                if most_common > len(self.forest.trees) * 0.5:
                    return self._suggest_specialization(avoid=max(specializations, key=specializations.get))
        
        return None
    
    def _suggest_specialization(self, avoid: Optional[str] = None) -> str:
        """
        Suggest a specialization type for a new tree.
        
        Args:
            avoid: Specialization type to avoid (optional)
            
        Returns:
            Suggested specialization string
        """
        # Common specializations across modalities
        specializations = [
            "classification",
            "detection", 
            "segmentation",
            "general",
            "feature_extraction",
            "pattern_recognition"
        ]
        
        # Remove the one to avoid
        if avoid:
            specializations = [s for s in specializations if s != avoid]
        
        return random.choice(specializations)
    
    def _plant_specialist(self, specialization: str):
        """
        Plant a new specialist tree in the forest.
        
        Args:
            specialization: Type of specialization for the new tree
        """
        try:
            if hasattr(self.forest, 'plant_specialist'):
                self.forest.plant_specialist(specialization)
            elif hasattr(self.forest, '_grow_tree'):
                # Fallback to generic tree growing
                self.forest._grow_tree()
            
            self._planting_history.append({
                'specialization': specialization,
                'num_trees_after': len(self.forest.trees) if hasattr(self.forest, 'trees') else None
            })
        except Exception as e:
            print(f"⚠️ Warning: Could not plant tree: {e}")
    
    def encourage_exploration(self, routing_weights) -> float:
        """
        Calculate exploration bonus to encourage using diverse trees.
        
        Args:
            routing_weights: Current routing weights for trees
            
        Returns:
            Exploration bonus value
        """
        # Entropy-based exploration: reward diverse tree usage
        import torch
        if isinstance(routing_weights, torch.Tensor):
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            probs = routing_weights + epsilon
            probs = probs / probs.sum()
            entropy = -(probs * torch.log(probs)).sum()
            
            # Normalize entropy (max entropy for uniform distribution)
            max_entropy = torch.log(torch.tensor(len(routing_weights), dtype=torch.float32))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            return normalized_entropy.item()
        
        return 0.0
    
    def get_growth_recommendations(self) -> List[str]:
        """
        Get recommendations for spring growth activities.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if hasattr(self.forest, 'trees'):
            num_trees = len(self.forest.trees)
            
            if num_trees < 3:
                recommendations.append("Forest is sparse. Plant more trees to increase capacity.")
            elif num_trees < 6:
                recommendations.append("Consider planting specialist trees for better coverage.")
            
            # Check tree diversity
            if hasattr(self.forest, 'trees') and num_trees > 0:
                specializations = set()
                for tree in self.forest.trees:
                    spec = getattr(tree, 'specialization', 'general')
                    specializations.add(spec)
                
                if len(specializations) < num_trees * 0.5:
                    recommendations.append("Low diversity detected. Plant trees with different specializations.")
        
        if not recommendations:
            recommendations.append("Forest growth is healthy. Continue with current strategy.")
        
        return recommendations
    
    def get_planting_history(self) -> List[Dict]:
        """Get history of trees planted during spring."""
        return self._planting_history.copy()
    
    def reset_history(self):
        """Reset planting history."""
        self._planting_history = []
