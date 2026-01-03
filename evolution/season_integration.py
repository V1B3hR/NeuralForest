"""
Integration between Evolution System and Seasonal Cycles.

This module connects the evolutionary mechanisms (NAS, mutation, crossover)
with the seasonal training cycles, allowing evolutionary operations to adapt
based on the current season.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class SeasonalEvolution:
    """
    Coordinates evolutionary operations with seasonal cycles.
    
    Different seasons influence:
    - Mutation rates and strategies
    - Crossover probability
    - Selection pressure
    - Architecture search parameters
    """
    
    def __init__(self, base_mutation_rate: float = 0.1, base_crossover_prob: float = 0.3):
        """
        Initialize seasonal evolution controller.
        
        Args:
            base_mutation_rate: Base mutation rate (adjusted by season)
            base_crossover_prob: Base crossover probability (adjusted by season)
        """
        self.base_mutation_rate = base_mutation_rate
        self.base_crossover_prob = base_crossover_prob
        self.evolution_history = []
        
    def get_season_modifiers(self, season: str) -> Dict[str, float]:
        """
        Get season-specific modifiers for evolutionary operations.
        
        Args:
            season: Current season name (spring, summer, autumn, winter)
            
        Returns:
            Dictionary of modifier values for different evolutionary parameters
        """
        modifiers = {
            "spring": {
                "mutation_rate_multiplier": 1.5,  # High mutation for exploration
                "crossover_prob_multiplier": 1.3,  # Encourage mixing
                "selection_pressure": 0.6,  # Lower pressure, keep more diversity
                "architecture_exploration": 1.4,  # Try more architectures
                "novelty_bonus": 1.5,  # Reward novel architectures
                "parent_selection_diversity": 0.8,  # Prefer diverse parents
                "elite_preservation": 0.1,  # Few elites preserved
            },
            "summer": {
                "mutation_rate_multiplier": 1.0,  # Normal mutation
                "crossover_prob_multiplier": 1.0,  # Normal crossover
                "selection_pressure": 0.8,  # Moderate pressure
                "architecture_exploration": 1.0,  # Balanced exploration
                "novelty_bonus": 1.0,  # Balanced novelty reward
                "parent_selection_diversity": 0.5,  # Balanced diversity
                "elite_preservation": 0.15,  # Moderate elite preservation
            },
            "autumn": {
                "mutation_rate_multiplier": 0.7,  # Less mutation
                "crossover_prob_multiplier": 0.8,  # Less crossover
                "selection_pressure": 1.2,  # Higher pressure, stronger selection
                "architecture_exploration": 0.7,  # Less exploration
                "novelty_bonus": 0.7,  # Less novelty reward
                "parent_selection_diversity": 0.3,  # Prefer fit parents
                "elite_preservation": 0.25,  # More elites preserved
            },
            "winter": {
                "mutation_rate_multiplier": 0.5,  # Minimal mutation
                "crossover_prob_multiplier": 0.6,  # Reduced crossover
                "selection_pressure": 1.0,  # Moderate pressure
                "architecture_exploration": 0.5,  # Minimal exploration
                "novelty_bonus": 0.5,  # Minimal novelty reward
                "parent_selection_diversity": 0.2,  # Prefer proven parents
                "elite_preservation": 0.3,  # Maximum elite preservation
            },
        }
        
        return modifiers.get(season, modifiers["summer"])
    
    def get_evolutionary_params(self, season: str) -> Dict[str, Any]:
        """
        Get complete evolutionary parameters adjusted for current season.
        
        Args:
            season: Current season name
            
        Returns:
            Dictionary of evolutionary parameters
        """
        modifiers = self.get_season_modifiers(season)
        
        return {
            "mutation_rate": self.base_mutation_rate * modifiers["mutation_rate_multiplier"],
            "crossover_prob": self.base_crossover_prob * modifiers["crossover_prob_multiplier"],
            "selection_pressure": modifiers["selection_pressure"],
            "architecture_exploration": modifiers["architecture_exploration"],
            "novelty_bonus": modifiers["novelty_bonus"],
            "parent_selection_diversity": modifiers["parent_selection_diversity"],
            "elite_preservation": modifiers["elite_preservation"],
            "season": season,
        }
    
    def should_trigger_evolution(self, season: str, season_progress: float) -> bool:
        """
        Determine if evolutionary operations should be triggered based on season.
        
        Args:
            season: Current season name
            season_progress: Progress within season (0.0 to 1.0)
            
        Returns:
            True if evolution should be triggered
        """
        # Evolution triggers based on season
        triggers = {
            "spring": season_progress > 0.2,  # After initial growth
            "summer": season_progress > 0.5,  # Mid-season evaluation
            "autumn": season_progress > 0.7,  # Before winter consolidation
            "winter": season_progress > 0.8,  # Rarely in winter
        }
        
        return triggers.get(season, False)
    
    def get_nas_parameters(self, season: str) -> Dict[str, Any]:
        """
        Get Neural Architecture Search parameters adjusted for season.
        
        Args:
            season: Current season name
            
        Returns:
            Dictionary of NAS parameters
        """
        modifiers = self.get_season_modifiers(season)
        
        params = {
            "spring": {
                "population_size": 20,  # Large population for exploration
                "num_generations": 5,  # Many generations
                "search_space_diversity": 1.5,  # Wide search space
                "early_stopping_patience": 10,  # Patient with new architectures
                "validation_frequency": 5,  # Less frequent validation
            },
            "summer": {
                "population_size": 15,  # Moderate population
                "num_generations": 7,  # Balanced generations
                "search_space_diversity": 1.0,  # Balanced search space
                "early_stopping_patience": 7,  # Moderate patience
                "validation_frequency": 3,  # Balanced validation
            },
            "autumn": {
                "population_size": 10,  # Smaller population
                "num_generations": 5,  # Fewer generations
                "search_space_diversity": 0.7,  # Narrower search
                "early_stopping_patience": 5,  # Less patient
                "validation_frequency": 2,  # More frequent validation
            },
            "winter": {
                "population_size": 8,  # Small population
                "num_generations": 3,  # Few generations
                "search_space_diversity": 0.5,  # Very narrow search
                "early_stopping_patience": 3,  # Impatient
                "validation_frequency": 1,  # Constant validation
            },
        }
        
        return params.get(season, params["summer"])
    
    def record_evolution_event(self, season: str, event_type: str, details: Dict[str, Any]):
        """
        Record an evolutionary event with seasonal context.
        
        Args:
            season: Current season
            event_type: Type of evolutionary event
            details: Event details
        """
        event = {
            "season": season,
            "event_type": event_type,
            "timestamp": len(self.evolution_history),
            **details,
        }
        self.evolution_history.append(event)
        
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about evolutionary operations across seasons.
        
        Returns:
            Dictionary of statistics
        """
        if not self.evolution_history:
            return {
                "total_events": 0,
                "events_by_season": {},
                "events_by_type": {},
            }
        
        stats = {
            "total_events": len(self.evolution_history),
            "events_by_season": {},
            "events_by_type": {},
        }
        
        for event in self.evolution_history:
            season = event["season"]
            event_type = event["event_type"]
            
            stats["events_by_season"][season] = stats["events_by_season"].get(season, 0) + 1
            stats["events_by_type"][event_type] = stats["events_by_type"].get(event_type, 0) + 1
        
        return stats
    
    def get_recommendations(self, season: str, forest_state: Dict[str, Any]) -> List[str]:
        """
        Get season-specific recommendations for evolutionary operations.
        
        Args:
            season: Current season
            forest_state: Current state of the forest (trees, fitness, etc.)
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        modifiers = self.get_season_modifiers(season)
        
        num_trees = forest_state.get("num_trees", 0)
        avg_fitness = forest_state.get("avg_fitness", 0)
        diversity = forest_state.get("diversity", 0)
        
        if season == "spring":
            if num_trees < 5:
                recommendations.append("🌱 Spring: Plant more diverse trees to expand population")
            if diversity < 0.5:
                recommendations.append("🌱 Spring: Increase mutation rate to boost diversity")
            recommendations.append("🌱 Spring: High exploration mode - try novel architectures")
            
        elif season == "summer":
            if avg_fitness < 5.0:
                recommendations.append("☀️ Summer: Focus on training existing trees for better fitness")
            recommendations.append("☀️ Summer: Balanced evolution - refine successful architectures")
            
        elif season == "autumn":
            if num_trees > 10:
                recommendations.append("🍂 Autumn: Consider pruning weak trees before winter")
            if diversity > 0.8:
                recommendations.append("🍂 Autumn: Reduce mutation to consolidate successful patterns")
            recommendations.append("🍂 Autumn: Strong selection - preserve only fit trees")
            
        elif season == "winter":
            recommendations.append("❄️ Winter: Minimal evolution - focus on knowledge consolidation")
            if avg_fitness > 8.0:
                recommendations.append("❄️ Winter: Preserve elite architectures for next year")
            recommendations.append("❄️ Winter: Prepare for spring rebirth with archived knowledge")
        
        return recommendations


def integrate_season_with_nas(seasonal_cycle, nas_instance) -> Dict[str, Any]:
    """
    Helper function to integrate seasonal parameters into NAS.
    
    Args:
        seasonal_cycle: SeasonalCycle instance
        nas_instance: NAS instance to configure
        
    Returns:
        Dictionary of applied parameters
    """
    season = seasonal_cycle.current_season
    seasonal_evo = SeasonalEvolution()
    
    nas_params = seasonal_evo.get_nas_parameters(season)
    evo_params = seasonal_evo.get_evolutionary_params(season)
    
    logger.info(f"Integrating season '{season}' with NAS: {nas_params}")
    
    return {
        "season": season,
        "nas_params": nas_params,
        "evolution_params": evo_params,
    }
