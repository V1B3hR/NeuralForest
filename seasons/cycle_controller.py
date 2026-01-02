"""
Seasonal Cycle Controller for NeuralForest.

Manages training phases inspired by natural seasons, where each season
emphasizes different aspects of learning and forest management.
"""

from typing import Dict, List


class SeasonalCycle:
    """
    Manages training phases inspired by natural seasons.
    Each season emphasizes different aspects of learning.
    
    Seasons:
    - Spring: Growth phase - plant new trees, high exploration
    - Summer: Productivity phase - maximum learning, build expertise
    - Autumn: Pruning phase - remove weak trees, analyze fitness
    - Winter: Consolidation phase - strengthen memories, distillation
    """
    
    SEASONS = ["spring", "summer", "autumn", "winter"]
    
    def __init__(self, steps_per_season: int = 1000):
        """
        Initialize seasonal cycle controller.
        
        Args:
            steps_per_season: Number of training steps per season
        """
        self.steps_per_season = steps_per_season
        self.current_step = 0
        self.year = 0
        self._season_history = []
        
    @property
    def current_season(self) -> str:
        """Get the current season name."""
        season_idx = (self.current_step // self.steps_per_season) % 4
        return self.SEASONS[season_idx]
    
    @property
    def season_index(self) -> int:
        """Get current season index (0-3)."""
        return (self.current_step // self.steps_per_season) % 4
    
    @property
    def season_progress(self) -> float:
        """Progress within current season (0.0 to 1.0)."""
        return (self.current_step % self.steps_per_season) / self.steps_per_season
    
    @property
    def steps_remaining_in_season(self) -> int:
        """Number of steps remaining in current season."""
        return self.steps_per_season - (self.current_step % self.steps_per_season)
    
    def get_training_config(self) -> Dict:
        """
        Return season-appropriate hyperparameters and training configuration.
        
        Returns:
            Dictionary containing training hyperparameters for current season
        """
        configs = {
            "spring": {
                "description": "Growth phase - plant new trees, high exploration",
                "learning_rate": 0.03,
                "plasticity": 0.9,          # High plasticity
                "growth_probability": 0.3,   # Likely to plant new trees
                "prune_probability": 0.05,   # Rarely prune
                "replay_ratio": 0.5,         # Less replay, more new learning
                "distill_weight": 0.1,       # Low distillation
                "exploration_bonus": 0.2,    # Encourage trying new trees
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
            "summer": {
                "description": "Productivity phase - maximum learning, build expertise",
                "learning_rate": 0.02,
                "plasticity": 0.7,
                "growth_probability": 0.1,
                "prune_probability": 0.1,
                "replay_ratio": 1.0,
                "distill_weight": 0.2,
                "exploration_bonus": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
            "autumn": {
                "description": "Pruning phase - remove weak trees, analyze fitness",
                "learning_rate": 0.01,
                "plasticity": 0.5,
                "growth_probability": 0.02,
                "prune_probability": 0.25,   # Active pruning
                "replay_ratio": 1.2,
                "distill_weight": 0.3,
                "exploration_bonus": 0.05,
                "momentum": 0.85,
                "weight_decay": 2e-4,
            },
            "winter": {
                "description": "Consolidation phase - strengthen memories, distillation",
                "learning_rate": 0.005,
                "plasticity": 0.3,           # Low plasticity, protect knowledge
                "growth_probability": 0.01,
                "prune_probability": 0.02,
                "replay_ratio": 1.5,         # Heavy replay
                "distill_weight": 0.5,       # Strong distillation
                "exploration_bonus": 0.0,
                "momentum": 0.8,
                "weight_decay": 5e-4,
            },
        }
        return configs[self.current_season]
    
    def step(self) -> Dict:
        """
        Advance to next training step.
        
        Returns:
            Dictionary with season transition information
        """
        prev_season = self.current_season
        self.current_step += 1
        new_season = self.current_season
        
        # Check for year completion
        if self.current_step % (self.steps_per_season * 4) == 0:
            self.year += 1
            result = {
                "step": self.current_step,
                "season": new_season,
                "year_completed": True,
                "year": self.year,
                "message": f"🎊 Year {self.year} complete! Forest has matured."
            }
            self._season_history.append(result)
            return result
        
        # Check for season transition
        if prev_season != new_season:
            result = {
                "step": self.current_step,
                "season": new_season,
                "season_transition": True,
                "from_season": prev_season,
                "to_season": new_season,
                "message": self._get_season_emoji(new_season) + f" {new_season.capitalize()} begins!"
            }
            self._season_history.append(result)
            return result
        
        return {
            "step": self.current_step,
            "season": new_season,
            "season_transition": False,
        }
    
    def get_status(self) -> Dict:
        """
        Get comprehensive status of seasonal cycle.
        
        Returns:
            Dictionary with current cycle status
        """
        return {
            "current_step": self.current_step,
            "current_season": self.current_season,
            "season_progress": self.season_progress,
            "steps_remaining": self.steps_remaining_in_season,
            "year": self.year,
            "total_seasons_completed": self.current_step // self.steps_per_season,
            "config": self.get_training_config(),
        }
    
    def reset(self):
        """Reset the seasonal cycle to beginning."""
        self.current_step = 0
        self.year = 0
        self._season_history = []
    
    def get_season_history(self) -> List[Dict]:
        """Get history of major season events."""
        return self._season_history.copy()
    
    def should_plant_tree(self, random_val: float = None) -> bool:
        """
        Determine if a new tree should be planted based on current season.
        
        Args:
            random_val: Random value [0, 1] for probability check
                       If None, generates a random value internally
        
        Returns:
            True if a tree should be planted
        """
        config = self.get_training_config()
        if random_val is None:
            import random
            random_val = random.random()
        return random_val < config["growth_probability"]
    
    def should_prune_tree(self, random_val: float = None) -> bool:
        """
        Determine if trees should be pruned based on current season.
        
        Args:
            random_val: Random value [0, 1] for probability check
                       If None, generates a random value internally
        
        Returns:
            True if pruning should occur
        """
        config = self.get_training_config()
        if random_val is None:
            import random
            random_val = random.random()
        return random_val < config["prune_probability"]
    
    @staticmethod
    def _get_season_emoji(season: str) -> str:
        """Get emoji for season."""
        emojis = {
            "spring": "🌸",
            "summer": "☀️",
            "autumn": "🍂",
            "winter": "❄️"
        }
        return emojis.get(season, "🌲")
