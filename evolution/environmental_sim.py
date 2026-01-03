"""
Environmental Simulation System for NeuralForest Phase 6.

Implements dynamic environmental conditions including:
- Climate variations (temperature, humidity, seasons)
- Environmental stressors (drought, flood, disease, fire)
- Data availability changes
- Task distribution shifts
- Resource constraints
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ClimateType(Enum):
    """Types of climate conditions."""
    TEMPERATE = "temperate"  # Moderate, stable conditions
    TROPICAL = "tropical"    # High resources, high competition
    ARCTIC = "arctic"        # Low resources, harsh conditions
    DESERT = "desert"        # Scarce resources, extreme variations
    CHANGING = "changing"    # Unpredictable, rapid shifts


class StressorType(Enum):
    """Types of environmental stressors."""
    DROUGHT = "drought"      # Data scarcity
    FLOOD = "flood"          # Data overload with noise
    DISEASE = "disease"      # Corrupted or mislabeled data
    FIRE = "fire"            # Sudden catastrophic loss
    COMPETITION = "competition"  # Increased resource competition


@dataclass
class EnvironmentalState:
    """Current state of the environment."""
    
    climate: ClimateType
    temperature: float  # 0.0 (cold) to 1.0 (hot)
    resource_availability: float  # 0.0 (scarce) to 1.0 (abundant)
    data_quality: float  # 0.0 (poor) to 1.0 (excellent)
    competition_level: float  # 0.0 (low) to 1.0 (high)
    active_stressors: List[StressorType] = field(default_factory=list)
    time_step: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'climate': self.climate.value,
            'temperature': self.temperature,
            'resource_availability': self.resource_availability,
            'data_quality': self.data_quality,
            'competition_level': self.competition_level,
            'active_stressors': [s.value for s in self.active_stressors],
            'time_step': self.time_step,
        }


class EnvironmentalSimulator:
    """
    Simulates dynamic environmental conditions for the forest.
    
    Features:
    - Climate modeling with multiple types
    - Dynamic resource availability
    - Environmental stressors
    - Task distribution shifts
    - Adaptive difficulty
    """
    
    def __init__(
        self,
        initial_climate: ClimateType = ClimateType.TEMPERATE,
        stressor_probability: float = 0.1,
        climate_change_rate: float = 0.01
    ):
        """
        Initialize environmental simulator.
        
        Args:
            initial_climate: Starting climate type
            stressor_probability: Probability of stressor occurring each step
            climate_change_rate: Rate of gradual climate change
        """
        self.current_state = EnvironmentalState(
            climate=initial_climate,
            temperature=0.5,
            resource_availability=0.7,
            data_quality=0.9,
            competition_level=0.3,
        )
        
        self.stressor_probability = stressor_probability
        self.climate_change_rate = climate_change_rate
        self.history = []
        
        # Climate-specific base conditions
        self.climate_conditions = {
            ClimateType.TEMPERATE: {
                'temp_range': (0.4, 0.6),
                'resource_range': (0.6, 0.8),
                'quality_range': (0.8, 0.95),
            },
            ClimateType.TROPICAL: {
                'temp_range': (0.7, 0.9),
                'resource_range': (0.8, 1.0),
                'quality_range': (0.7, 0.9),
            },
            ClimateType.ARCTIC: {
                'temp_range': (0.0, 0.3),
                'resource_range': (0.2, 0.5),
                'quality_range': (0.6, 0.8),
            },
            ClimateType.DESERT: {
                'temp_range': (0.6, 1.0),
                'resource_range': (0.1, 0.4),
                'quality_range': (0.5, 0.7),
            },
            ClimateType.CHANGING: {
                'temp_range': (0.2, 0.8),
                'resource_range': (0.3, 0.9),
                'quality_range': (0.5, 0.95),
            },
        }
    
    def step(self) -> EnvironmentalState:
        """
        Advance environment by one time step.
        
        Returns:
            Updated environmental state
        """
        # Update time
        self.current_state.time_step += 1
        
        # Gradual climate evolution
        self._update_climate()
        
        # Update base conditions based on climate
        self._update_base_conditions()
        
        # Check for new stressors
        self._check_stressors()
        
        # Apply active stressors
        self._apply_stressors()
        
        # Add small random fluctuations
        self.current_state.temperature = np.clip(
            self.current_state.temperature + np.random.randn() * 0.05,
            0.0, 1.0
        )
        self.current_state.resource_availability = np.clip(
            self.current_state.resource_availability + np.random.randn() * 0.05,
            0.0, 1.0
        )
        
        # Save to history
        self.history.append(self.current_state.to_dict())
        
        return self.current_state
    
    def _update_climate(self) -> None:
        """Gradually evolve climate over time."""
        if np.random.rand() < self.climate_change_rate:
            # Potential climate shift
            if self.current_state.climate == ClimateType.CHANGING:
                # Randomly change to another climate
                new_climate = np.random.choice([
                    ClimateType.TEMPERATE,
                    ClimateType.TROPICAL,
                    ClimateType.ARCTIC,
                    ClimateType.DESERT,
                ])
                logger.info(f"Climate shifting from {self.current_state.climate.value} to {new_climate.value}")
                self.current_state.climate = new_climate
    
    def _update_base_conditions(self) -> None:
        """Update base environmental conditions based on current climate."""
        conditions = self.climate_conditions[self.current_state.climate]
        
        # Target values within climate ranges
        temp_target = np.random.uniform(*conditions['temp_range'])
        resource_target = np.random.uniform(*conditions['resource_range'])
        quality_target = np.random.uniform(*conditions['quality_range'])
        
        # Gradually move towards targets
        alpha = 0.1  # Smoothing factor
        self.current_state.temperature = (
            alpha * temp_target + (1 - alpha) * self.current_state.temperature
        )
        self.current_state.resource_availability = (
            alpha * resource_target + (1 - alpha) * self.current_state.resource_availability
        )
        self.current_state.data_quality = (
            alpha * quality_target + (1 - alpha) * self.current_state.data_quality
        )
    
    def _check_stressors(self) -> None:
        """Check if new stressors should be activated."""
        # Clear expired stressors (last for 5 steps)
        if self.current_state.time_step % 5 == 0:
            self.current_state.active_stressors = []
        
        # Check for new stressors
        if np.random.rand() < self.stressor_probability:
            new_stressor = np.random.choice(list(StressorType))
            self.current_state.active_stressors.append(new_stressor)
            logger.info(f"Environmental stressor activated: {new_stressor.value}")
    
    def _apply_stressors(self) -> None:
        """Apply effects of active stressors."""
        for stressor in self.current_state.active_stressors:
            if stressor == StressorType.DROUGHT:
                # Reduce resource availability
                self.current_state.resource_availability *= 0.5
            
            elif stressor == StressorType.FLOOD:
                # Reduce data quality but increase availability
                self.current_state.data_quality *= 0.6
                self.current_state.resource_availability = min(1.0, self.current_state.resource_availability * 1.5)
            
            elif stressor == StressorType.DISEASE:
                # Significantly reduce data quality
                self.current_state.data_quality *= 0.4
            
            elif stressor == StressorType.FIRE:
                # Catastrophic reduction in all resources
                self.current_state.resource_availability *= 0.2
                self.current_state.data_quality *= 0.5
            
            elif stressor == StressorType.COMPETITION:
                # Increase competition level
                self.current_state.competition_level = min(1.0, self.current_state.competition_level + 0.3)
    
    def apply_to_data(
        self,
        data_x: torch.Tensor,
        data_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply environmental effects to training data.
        
        Args:
            data_x: Input data
            data_y: Target data
            
        Returns:
            Modified (data_x, data_y) based on environmental conditions
        """
        modified_x = data_x.clone()
        modified_y = data_y.clone()
        
        # Apply resource scarcity (reduce batch size)
        if self.current_state.resource_availability < 0.5:
            keep_ratio = self.current_state.resource_availability
            keep_samples = max(1, int(len(data_x) * keep_ratio))
            indices = torch.randperm(len(data_x))[:keep_samples]
            modified_x = modified_x[indices]
            modified_y = modified_y[indices]
        
        # Apply data quality degradation
        if self.current_state.data_quality < 0.9:
            # Add noise proportional to quality degradation
            noise_level = (1.0 - self.current_state.data_quality) * 0.5
            noise = torch.randn_like(modified_x) * noise_level
            modified_x = modified_x + noise
            
            # Potentially corrupt labels
            if np.random.rand() < (1.0 - self.current_state.data_quality):
                # Corrupt a small fraction of labels
                corrupt_ratio = (1.0 - self.current_state.data_quality) * 0.1
                num_corrupt = max(1, int(len(modified_y) * corrupt_ratio))
                corrupt_indices = torch.randperm(len(modified_y))[:num_corrupt]
                modified_y[corrupt_indices] = torch.randn_like(modified_y[corrupt_indices])
        
        return modified_x, modified_y
    
    def set_climate(self, climate: ClimateType) -> None:
        """Manually set the climate type."""
        logger.info(f"Climate manually changed to {climate.value}")
        self.current_state.climate = climate
    
    def trigger_stressor(self, stressor: StressorType, duration: int = 5) -> None:
        """Manually trigger an environmental stressor."""
        logger.info(f"Environmental stressor triggered: {stressor.value}")
        self.current_state.active_stressors.append(stressor)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current environmental state."""
        return {
            **self.current_state.to_dict(),
            'severity': self._calculate_severity(),
        }
    
    def _calculate_severity(self) -> str:
        """Calculate overall environmental severity."""
        # Combine factors
        score = (
            (1.0 - self.current_state.resource_availability) * 0.4 +
            (1.0 - self.current_state.data_quality) * 0.3 +
            self.current_state.competition_level * 0.2 +
            len(self.current_state.active_stressors) * 0.1
        )
        
        if score < 0.3:
            return "mild"
        elif score < 0.6:
            return "moderate"
        else:
            return "severe"


class DataDistributionShift:
    """
    Simulates distribution shifts in data over time.
    
    Features:
    - Gradual drift
    - Sudden shifts
    - Cyclical patterns
    - Concept drift
    """
    
    def __init__(
        self,
        shift_type: str = 'gradual',
        shift_rate: float = 0.01
    ):
        """
        Initialize distribution shift simulator.
        
        Args:
            shift_type: Type of shift ('gradual', 'sudden', 'cyclical')
            shift_rate: Rate of distribution change
        """
        self.shift_type = shift_type
        self.shift_rate = shift_rate
        self.time_step = 0
        self.shift_history = []
    
    def apply_shift(
        self,
        data: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply distribution shift to data.
        
        Args:
            data: Input data
            targets: Optional target data
            
        Returns:
            Shifted data and targets
        """
        self.time_step += 1
        
        shifted_data = data.clone()
        shifted_targets = targets.clone() if targets is not None else None
        
        if self.shift_type == 'gradual':
            # Gradual drift over time
            drift = torch.randn_like(data) * self.shift_rate * self.time_step
            shifted_data = data + drift
        
        elif self.shift_type == 'sudden':
            # Sudden shift at random intervals
            if self.time_step % 100 == 0:
                shift = torch.randn_like(data) * 0.5
                shifted_data = data + shift
                self.shift_history.append({
                    'time': self.time_step,
                    'type': 'sudden',
                    'magnitude': 0.5,
                })
        
        elif self.shift_type == 'cyclical':
            # Cyclical pattern
            phase = (self.time_step * 2 * np.pi) / 100
            magnitude = np.sin(phase) * 0.3
            shift = torch.randn_like(data) * magnitude
            shifted_data = data + shift
        
        return shifted_data, shifted_targets
    
    def get_shift_info(self) -> Dict[str, Any]:
        """Get information about distribution shifts."""
        return {
            'shift_type': self.shift_type,
            'time_step': self.time_step,
            'num_shifts': len(self.shift_history),
            'recent_shifts': self.shift_history[-5:] if self.shift_history else [],
        }
