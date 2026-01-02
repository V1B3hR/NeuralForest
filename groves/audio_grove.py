"""
Audio Grove: Specialized tree cluster for audio/sound processing tasks.
Handles audio-related tasks like speech recognition, music classification, etc.
"""

from typing import List
from .base_grove import Grove


class AudioGrove(Grove):
    """
    A grove specialized for audio/sound processing tasks.
    Trees in this grove handle various audio analysis tasks.
    """

    # Audio task specializations
    SPECIALIZATIONS = [
        "transcription",
        "genre",
        "emotion",
        "speaker_recognition",
        "music_classification",
        "sound_event_detection",
        "rhythm_analysis",
    ]

    def __init__(self, input_dim: int = 512, hidden_dim: int = 64, max_trees: int = 12):
        """
        Initialize Audio Grove.

        Args:
            input_dim: Dimension of input features from AudioSoil
            hidden_dim: Hidden dimension for tree networks
            max_trees: Maximum number of trees in this grove
        """
        super().__init__(
            modality="audio",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_trees=max_trees,
        )

        # Plant initial specialist trees
        self._plant_initial_specialists()

    def _plant_initial_specialists(self):
        """Plant a few initial specialist trees for common audio tasks."""
        initial_specialists = ["transcription", "genre", "emotion"]

        for spec in initial_specialists:
            self.plant_specialist(spec)

    def get_suggested_specializations(self, task_performance: dict) -> List[str]:
        """
        Suggest new specializations based on task performance gaps.

        Args:
            task_performance: Dict mapping task names to performance scores

        Returns:
            List of suggested specializations to plant
        """
        suggestions = []

        # Check which specializations are missing
        current_specs = {tree.specialization for tree in self.trees}

        for spec in self.SPECIALIZATIONS:
            if spec not in current_specs:
                # Suggest if this specialization might be needed
                if spec in task_performance and task_performance[spec] < 0.7:
                    suggestions.append(spec)

        return suggestions
