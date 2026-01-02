"""
Text Grove: Specialized tree cluster for text/NLP tasks.
Handles text-related tasks like sentiment analysis, NER, summarization, etc.
"""

import torch
from typing import List
from .base_grove import Grove


class TextGrove(Grove):
    """
    A grove specialized for text/NLP tasks.
    Trees in this grove handle various natural language processing tasks.
    """
    
    # Text task specializations
    SPECIALIZATIONS = [
        "sentiment",
        "ner",
        "summarization",
        "classification",
        "question_answering",
        "intent_detection",
        "entity_extraction",
    ]
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 64, max_trees: int = 12):
        """
        Initialize Text Grove.
        
        Args:
            input_dim: Dimension of input features from TextSoil
            hidden_dim: Hidden dimension for tree networks
            max_trees: Maximum number of trees in this grove
        """
        super().__init__(
            modality="text",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            max_trees=max_trees
        )
        
        # Plant initial specialist trees
        self._plant_initial_specialists()
    
    def _plant_initial_specialists(self):
        """Plant a few initial specialist trees for common text tasks."""
        initial_specialists = ["sentiment", "ner", "summarization"]
        
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
