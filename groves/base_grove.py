"""
Base Grove implementation: A cluster of specialized trees for a specific modality or domain.
Manages internal routing and knowledge sharing within the grove.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
from typing import Optional, List, Dict, Tuple


class SpecialistTree(nn.Module):
    """
    Enhanced tree with explicit specialization tracking.
    Builds on the base TreeExpert concept with specialization-specific heads.
    """
    def __init__(self, input_dim: int, hidden_dim: int, tree_id: int, 
                 specialization: str = "general", modality: str = "general"):
        super().__init__()
        self.id = tree_id
        self.age = 0
        self.bark = 0.0  # plasticity shield
        self.fitness = 5.0
        self.specialization = specialization
        self.modality = modality
        self.expertise_score = 0.0  # How good at specialization
        
        # Core architecture
        self.trunk = nn.Linear(input_dim, hidden_dim)
        self.act = nn.Tanh()
        
        # Specialization-specific head
        self.specialist_head = self._create_head(specialization, hidden_dim)
        
    def _create_head(self, spec: str, hidden_dim: int) -> nn.Module:
        """Create task-specific head based on specialization."""
        # Dictionary of common task heads
        heads = {
            # Vision tasks
            "classification": nn.Linear(hidden_dim, 1),
            "object_detection": nn.Linear(hidden_dim, 1),
            "segmentation": nn.Linear(hidden_dim, 1),
            
            # Audio tasks
            "transcription": nn.Linear(hidden_dim, 1),
            "genre": nn.Linear(hidden_dim, 1),
            "emotion": nn.Linear(hidden_dim, 1),
            
            # Text tasks
            "sentiment": nn.Linear(hidden_dim, 1),
            "ner": nn.Linear(hidden_dim, 1),
            "summarization": nn.Linear(hidden_dim, 1),
            
            # General
            "general": nn.Linear(hidden_dim, 1),
        }
        return heads.get(spec, nn.Linear(hidden_dim, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through trunk and specialist head."""
        features = self.act(self.trunk(x))
        return self.specialist_head(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features for knowledge transfer."""
        return self.act(self.trunk(x))
    
    def step_age(self):
        """Age the tree and potentially increase bark (protection)."""
        self.age += 1
        if self.age > 80:
            self.bark = min(0.985, self.bark + 0.01)
    
    def update_fitness(self, loss_value: float):
        """Update fitness score based on performance."""
        reward = 1.0 / (float(loss_value) + 1e-4)
        self.fitness = 0.97 * self.fitness + 0.03 * reward
    
    def update_expertise(self, task_performance: float):
        """Update expertise score for the specialization."""
        self.expertise_score = 0.95 * self.expertise_score + 0.05 * task_performance


class LocalCanopy(nn.Module):
    """
    Local routing within a grove.
    Routes inputs to the most appropriate trees within the grove.
    """
    def __init__(self, max_trees: int, input_dim: int = 512, hidden: int = 64):
        super().__init__()
        self.max_trees = max_trees
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max_trees)
        )
    
    def forward(self, x: torch.Tensor, num_trees: int) -> torch.Tensor:
        """
        Compute routing scores for each tree in the grove.
        
        Args:
            x: Input tensor [B, input_dim]
            num_trees: Current number of active trees
            
        Returns:
            scores: [B, num_trees] routing scores
        """
        scores = self.net(x)[:, :num_trees]
        return scores


class GroveMulch:
    """
    Memory system specific to a grove.
    Stores experiences relevant to the grove's specialization.
    """
    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.data = deque(maxlen=capacity)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def add(self, x: torch.Tensor, y: torch.Tensor, priority: float = 1.0):
        """Add experience to grove memory."""
        self.data.append((
            x.detach().cpu(),
            y.detach().cpu(),
            float(priority)
        ))
    
    def sample(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample batch from memory."""
        if len(self.data) < batch_size:
            return None, None
        
        import random
        batch = random.sample(self.data, batch_size)
        xs = torch.stack([item[0] for item in batch])
        ys = torch.stack([item[1] for item in batch])
        return xs, ys


class Grove(nn.Module):
    """
    A cluster of specialized trees for a specific modality or domain.
    Manages internal routing and knowledge sharing within the grove.
    """
    def __init__(self, modality: str, input_dim: int = 512, hidden_dim: int = 64, max_trees: int = 12):
        super().__init__()
        self.modality = modality
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_trees = max_trees
        
        self.trees = nn.ModuleList()
        self.local_router = LocalCanopy(max_trees, input_dim=input_dim)
        self.grove_memory = GroveMulch(capacity=2000)
        
        # Track inter-tree connections for knowledge sharing
        self.mycelium_connections = defaultdict(list)
        
        self.tree_counter = 0
    
    def num_trees(self) -> int:
        """Return number of trees in this grove."""
        return len(self.trees)
    
    def plant_specialist(self, specialization: str) -> Optional[int]:
        """
        Grow a new tree with specific expertise.
        
        Args:
            specialization: Type of task this tree specializes in
            
        Returns:
            tree_id if planted, None if grove is full
        """
        if self.num_trees() >= self.max_trees:
            return None
        
        tree = SpecialistTree(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            tree_id=self.tree_counter,
            specialization=specialization,
            modality=self.modality
        )
        self.trees.append(tree)
        self.tree_counter += 1
        
        # Connect to similar trees for knowledge sharing
        self._connect_to_similar_trees(tree)
        
        return tree.id
    
    def _connect_to_similar_trees(self, new_tree: SpecialistTree):
        """
        Establish mycelium connections between the new tree and existing trees
        with similar specializations.
        """
        for existing_tree in self.trees:
            if existing_tree.id != new_tree.id:
                # Connect trees with same specialization more strongly
                if existing_tree.specialization == new_tree.specialization:
                    strength = 1.0
                else:
                    strength = 0.3
                
                self.mycelium_connections[new_tree.id].append(
                    (existing_tree.id, strength)
                )
                self.mycelium_connections[existing_tree.id].append(
                    (new_tree.id, strength)
                )
    
    def forward(self, x: torch.Tensor, top_k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route input to best-matching trees in this grove.
        
        Args:
            x: Input tensor [B, input_dim]
            top_k: Number of trees to activate
            
        Returns:
            combined: Weighted combination of tree outputs [B, 1]
            weights: Routing weights [B, num_trees]
        """
        if self.num_trees() == 0:
            raise ValueError(f"Grove {self.modality} has no trees")
        
        # Get routing scores
        scores = self.local_router(x, num_trees=self.num_trees())
        
        # Apply top-k routing
        weights = self._topk_softmax(scores, k=min(top_k, self.num_trees()))
        
        # Gather outputs from all trees
        outputs = []
        for tree in self.trees:
            outputs.append(tree(x))
        outputs = torch.stack(outputs, dim=-1)  # [B, 1, num_trees]
        
        # Weighted combination
        weights_expanded = weights.unsqueeze(1)  # [B, 1, num_trees]
        combined = (outputs * weights_expanded).sum(dim=-1)  # [B, 1]
        
        return combined, weights
    
    def _topk_softmax(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """
        Apply softmax only to top-k scores, set others to 0.
        
        Args:
            scores: [B, num_trees]
            k: Number of top trees to keep
            
        Returns:
            weights: [B, num_trees] with non-top-k set to 0
        """
        B, T = scores.shape
        k = min(k, T)
        
        topv, topi = torch.topk(scores, k=k, dim=1)
        w = torch.softmax(topv, dim=1)  # [B, k]
        
        weights = torch.zeros_like(scores)
        weights.scatter_(1, topi, w)
        
        return weights
    
    def get_grove_stats(self) -> Dict:
        """Return statistics about the grove."""
        return {
            "modality": self.modality,
            "num_trees": self.num_trees(),
            "max_trees": self.max_trees,
            "memory_size": len(self.grove_memory),
            "trees": [
                {
                    "id": tree.id,
                    "age": tree.age,
                    "fitness": tree.fitness,
                    "specialization": tree.specialization,
                    "expertise": tree.expertise_score,
                    "bark": tree.bark,
                }
                for tree in self.trees
            ]
        }
