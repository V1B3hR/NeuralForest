"""
Advanced Genealogy Tracking and Visualization for NeuralForest.

Tracks family trees of evolved neural networks, enabling visualization
and analysis of evolutionary lineages.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
import logging

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    nx = None
    plt = None

logger = logging.getLogger(__name__)


@dataclass
class TreeLineage:
    """Represents a single tree's lineage information."""
    
    tree_id: int
    generation: int
    parent_ids: List[int] = field(default_factory=list)
    children_ids: List[int] = field(default_factory=list)
    
    # Architecture info
    architecture: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    birth_fitness: float = 0.0
    peak_fitness: float = 0.0
    final_fitness: float = 0.0
    age_at_death: Optional[int] = None
    
    # Evolutionary origin
    creation_method: str = "random"  # random, mutation, crossover, resurrection
    mutation_type: Optional[str] = None  # if created by mutation
    
    # Status
    is_alive: bool = True
    elimination_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TreeLineage:
        """Create from dictionary."""
        return cls(**data)


class GenealogyTracker:
    """
    Tracks and manages the genealogy of all trees in the forest.
    
    Features:
    - Family tree construction
    - Lineage analysis
    - Evolutionary path tracking
    - Ancestry queries
    """
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize genealogy tracker.
        
        Args:
            save_dir: Directory to save genealogy data
        """
        self.lineages: Dict[int, TreeLineage] = {}
        self.generation_map: Dict[int, List[int]] = defaultdict(list)
        self.save_dir = Path(save_dir) if save_dir else None
        self.current_generation = 0
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def register_tree(
        self,
        tree_id: int,
        generation: int,
        parent_ids: Optional[List[int]] = None,
        architecture: Optional[Dict[str, Any]] = None,
        creation_method: str = "random",
        mutation_type: Optional[str] = None,
        birth_fitness: float = 0.0,
    ) -> TreeLineage:
        """
        Register a new tree in the genealogy system.
        
        Args:
            tree_id: Unique tree identifier
            generation: Generation number
            parent_ids: IDs of parent trees
            architecture: Tree architecture details
            creation_method: How the tree was created
            mutation_type: Type of mutation if applicable
            birth_fitness: Initial fitness
            
        Returns:
            TreeLineage object
        """
        parent_ids = parent_ids or []
        architecture = architecture or {}
        
        lineage = TreeLineage(
            tree_id=tree_id,
            generation=generation,
            parent_ids=parent_ids,
            architecture=architecture,
            creation_method=creation_method,
            mutation_type=mutation_type,
            birth_fitness=birth_fitness,
            peak_fitness=birth_fitness,
            final_fitness=birth_fitness,
        )
        
        self.lineages[tree_id] = lineage
        self.generation_map[generation].append(tree_id)
        self.current_generation = max(self.current_generation, generation)
        
        # Update parent trees' children lists
        for parent_id in parent_ids:
            if parent_id in self.lineages:
                self.lineages[parent_id].children_ids.append(tree_id)
        
        logger.debug(f"Registered tree {tree_id} in generation {generation}")
        return lineage
    
    def update_fitness(self, tree_id: int, fitness: float):
        """
        Update fitness metrics for a tree.
        
        Args:
            tree_id: Tree identifier
            fitness: New fitness value
        """
        if tree_id in self.lineages:
            lineage = self.lineages[tree_id]
            lineage.final_fitness = fitness
            lineage.peak_fitness = max(lineage.peak_fitness, fitness)
    
    def mark_eliminated(
        self,
        tree_id: int,
        age: int,
        reason: str = "low_fitness"
    ):
        """
        Mark a tree as eliminated.
        
        Args:
            tree_id: Tree identifier
            age: Age at elimination
            reason: Elimination reason
        """
        if tree_id in self.lineages:
            lineage = self.lineages[tree_id]
            lineage.is_alive = False
            lineage.age_at_death = age
            lineage.elimination_reason = reason
    
    def get_ancestors(self, tree_id: int, max_depth: Optional[int] = None) -> List[int]:
        """
        Get all ancestors of a tree.
        
        Args:
            tree_id: Tree identifier
            max_depth: Maximum depth to search (None for unlimited)
            
        Returns:
            List of ancestor tree IDs
        """
        ancestors = []
        visited = set()
        queue = deque([(tree_id, 0)])
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited or current_id not in self.lineages:
                continue
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            visited.add(current_id)
            lineage = self.lineages[current_id]
            
            for parent_id in lineage.parent_ids:
                if parent_id not in visited:
                    ancestors.append(parent_id)
                    queue.append((parent_id, depth + 1))
        
        return ancestors
    
    def get_descendants(self, tree_id: int, max_depth: Optional[int] = None) -> List[int]:
        """
        Get all descendants of a tree.
        
        Args:
            tree_id: Tree identifier
            max_depth: Maximum depth to search (None for unlimited)
            
        Returns:
            List of descendant tree IDs
        """
        descendants = []
        visited = set()
        queue = deque([(tree_id, 0)])
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in visited or current_id not in self.lineages:
                continue
            
            if max_depth is not None and depth >= max_depth:
                continue
            
            visited.add(current_id)
            lineage = self.lineages[current_id]
            
            for child_id in lineage.children_ids:
                if child_id not in visited:
                    descendants.append(child_id)
                    queue.append((child_id, depth + 1))
        
        return descendants
    
    def get_family_tree(self, tree_id: int) -> Dict[str, List[int]]:
        """
        Get complete family tree for a tree.
        
        Args:
            tree_id: Tree identifier
            
        Returns:
            Dictionary with ancestors, descendants, and siblings
        """
        lineage = self.lineages.get(tree_id)
        if not lineage:
            return {"ancestors": [], "descendants": [], "siblings": []}
        
        ancestors = self.get_ancestors(tree_id)
        descendants = self.get_descendants(tree_id)
        
        # Find siblings (trees with same parents)
        siblings = []
        for other_id, other_lineage in self.lineages.items():
            if other_id != tree_id:
                if set(lineage.parent_ids) & set(other_lineage.parent_ids):
                    siblings.append(other_id)
        
        return {
            "ancestors": ancestors,
            "descendants": descendants,
            "siblings": siblings,
        }
    
    def get_lineage_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all lineages.
        
        Returns:
            Dictionary of statistics
        """
        total_trees = len(self.lineages)
        alive_trees = sum(1 for l in self.lineages.values() if l.is_alive)
        dead_trees = total_trees - alive_trees
        
        # Fitness statistics
        peak_fitnesses = [l.peak_fitness for l in self.lineages.values()]
        avg_peak_fitness = sum(peak_fitnesses) / len(peak_fitnesses) if peak_fitnesses else 0
        
        # Age statistics
        dead_ages = [l.age_at_death for l in self.lineages.values() if l.age_at_death is not None]
        avg_age_at_death = sum(dead_ages) / len(dead_ages) if dead_ages else 0
        
        # Creation methods
        creation_methods = defaultdict(int)
        for lineage in self.lineages.values():
            creation_methods[lineage.creation_method] += 1
        
        # Elimination reasons
        elimination_reasons = defaultdict(int)
        for lineage in self.lineages.values():
            if lineage.elimination_reason:
                elimination_reasons[lineage.elimination_reason] += 1
        
        return {
            "total_trees": total_trees,
            "alive_trees": alive_trees,
            "dead_trees": dead_trees,
            "avg_peak_fitness": avg_peak_fitness,
            "avg_age_at_death": avg_age_at_death,
            "creation_methods": dict(creation_methods),
            "elimination_reasons": dict(elimination_reasons),
            "total_generations": self.current_generation,
        }
    
    def find_most_successful_lineage(self) -> Optional[Tuple[int, List[int]]]:
        """
        Find the most successful lineage based on fitness.
        
        Returns:
            Tuple of (root_tree_id, lineage_tree_ids) or None
        """
        if not self.lineages:
            return None
        
        # Find tree with highest peak fitness
        best_tree_id = max(self.lineages.keys(), key=lambda tid: self.lineages[tid].peak_fitness)
        
        # Get all ancestors
        ancestors = self.get_ancestors(best_tree_id)
        lineage = [best_tree_id] + ancestors
        
        # Find the root (oldest ancestor)
        root_id = best_tree_id
        for ancestor_id in ancestors:
            ancestor_lineage = self.lineages[ancestor_id]
            current_lineage = self.lineages[root_id]
            if ancestor_lineage.generation < current_lineage.generation:
                root_id = ancestor_id
        
        return root_id, lineage
    
    def export_genealogy_graph(self, output_path: Optional[Path] = None) -> Optional[str]:
        """
        Export genealogy as a graph structure.
        
        Args:
            output_path: Path to save graph data (JSON format)
            
        Returns:
            JSON string of graph data
        """
        nodes = []
        edges = []
        
        for tree_id, lineage in self.lineages.items():
            nodes.append({
                "id": tree_id,
                "generation": lineage.generation,
                "fitness": lineage.peak_fitness,
                "is_alive": lineage.is_alive,
                "creation_method": lineage.creation_method,
            })
            
            for parent_id in lineage.parent_ids:
                edges.append({
                    "source": parent_id,
                    "target": tree_id,
                })
        
        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "metadata": self.get_lineage_statistics(),
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
        
        return json.dumps(graph_data, indent=2)
    
    def visualize_family_tree(
        self,
        tree_id: int,
        output_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        Visualize the family tree of a specific tree.
        
        Args:
            tree_id: Tree identifier
            output_path: Path to save visualization
            show: Whether to display the plot
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available. Install networkx and matplotlib.")
            return
        
        family = self.get_family_tree(tree_id)
        all_family_ids = [tree_id] + family["ancestors"] + family["descendants"]
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for tid in all_family_ids:
            if tid in self.lineages:
                lineage = self.lineages[tid]
                G.add_node(
                    tid,
                    generation=lineage.generation,
                    fitness=lineage.peak_fitness,
                    alive=lineage.is_alive
                )
        
        # Add edges
        for tid in all_family_ids:
            if tid in self.lineages:
                lineage = self.lineages[tid]
                for parent_id in lineage.parent_ids:
                    if parent_id in all_family_ids:
                        G.add_edge(parent_id, tid)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Layout
        try:
            pos = nx.spring_layout(G, k=2, iterations=50)
        except:
            pos = nx.random_layout(G)
        
        # Node colors based on fitness
        node_colors = []
        for node in G.nodes():
            fitness = G.nodes[node].get('fitness', 0)
            alive = G.nodes[node].get('alive', False)
            if node == tree_id:
                node_colors.append('gold')  # Highlight target tree
            elif alive:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightcoral')
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title(f"Family Tree for Tree {tree_id}")
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_full_genealogy(
        self,
        output_path: Optional[Path] = None,
        show: bool = True,
        max_trees: int = 50
    ):
        """
        Visualize the complete genealogy of the forest.
        
        Args:
            output_path: Path to save visualization
            show: Whether to display the plot
            max_trees: Maximum number of trees to visualize
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available. Install networkx and matplotlib.")
            return
        
        # Limit number of trees for readability
        tree_ids = list(self.lineages.keys())[:max_trees]
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for tid in tree_ids:
            lineage = self.lineages[tid]
            G.add_node(
                tid,
                generation=lineage.generation,
                fitness=lineage.peak_fitness,
                alive=lineage.is_alive
            )
            
            for parent_id in lineage.parent_ids:
                if parent_id in tree_ids:
                    G.add_edge(parent_id, tid)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Use hierarchical layout by generation
        pos = {}
        generation_counts = defaultdict(int)
        
        for node in G.nodes():
            gen = G.nodes[node]['generation']
            x = generation_counts[gen]
            y = -gen  # Negative so newer generations are at bottom
            pos[node] = (x, y)
            generation_counts[gen] += 1
        
        # Node colors
        node_colors = []
        for node in G.nodes():
            alive = G.nodes[node].get('alive', False)
            if alive:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightcoral')
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
        
        ax.set_title(f"Forest Genealogy ({len(tree_ids)} trees)")
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save(self, path: Optional[Path] = None):
        """
        Save genealogy data to disk.
        
        Args:
            path: Path to save file
        """
        save_path = path or (self.save_dir / "genealogy.json" if self.save_dir else None)
        
        if not save_path:
            logger.warning("No save path specified")
            return
        
        data = {
            "lineages": {tid: lineage.to_dict() for tid, lineage in self.lineages.items()},
            "generation_map": {gen: ids for gen, ids in self.generation_map.items()},
            "current_generation": self.current_generation,
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Genealogy saved to {save_path}")
    
    def load(self, path: Path):
        """
        Load genealogy data from disk.
        
        Args:
            path: Path to load from
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.lineages = {
            int(tid): TreeLineage.from_dict(lineage_data)
            for tid, lineage_data in data["lineages"].items()
        }
        
        self.generation_map = defaultdict(list)
        for gen_str, ids in data["generation_map"].items():
            self.generation_map[int(gen_str)] = ids
        
        self.current_generation = data["current_generation"]
        
        logger.info(f"Genealogy loaded from {path}")
