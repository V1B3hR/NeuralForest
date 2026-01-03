"""
Tree Graveyard / Legacy Repository for NeuralForest Phase 3b

This module implements the memory and archival system for eliminated trees,
enabling post-mortem analysis, resurrection, and evolutionary insights.

Key Features:
- Archive eliminated trees with full metadata (architecture, weights, history, genealogy)
- Post-mortem analysis of failed/eliminated trees
- Resurrection mechanism to reintroduce archived trees
- Knowledge repository for evolutionary insights
"""

from __future__ import annotations

import json
import time
from collections import deque, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class TreeRecord:
    """Complete record of an eliminated tree."""
    
    # Identity
    tree_id: int
    timestamp: float
    
    # Architecture
    architecture: Dict[str, Any]
    num_parameters: int
    
    # Performance metrics
    final_fitness: float
    fitness_history: List[float] = field(default_factory=list)
    age_at_elimination: int = 0
    bark_at_elimination: float = 0.0
    
    # Elimination context
    elimination_reason: str = "unknown"
    generation: int = 0
    
    # Environmental context
    recent_disruptions: List[Dict[str, Any]] = field(default_factory=list)
    resource_allocation_history: List[float] = field(default_factory=list)
    
    # Genealogy
    parent_ids: List[int] = field(default_factory=list)
    children_ids: List[int] = field(default_factory=list)
    
    # Optional: saved weights path
    weights_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TreeRecord:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GraveyardStats:
    """Statistics about the tree graveyard."""
    
    total_eliminated: int = 0
    avg_age_at_elimination: float = 0.0
    avg_fitness_at_elimination: float = 0.0
    elimination_reasons: Dict[str, int] = field(default_factory=dict)
    avg_parameters: float = 0.0
    resurrection_count: int = 0


class TreeGraveyard:
    """
    Repository for eliminated trees with full archival and analysis capabilities.
    
    Features:
    - Store complete tree records before elimination
    - Query and analyze eliminated trees
    - Resurrection mechanism for reintroducing trees
    - Post-mortem analysis utilities
    - Evolutionary insights extraction
    """
    
    def __init__(
        self,
        max_records: int = 10000,
        save_weights: bool = False,
        weights_dir: Optional[Path] = None,
        auto_save: bool = True,
        save_path: Optional[Path] = None,
    ):
        """
        Initialize the Tree Graveyard.
        
        Args:
            max_records: Maximum number of records to keep in memory
            save_weights: Whether to save model weights for eliminated trees
            weights_dir: Directory to save weights (if save_weights=True)
            auto_save: Automatically save graveyard to disk periodically
            save_path: Path to save graveyard JSON records
        """
        self.max_records = max_records
        self.save_weights = save_weights
        self.weights_dir = Path(weights_dir) if weights_dir else Path("/tmp/graveyard_weights")
        self.auto_save = auto_save
        self.save_path = Path(save_path) if save_path else Path("/tmp/tree_graveyard.json")
        
        # Storage
        self.records: deque[TreeRecord] = deque(maxlen=max_records)
        self.records_by_id: Dict[int, TreeRecord] = {}
        
        # Indexes for fast queries
        self.records_by_reason: Dict[str, List[TreeRecord]] = defaultdict(list)
        self.records_by_generation: Dict[int, List[TreeRecord]] = defaultdict(list)
        
        # Statistics
        self.stats = GraveyardStats()
        
        # Create directories
        if self.save_weights:
            self.weights_dir.mkdir(parents=True, exist_ok=True)
        if self.auto_save:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
    
    def archive_tree(
        self,
        tree,
        elimination_reason: str = "low_fitness",
        generation: int = 0,
        recent_disruptions: Optional[List[Dict[str, Any]]] = None,
        resource_history: Optional[List[float]] = None,
        parent_ids: Optional[List[int]] = None,
        children_ids: Optional[List[int]] = None,
    ) -> TreeRecord:
        """
        Archive a tree before elimination.
        
        Args:
            tree: The tree object to archive
            elimination_reason: Why this tree is being eliminated
            generation: Current generation number
            recent_disruptions: Recent environmental disruptions affecting this tree
            resource_history: History of resource allocation for this tree
            parent_ids: IDs of parent trees (for genealogy)
            children_ids: IDs of child trees (for genealogy)
        
        Returns:
            TreeRecord: The created archive record
        """
        # Extract tree metadata
        tree_id = getattr(tree, 'id', -1)
        age = getattr(tree, 'age', 0)
        bark = getattr(tree, 'bark', 0.0)
        fitness = getattr(tree, 'fitness', 0.0)
        fitness_history = getattr(tree, 'fitness_history', [fitness])
        
        # Extract architecture
        if hasattr(tree, 'arch'):
            arch = tree.arch
            architecture = arch.to_dict() if hasattr(arch, 'to_dict') else asdict(arch)
        else:
            # Fallback: try to extract from tree attributes
            architecture = {
                'num_layers': getattr(tree, 'num_layers', 0),
                'hidden_dim': getattr(tree, 'hidden_dim', 0),
                'activation': getattr(tree, 'activation', 'unknown'),
                'dropout': getattr(tree, 'dropout', 0.0),
                'normalization': getattr(tree, 'normalization', 'none'),
                'residual': getattr(tree, 'residual', False),
            }
        
        # Count parameters
        num_params = sum(p.numel() for p in tree.parameters()) if hasattr(tree, 'parameters') else 0
        
        # Save weights if requested
        weights_path = None
        if self.save_weights:
            weights_path = str(self.weights_dir / f"tree_{tree_id}_gen_{generation}.pt")
            try:
                torch.save(tree.state_dict(), weights_path)
                logger.info(f"Saved weights for tree {tree_id} to {weights_path}")
            except Exception as e:
                logger.warning(f"Failed to save weights for tree {tree_id}: {e}")
                weights_path = None
        
        # Create record
        record = TreeRecord(
            tree_id=tree_id,
            timestamp=time.time(),
            architecture=architecture,
            num_parameters=num_params,
            final_fitness=fitness,
            fitness_history=fitness_history if isinstance(fitness_history, list) else [fitness],
            age_at_elimination=age,
            bark_at_elimination=bark,
            elimination_reason=elimination_reason,
            generation=generation,
            recent_disruptions=recent_disruptions or [],
            resource_allocation_history=resource_history or [],
            parent_ids=parent_ids or [],
            children_ids=children_ids or [],
            weights_path=weights_path,
        )
        
        # Store record
        self.records.append(record)
        self.records_by_id[tree_id] = record
        self.records_by_reason[elimination_reason].append(record)
        self.records_by_generation[generation].append(record)
        
        # Update statistics
        self._update_stats(record)
        
        # Auto-save if enabled
        if self.auto_save and len(self.records) % 100 == 0:
            self.save_to_disk()
        
        logger.info(
            f"Archived tree {tree_id}: reason={elimination_reason}, "
            f"fitness={fitness:.3f}, age={age}, gen={generation}"
        )
        
        return record
    
    def get_record(self, tree_id: int) -> Optional[TreeRecord]:
        """Get a specific tree record by ID."""
        return self.records_by_id.get(tree_id)
    
    def query_by_reason(self, reason: str) -> List[TreeRecord]:
        """Query records by elimination reason."""
        return self.records_by_reason.get(reason, [])
    
    def query_by_generation(self, generation: int) -> List[TreeRecord]:
        """Query records by generation."""
        return self.records_by_generation.get(generation, [])
    
    def query_by_fitness_range(
        self,
        min_fitness: float,
        max_fitness: float
    ) -> List[TreeRecord]:
        """Query records within a fitness range."""
        return [
            r for r in self.records
            if min_fitness <= r.final_fitness <= max_fitness
        ]
    
    def get_resurrection_candidates(
        self,
        min_fitness: Optional[float] = None,
        max_age: Optional[int] = None,
        exclude_reasons: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[TreeRecord]:
        """
        Get candidate trees for resurrection.
        
        Args:
            min_fitness: Minimum fitness threshold
            max_age: Maximum age at elimination
            exclude_reasons: Elimination reasons to exclude
            limit: Maximum number of candidates to return
        
        Returns:
            List of TreeRecord sorted by fitness (descending)
        """
        candidates = []
        exclude_set = set(exclude_reasons or [])
        
        for record in self.records:
            # Apply filters
            if min_fitness and record.final_fitness < min_fitness:
                continue
            if max_age and record.age_at_elimination > max_age:
                continue
            if record.elimination_reason in exclude_set:
                continue
            
            candidates.append(record)
        
        # Sort by fitness (descending)
        candidates.sort(key=lambda r: r.final_fitness, reverse=True)
        
        return candidates[:limit]
    
    def resurrect_tree(
        self,
        record: TreeRecord,
        tree_class,
        input_dim: int,
        new_tree_id: int,
    ):
        """
        Resurrect a tree from the graveyard.
        
        Args:
            record: The TreeRecord to resurrect
            tree_class: The tree class to instantiate (e.g., TreeExpert)
            input_dim: Input dimension for the new tree
            new_tree_id: ID for the resurrected tree
        
        Returns:
            The resurrected tree instance
        """
        # Create architecture object
        try:
            # Try to import the canonical TreeArch
            from NeuralForest import TreeArch
        except ImportError:
            # Fallback to dataclass
            from dataclasses import dataclass
            
            @dataclass(frozen=True)
            class TreeArch:
                num_layers: int
                hidden_dim: int
                activation: str
                dropout: float
                normalization: str
                residual: bool
                
                def to_dict(self):
                    return asdict(self)
        
        arch = TreeArch(**record.architecture)
        
        # Create new tree with the archived architecture
        tree = tree_class(input_dim, new_tree_id, arch)
        
        # Load weights if available
        if record.weights_path and Path(record.weights_path).exists():
            try:
                tree.load_state_dict(torch.load(record.weights_path))
                logger.info(f"Loaded weights for resurrected tree {new_tree_id} from {record.weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load weights for resurrection: {e}")
        
        # Update resurrection counter
        self.stats.resurrection_count += 1
        
        logger.info(
            f"Resurrected tree {record.tree_id} as new tree {new_tree_id} "
            f"(original fitness={record.final_fitness:.3f})"
        )
        
        return tree
    
    def analyze_elimination_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in tree elimination.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.records:
            return {
                'total_records': 0,
                'message': 'No records available for analysis'
            }
        
        # Collect data
        fitnesses = [r.final_fitness for r in self.records]
        ages = [r.age_at_elimination for r in self.records]
        params = [r.num_parameters for r in self.records]
        
        # Architecture distribution
        arch_counts = defaultdict(int)
        for r in self.records:
            arch_key = (
                r.architecture.get('num_layers', 0),
                r.architecture.get('hidden_dim', 0),
                r.architecture.get('activation', 'unknown')
            )
            arch_counts[arch_key] += 1
        
        # Fitness trajectory analysis
        improving_trees = 0
        declining_trees = 0
        for r in self.records:
            if len(r.fitness_history) >= 2:
                if r.fitness_history[-1] > r.fitness_history[0]:
                    improving_trees += 1
                else:
                    declining_trees += 1
        
        return {
            'total_records': len(self.records),
            'fitness_stats': {
                'mean': sum(fitnesses) / len(fitnesses),
                'min': min(fitnesses),
                'max': max(fitnesses),
            },
            'age_stats': {
                'mean': sum(ages) / len(ages),
                'min': min(ages),
                'max': max(ages),
            },
            'parameter_stats': {
                'mean': sum(params) / len(params),
                'min': min(params),
                'max': max(params),
            },
            'elimination_reasons': dict(self.stats.elimination_reasons),
            'architecture_distribution': {
                str(k): v for k, v in arch_counts.items()
            },
            'trajectory_analysis': {
                'improving_at_elimination': improving_trees,
                'declining_at_elimination': declining_trees,
            },
            'resurrection_count': self.stats.resurrection_count,
        }
    
    def identify_dead_ends(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Identify architectural dead-ends (patterns that consistently fail).
        
        Args:
            threshold: Fitness threshold below which architectures are considered failed
        
        Returns:
            List of architectural patterns with low success rates
        """
        arch_performance = defaultdict(lambda: {'count': 0, 'total_fitness': 0.0, 'records': []})
        
        for record in self.records:
            arch_key = json.dumps(record.architecture, sort_keys=True)
            arch_performance[arch_key]['count'] += 1
            arch_performance[arch_key]['total_fitness'] += record.final_fitness
            arch_performance[arch_key]['records'].append(record)
        
        dead_ends = []
        for arch_key, perf in arch_performance.items():
            avg_fitness = perf['total_fitness'] / perf['count']
            if avg_fitness < threshold and perf['count'] >= 3:
                dead_ends.append({
                    'architecture': json.loads(arch_key),
                    'count': perf['count'],
                    'avg_fitness': avg_fitness,
                    'sample_tree_ids': [r.tree_id for r in perf['records'][:5]],
                })
        
        # Sort by count (most common dead-ends first)
        dead_ends.sort(key=lambda x: x['count'], reverse=True)
        
        return dead_ends
    
    def get_successful_patterns(self, threshold: float = 5.0) -> List[Dict[str, Any]]:
        """
        Identify architectural patterns that were successful before elimination.
        
        Args:
            threshold: Fitness threshold above which architectures are considered successful
        
        Returns:
            List of successful architectural patterns
        """
        arch_performance = defaultdict(lambda: {'count': 0, 'total_fitness': 0.0, 'records': []})
        
        for record in self.records:
            if record.final_fitness >= threshold:
                arch_key = json.dumps(record.architecture, sort_keys=True)
                arch_performance[arch_key]['count'] += 1
                arch_performance[arch_key]['total_fitness'] += record.final_fitness
                arch_performance[arch_key]['records'].append(record)
        
        successful = []
        for arch_key, perf in arch_performance.items():
            if perf['count'] > 0:
                avg_fitness = perf['total_fitness'] / perf['count']
                successful.append({
                    'architecture': json.loads(arch_key),
                    'count': perf['count'],
                    'avg_fitness': avg_fitness,
                    'sample_tree_ids': [r.tree_id for r in perf['records'][:5]],
                })
        
        # Sort by average fitness (best first)
        successful.sort(key=lambda x: x['avg_fitness'], reverse=True)
        
        return successful
    
    def save_to_disk(self, path: Optional[Path] = None) -> None:
        """Save graveyard records to disk."""
        save_path = path or self.save_path
        
        data = {
            'records': [r.to_dict() for r in self.records],
            'stats': asdict(self.stats),
            'metadata': {
                'total_archived': len(self.records),
                'timestamp': time.time(),
            }
        }
        
        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.records)} graveyard records to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save graveyard to disk: {e}")
    
    def load_from_disk(self, path: Optional[Path] = None) -> None:
        """Load graveyard records from disk."""
        load_path = path or self.save_path
        
        if not load_path.exists():
            logger.warning(f"Graveyard file not found: {load_path}")
            return
        
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            # Clear current records
            self.records.clear()
            self.records_by_id.clear()
            self.records_by_reason.clear()
            self.records_by_generation.clear()
            
            # Load records
            for record_dict in data['records']:
                record = TreeRecord.from_dict(record_dict)
                self.records.append(record)
                self.records_by_id[record.tree_id] = record
                self.records_by_reason[record.elimination_reason].append(record)
                self.records_by_generation[record.generation].append(record)
            
            # Load stats
            if 'stats' in data:
                self.stats = GraveyardStats(**data['stats'])
            
            logger.info(f"Loaded {len(self.records)} graveyard records from {load_path}")
        except Exception as e:
            logger.error(f"Failed to load graveyard from disk: {e}")
    
    def _update_stats(self, record: TreeRecord) -> None:
        """Update aggregate statistics."""
        self.stats.total_eliminated += 1
        
        # Update running averages
        n = self.stats.total_eliminated
        self.stats.avg_age_at_elimination = (
            (self.stats.avg_age_at_elimination * (n - 1) + record.age_at_elimination) / n
        )
        self.stats.avg_fitness_at_elimination = (
            (self.stats.avg_fitness_at_elimination * (n - 1) + record.final_fitness) / n
        )
        self.stats.avg_parameters = (
            (self.stats.avg_parameters * (n - 1) + record.num_parameters) / n
        )
        
        # Update elimination reasons count
        reason = record.elimination_reason
        self.stats.elimination_reasons[reason] = self.stats.elimination_reasons.get(reason, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current graveyard statistics."""
        return asdict(self.stats)
    
    def clear(self) -> None:
        """Clear all records (use with caution!)."""
        self.records.clear()
        self.records_by_id.clear()
        self.records_by_reason.clear()
        self.records_by_generation.clear()
        self.stats = GraveyardStats()
        logger.warning("Graveyard cleared - all records removed")
