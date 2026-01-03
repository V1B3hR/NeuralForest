"""
Tests for Phase 3: Evolution and Legacy Management
Tests the TreeGraveyard, archival, and resurrection mechanisms.
"""

import pytest
import torch
import tempfile
from pathlib import Path
import json

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NeuralForest import ForestEcosystem, TreeExpert, TreeArch, set_seed
from evolution import TreeGraveyard, TreeRecord


class TestTreeGraveyard:
    """Test suite for TreeGraveyard functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        set_seed(42)
        self.graveyard = TreeGraveyard(
            max_records=100,
            save_weights=False,
            auto_save=False,
        )
    
    def test_graveyard_initialization(self):
        """Test that graveyard initializes correctly."""
        assert len(self.graveyard.records) == 0
        assert len(self.graveyard.records_by_id) == 0
        assert self.graveyard.stats.total_eliminated == 0
    
    def test_archive_tree(self):
        """Test archiving a tree."""
        # Create a simple tree
        arch = TreeArch(
            num_layers=2,
            hidden_dim=32,
            activation="relu",
            dropout=0.1,
            normalization="layer",
            residual=True
        )
        tree = TreeExpert(input_dim=4, tree_id=0, arch=arch)
        tree.fitness = 5.5
        tree.age = 10
        tree.bark = 1.2
        
        # Archive the tree
        record = self.graveyard.archive_tree(
            tree=tree,
            elimination_reason="low_fitness",
            generation=5,
            parent_ids=[1, 2],
            children_ids=[3, 4]
        )
        
        # Verify record
        assert record.tree_id == 0
        assert record.final_fitness == 5.5
        assert record.age_at_elimination == 10
        assert record.elimination_reason == "low_fitness"
        assert record.generation == 5
        assert record.parent_ids == [1, 2]
        assert record.children_ids == [3, 4]
        
        # Verify graveyard state
        assert len(self.graveyard.records) == 1
        assert 0 in self.graveyard.records_by_id
        assert self.graveyard.stats.total_eliminated == 1
    
    def test_query_by_reason(self):
        """Test querying records by elimination reason."""
        # Create and archive multiple trees
        for i in range(5):
            arch = TreeArch(num_layers=1, hidden_dim=16, activation="relu",
                          dropout=0.0, normalization="none", residual=False)
            tree = TreeExpert(input_dim=4, tree_id=i, arch=arch)
            tree.fitness = float(i)
            
            reason = "low_fitness" if i < 3 else "old_age"
            self.graveyard.archive_tree(tree, elimination_reason=reason, generation=0)
        
        # Query by reason
        low_fitness_records = self.graveyard.query_by_reason("low_fitness")
        old_age_records = self.graveyard.query_by_reason("old_age")
        
        assert len(low_fitness_records) == 3
        assert len(old_age_records) == 2
    
    def test_query_by_generation(self):
        """Test querying records by generation."""
        # Create trees in different generations
        for gen in [0, 0, 1, 1, 1, 2]:
            arch = TreeArch(num_layers=1, hidden_dim=16, activation="relu",
                          dropout=0.0, normalization="none", residual=False)
            tree = TreeExpert(input_dim=4, tree_id=gen*10, arch=arch)
            self.graveyard.archive_tree(tree, elimination_reason="test", generation=gen)
        
        gen0 = self.graveyard.query_by_generation(0)
        gen1 = self.graveyard.query_by_generation(1)
        gen2 = self.graveyard.query_by_generation(2)
        
        assert len(gen0) == 2
        assert len(gen1) == 3
        assert len(gen2) == 1
    
    def test_query_by_fitness_range(self):
        """Test querying records by fitness range."""
        # Create trees with different fitness values
        for i, fitness in enumerate([1.5, 3.0, 5.5, 7.0, 9.0]):
            arch = TreeArch(num_layers=1, hidden_dim=16, activation="relu",
                          dropout=0.0, normalization="none", residual=False)
            tree = TreeExpert(input_dim=4, tree_id=i, arch=arch)
            tree.fitness = fitness
            self.graveyard.archive_tree(tree, elimination_reason="test", generation=0)
        
        # Query different ranges
        low = self.graveyard.query_by_fitness_range(0.0, 4.0)
        mid = self.graveyard.query_by_fitness_range(4.0, 8.0)
        high = self.graveyard.query_by_fitness_range(8.0, 10.0)
        
        assert len(low) == 2  # 1.5, 3.0
        assert len(mid) == 2  # 5.5, 7.0
        assert len(high) == 1  # 9.0
    
    def test_resurrection_candidates(self):
        """Test getting resurrection candidates."""
        # Create trees with different fitness
        for i in range(5):
            arch = TreeArch(num_layers=1, hidden_dim=16, activation="relu",
                          dropout=0.0, normalization="none", residual=False)
            tree = TreeExpert(input_dim=4, tree_id=i, arch=arch)
            tree.fitness = float(i * 2)  # 0, 2, 4, 6, 8
            tree.age = i + 1
            self.graveyard.archive_tree(tree, elimination_reason="test", generation=0)
        
        # Get candidates with min_fitness filter
        candidates = self.graveyard.get_resurrection_candidates(min_fitness=5.0)
        
        assert len(candidates) == 2  # fitness 6 and 8
        assert candidates[0].final_fitness == 8.0  # Should be sorted by fitness desc
        assert candidates[1].final_fitness == 6.0
    
    def test_analyze_elimination_patterns(self):
        """Test elimination pattern analysis."""
        # Create diverse tree population
        for i in range(10):
            arch = TreeArch(num_layers=1, hidden_dim=16, activation="relu",
                          dropout=0.0, normalization="none", residual=False)
            tree = TreeExpert(input_dim=4, tree_id=i, arch=arch)
            tree.fitness = float(i + 1)
            tree.age = i * 2
            self.graveyard.archive_tree(tree, elimination_reason="test", generation=0)
        
        analysis = self.graveyard.analyze_elimination_patterns()
        
        assert analysis['total_records'] == 10
        assert 'fitness_stats' in analysis
        assert 'age_stats' in analysis
        assert analysis['fitness_stats']['mean'] == pytest.approx(5.5, 0.1)
        assert analysis['age_stats']['min'] == 0
        assert analysis['age_stats']['max'] == 18
    
    def test_identify_dead_ends(self):
        """Test identifying architectural dead-ends."""
        # Create trees with same architecture but poor fitness
        arch1 = TreeArch(num_layers=1, hidden_dim=8, activation="tanh",
                        dropout=0.0, normalization="none", residual=False)
        arch2 = TreeArch(num_layers=3, hidden_dim=128, activation="gelu",
                        dropout=0.3, normalization="batch", residual=True)
        
        # Archive multiple trees with arch1 (low fitness)
        for i in range(4):
            tree = TreeExpert(input_dim=4, tree_id=i, arch=arch1)
            tree.fitness = 1.5 + i * 0.1  # All low fitness
            self.graveyard.archive_tree(tree, elimination_reason="low_fitness", generation=0)
        
        # Archive one tree with arch2 (high fitness)
        tree = TreeExpert(input_dim=4, tree_id=10, arch=arch2)
        tree.fitness = 8.0
        self.graveyard.archive_tree(tree, elimination_reason="old_age", generation=0)
        
        dead_ends = self.graveyard.identify_dead_ends(threshold=2.5)
        
        assert len(dead_ends) >= 1
        assert dead_ends[0]['count'] >= 3
        assert dead_ends[0]['avg_fitness'] < 2.5
    
    def test_save_and_load(self):
        """Test saving and loading graveyard to/from disk."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_graveyard.json"
            
            # Create graveyard with data
            graveyard = TreeGraveyard(save_path=save_path, auto_save=False)
            
            for i in range(3):
                arch = TreeArch(num_layers=1, hidden_dim=16, activation="relu",
                              dropout=0.0, normalization="none", residual=False)
                tree = TreeExpert(input_dim=4, tree_id=i, arch=arch)
                tree.fitness = float(i + 1)
                graveyard.archive_tree(tree, elimination_reason="test", generation=0)
            
            # Save to disk
            graveyard.save_to_disk()
            
            assert save_path.exists()
            
            # Create new graveyard and load
            graveyard2 = TreeGraveyard(save_path=save_path, auto_save=False)
            graveyard2.load_from_disk()
            
            # Verify loaded data
            assert len(graveyard2.records) == 3
            assert graveyard2.stats.total_eliminated == 3


class TestForestGraveyardIntegration:
    """Test TreeGraveyard integration with ForestEcosystem."""
    
    def setup_method(self):
        """Setup for each test method."""
        set_seed(43)
    
    def test_forest_with_graveyard(self):
        """Test that forest initializes with graveyard."""
        forest = ForestEcosystem(input_dim=4, hidden_dim=32, max_trees=10, 
                                enable_graveyard=True)
        
        assert forest.graveyard is not None
        assert forest.enable_graveyard is True
    
    def test_forest_without_graveyard(self):
        """Test that forest can be created without graveyard."""
        forest = ForestEcosystem(input_dim=4, hidden_dim=32, max_trees=10, 
                                enable_graveyard=False)
        
        assert forest.graveyard is None
        assert forest.enable_graveyard is False
    
    def test_prune_archives_trees(self):
        """Test that pruning archives trees to graveyard."""
        forest = ForestEcosystem(input_dim=4, hidden_dim=32, max_trees=10, 
                                enable_graveyard=True)
        
        # Plant more trees
        for _ in range(5):
            forest._plant_tree()
        
        initial_count = forest.num_trees()
        assert initial_count == 6
        
        # Assign fitness
        for tree in forest.trees:
            tree.fitness = torch.rand(1).item() * 10
        
        # Get weak trees
        weak_trees = sorted(forest.trees, key=lambda t: t.fitness)[:2]
        weak_ids = [t.id for t in weak_trees]
        
        # Prune
        forest._prune_trees(weak_ids, reason="test_elimination")
        
        # Verify trees removed from forest
        assert forest.num_trees() == initial_count - 2
        
        # Verify trees archived in graveyard
        assert len(forest.graveyard.records) == 2
        for wid in weak_ids:
            assert wid in forest.graveyard.records_by_id
    
    def test_resurrection_integration(self):
        """Test tree resurrection in forest context."""
        forest = ForestEcosystem(input_dim=4, hidden_dim=32, max_trees=10, 
                                enable_graveyard=True)
        
        # Plant trees
        for _ in range(4):
            forest._plant_tree()
        
        # Assign fitness
        for tree in forest.trees:
            tree.fitness = torch.rand(1).item() * 10 + 3.0  # Ensure some good fitness
        
        # Prune one tree
        weak_tree = min(forest.trees, key=lambda t: t.fitness)
        weak_id = weak_tree.id
        weak_fitness = weak_tree.fitness
        
        forest._prune_trees([weak_id], reason="test")
        
        initial_count = forest.num_trees()
        
        # Resurrect
        resurrected = forest.resurrect_tree(tree_id=weak_id)
        
        assert resurrected is not None
        assert forest.num_trees() == initial_count + 1
        assert resurrected.id != weak_id  # Should have new ID
        assert forest.graveyard.stats.resurrection_count == 1
    
    def test_generation_tracking(self):
        """Test that generation is tracked correctly."""
        forest = ForestEcosystem(input_dim=4, hidden_dim=32, max_trees=10, 
                                enable_graveyard=True)
        
        # Plant and prune across generations
        for gen in range(3):
            forest.current_generation = gen
            
            # Plant trees
            for _ in range(2):
                forest._plant_tree()
            
            # Assign fitness and prune
            for tree in forest.trees:
                tree.fitness = torch.rand(1).item() * 10
            
            if forest.num_trees() > 3:
                weak = min(forest.trees, key=lambda t: t.fitness)
                forest._prune_trees([weak.id], reason="generation_test")
        
        # Check graveyard has records from different generations
        records = list(forest.graveyard.records)
        generations = {r.generation for r in records}
        
        assert len(generations) >= 2  # Should have multiple generations


class TestTreeRecord:
    """Test TreeRecord dataclass functionality."""
    
    def test_record_creation(self):
        """Test creating a tree record."""
        record = TreeRecord(
            tree_id=5,
            timestamp=123456.0,
            architecture={'num_layers': 2, 'hidden_dim': 32},
            num_parameters=1000,
            final_fitness=7.5,
            age_at_elimination=15,
            elimination_reason="low_fitness",
            generation=10,
        )
        
        assert record.tree_id == 5
        assert record.final_fitness == 7.5
        assert record.age_at_elimination == 15
    
    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        record = TreeRecord(
            tree_id=5,
            timestamp=123456.0,
            architecture={'num_layers': 2},
            num_parameters=1000,
            final_fitness=7.5,
            age_at_elimination=15,
            elimination_reason="test",
            generation=10,
        )
        
        record_dict = record.to_dict()
        
        assert isinstance(record_dict, dict)
        assert record_dict['tree_id'] == 5
        assert record_dict['final_fitness'] == 7.5
    
    def test_record_from_dict(self):
        """Test creating record from dictionary."""
        data = {
            'tree_id': 5,
            'timestamp': 123456.0,
            'architecture': {'num_layers': 2},
            'num_parameters': 1000,
            'final_fitness': 7.5,
            'fitness_history': [5.0, 6.0, 7.5],
            'age_at_elimination': 15,
            'bark_at_elimination': 1.0,
            'elimination_reason': 'test',
            'generation': 10,
            'recent_disruptions': [],
            'resource_allocation_history': [],
            'parent_ids': [],
            'children_ids': [],
            'weights_path': None,
        }
        
        record = TreeRecord.from_dict(data)
        
        assert record.tree_id == 5
        assert record.final_fitness == 7.5
        assert record.age_at_elimination == 15


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
