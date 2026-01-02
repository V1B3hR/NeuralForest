"""
Neural Architecture Search for Trees

Automatically discovers optimal tree architectures using evolutionary strategies.
Trees can evolve different layer counts, dimensions, activations, and other
architectural choices to maximize performance.
"""

import random
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn


class TreeArchitectureSearch:
    """
    Automatically discovers optimal tree architectures.
    Uses evolutionary strategies to evolve tree structures.
    
    The search explores a space of architectural hyperparameters:
    - Number of layers
    - Hidden dimensions
    - Activation functions
    - Dropout rates
    - Normalization types
    - Residual connections
    """
    
    def __init__(self, forest):
        self.forest = forest
        self.search_space = {
            'num_layers': [2, 3, 4, 5, 6],
            'hidden_dim': [32, 64, 128, 256, 512],
            'activation': ['relu', 'gelu', 'tanh', 'swish'],
            'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
            'normalization': ['layer', 'batch', 'none'],
            'residual': [True, False],
        }
        self.population: List[Tuple[Dict, float]] = []
        self.hall_of_fame: List[Tuple[Dict, float]] = []
        self.generation = 0
    
    def random_architecture(self) -> Dict[str, Any]:
        """Generate a random architecture from the search space."""
        return {
            key: random.choice(values)
            for key, values in self.search_space.items()
        }
    
    def mutate(self, arch: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
        """
        Mutate an architecture by randomly changing some parameters.
        
        Args:
            arch: Architecture to mutate
            mutation_rate: Probability of mutating each parameter
            
        Returns:
            Mutated architecture
        """
        new_arch = arch.copy()
        for key in new_arch:
            if random.random() < mutation_rate:
                new_arch[key] = random.choice(self.search_space[key])
        return new_arch
    
    def crossover(self, arch1: Dict[str, Any], arch2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create child architecture by combining two parents.
        
        Args:
            arch1: First parent architecture
            arch2: Second parent architecture
            
        Returns:
            Child architecture with mixed parameters
        """
        child = {}
        for key in arch1:
            child[key] = random.choice([arch1[key], arch2[key]])
        return child
    
    def evaluate_architecture(
        self,
        arch: Dict[str, Any],
        eval_steps: int = 100,
        eval_samples: int = 100
    ) -> float:
        """
        Train a tree with this architecture and measure fitness.
        
        Note: This is a simplified evaluation. In a real system, you would
        train the tree on actual data and measure performance metrics.
        
        Args:
            arch: Architecture to evaluate
            eval_steps: Number of training steps
            eval_samples: Number of samples to use for evaluation
            
        Returns:
            Fitness score (higher is better)
        """
        # For demonstration, compute a heuristic fitness
        # In practice, you would train and evaluate a real tree
        
        fitness = 5.0  # Base fitness
        
        # Prefer moderate layer counts
        if 3 <= arch['num_layers'] <= 4:
            fitness += 1.0
        
        # Prefer reasonable hidden dimensions
        if 64 <= arch['hidden_dim'] <= 256:
            fitness += 1.0
        
        # Prefer modern activations
        if arch['activation'] in ['gelu', 'swish']:
            fitness += 0.5
        
        # Prefer moderate dropout
        if 0.1 <= arch['dropout'] <= 0.3:
            fitness += 0.5
        
        # Prefer normalization
        if arch['normalization'] != 'none':
            fitness += 0.5
        
        # Prefer residual connections for deeper networks
        if arch['residual'] and arch['num_layers'] >= 4:
            fitness += 1.0
        
        # Add some randomness to simulate real evaluation
        fitness += random.gauss(0, 0.5)
        
        return max(0.0, fitness)
    
    def search(
        self,
        generations: int = 20,
        population_size: int = 10,
        eval_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Run evolutionary architecture search.
        
        Args:
            generations: Number of generations to evolve
            population_size: Size of population
            eval_steps: Training steps per evaluation
            
        Returns:
            Best architecture found
        """
        print(f"🧬 Starting architecture search for {generations} generations...")
        
        # Initialize population
        self.population = [
            (self.random_architecture(), 0.0)
            for _ in range(population_size)
        ]
        
        for gen in range(generations):
            # Evaluate all architectures in population
            for i, (arch, _) in enumerate(self.population):
                fitness = self.evaluate_architecture(arch, eval_steps)
                self.population[i] = (arch, fitness)
            
            # Sort by fitness (highest first)
            self.population.sort(key=lambda x: x[1], reverse=True)
            
            # Keep best in hall of fame
            best_arch, best_fitness = self.population[0]
            self.hall_of_fame.append((best_arch.copy(), best_fitness))
            
            if (gen + 1) % 5 == 0 or gen == generations - 1:
                print(f"  Generation {gen + 1}/{generations}: Best fitness = {best_fitness:.4f}")
                print(f"    Architecture: {best_arch}")
            
            # Create next generation
            survivors = self.population[:population_size // 2]
            children = []
            
            while len(children) < population_size - len(survivors):
                # Select two parents
                parent1, parent2 = random.sample(survivors, 2)
                
                # Crossover and mutation
                child_arch = self.crossover(parent1[0], parent2[0])
                child_arch = self.mutate(child_arch)
                children.append((child_arch, 0.0))
            
            self.population = survivors + children
            self.generation = gen + 1
        
        # Return best architecture found
        best = max(self.hall_of_fame, key=lambda x: x[1])
        print(f"✅ Architecture search complete!")
        print(f"   Best fitness: {best[1]:.4f}")
        print(f"   Best architecture: {best[0]}")
        
        return best[0]
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """
        Get history of best architectures from each generation.
        
        Returns:
            List of dictionaries with generation, architecture, and fitness
        """
        return [
            {
                'generation': i,
                'architecture': arch,
                'fitness': fitness,
            }
            for i, (arch, fitness) in enumerate(self.hall_of_fame)
        ]
    
    def get_best_architecture(self) -> Tuple[Dict[str, Any], float]:
        """
        Get the best architecture found so far.
        
        Returns:
            Tuple of (architecture, fitness)
        """
        if not self.hall_of_fame:
            return self.random_architecture(), 0.0
        return max(self.hall_of_fame, key=lambda x: x[1])
    
    def save_search_results(self, path: str):
        """Save search history to file."""
        import json
        
        results = {
            'generations': self.generation,
            'search_space': self.search_space,
            'history': self.get_search_history(),
            'best_architecture': self.get_best_architecture()[0],
            'best_fitness': self.get_best_architecture()[1],
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Search results saved to {path}")
