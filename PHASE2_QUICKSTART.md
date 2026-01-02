# Phase 2 Ecosystem Quick Start Guide

This guide helps you get started with the Phase 2 ecosystem simulation features.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Demo

Run the comprehensive demo showcasing all Phase 2 features:

```bash
python phase2_ecosystem_demo.py
```

This will demonstrate:
1. Resource competition among trees
2. Drought robustness testing
3. Flood robustness testing
4. Full ecosystem simulation
5. Fitness-based selection and pruning
6. Dynamic tree planting
7. Statistics tracking

## Basic Usage

### 1. Create an Ecosystem Simulator

```python
from ecosystem_simulation import EcosystemSimulator
from NeuralForest import ForestEcosystem, DEVICE

# Create a forest
forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=10).to(DEVICE)

# Create simulator with configuration
simulator = EcosystemSimulator(
    forest,
    competition_fairness=0.3,  # Balance between fitness and fairness
    selection_threshold=0.3     # Prune bottom 30% of trees
)
```

### 2. Simulate a Generation

```python
import torch

# Prepare data
batch_x = torch.randn(100, 2).to(DEVICE)
batch_y = torch.randn(100, 1).to(DEVICE)

# Run simulation (normal conditions)
stats = simulator.simulate_generation(batch_x, batch_y)

print(f"Generation {stats.generation}")
print(f"Trees: {stats.num_trees}")
print(f"Avg fitness: {stats.avg_fitness:.2f}")
```

### 3. Test with Environmental Disruptions

```python
# Simulate drought (data scarcity)
stats = simulator.simulate_generation(
    batch_x, batch_y,
    disruption_type="drought",
    disruption_severity=0.5
)

# Simulate flood (data corruption/noise)
stats = simulator.simulate_generation(
    batch_x, batch_y,
    disruption_type="flood",
    disruption_severity=0.5
)
```

### 4. Apply Selection Pressure

```python
# Prune weak trees
pruned_count = simulator.prune_weak_trees(min_keep=3)
print(f"Removed {pruned_count} weak trees")

# Plant new trees
planted_count = simulator.plant_trees(count=2)
print(f"Planted {planted_count} new trees")
```

### 5. Get Statistics

```python
# Get current ecosystem summary
summary = simulator.get_summary()
print(f"Total trees: {summary['total_trees']}")
print(f"Fitness: {summary['current_fitness']}")
print(f"Diversity: {summary['architecture_diversity']}")

# Get full history
history = simulator.get_stats_history()
for gen_stats in history:
    print(f"Gen {gen_stats['generation']}: {gen_stats['avg_fitness']:.2f}")
```

## Running Tests

Run the comprehensive test suite:

```bash
python tests/test_ecosystem.py
```

Expected output: 19 tests, 100% pass rate

## Configuration Options

### EcosystemSimulator Parameters

- `competition_fairness` (float, 0.0-1.0): Balance between fitness-based (0.0) and equal (1.0) allocation
- `selection_threshold` (float, 0.0-1.0): Percentile below which trees are pruned
- `max_history` (int): Maximum generations to keep in history

### Disruption Severities

- **Drought**: 0.0 = no drought, 1.0 = 95% data removed
- **Flood**: 0.0 = clean data, 1.0 = high noise

## Architecture Overview

```
ecosystem_simulation.py
├── CompetitionSystem      # Resource allocation
├── RobustnessTester       # Environmental disruptions
├── EcosystemStats        # Metrics tracking
└── EcosystemSimulator    # Main orchestrator
```

## Key Metrics Tracked

For each generation, the following metrics are tracked:

- **Population**: Number of trees, average age
- **Fitness**: Mean, min, max, standard deviation
- **Diversity**: Unique architectures
- **Resources**: Total data allocated, competition events
- **Selection**: Trees pruned/planted, selection rate
- **Environment**: Disruption type and severity

## Example: Multi-Generation Evolution

```python
import torch
from ecosystem_simulation import EcosystemSimulator
from NeuralForest import ForestEcosystem, DEVICE

# Setup
forest = ForestEcosystem(input_dim=2, hidden_dim=32, max_trees=12).to(DEVICE)
simulator = EcosystemSimulator(forest)

# Evolve for 10 generations
for generation in range(10):
    # Generate data
    batch_x = torch.randn(100, 2).to(DEVICE)
    batch_y = torch.randn(100, 1).to(DEVICE)
    
    # Simulate generation
    stats = simulator.simulate_generation(batch_x, batch_y)
    
    # Update fitness (your training logic here)
    for tree in forest.trees:
        tree.fitness += torch.randn(1).item()
    
    # Apply selection every 3 generations
    if generation % 3 == 2:
        simulator.prune_weak_trees(min_keep=3)
        simulator.plant_trees(count=1)
    
    print(f"Gen {generation}: {stats.num_trees} trees, "
          f"fitness {stats.avg_fitness:.2f}")

# Final summary
summary = simulator.get_summary()
print(f"\nFinal state:")
print(f"  Trees: {summary['total_trees']}")
print(f"  Fitness: {summary['current_fitness']['avg']:.2f}")
print(f"  Recent pruned: {summary['recent_pruned']}")
print(f"  Recent planted: {summary['recent_planted']}")
```

## See Also

- `PHASE2_ECOSYSTEM_RESULTS.md`: Detailed implementation results
- `phase2_ecosystem_demo.py`: Comprehensive demonstrations
- `tests/test_ecosystem.py`: Test suite
- `roadmap2.md`: Full roadmap with Phase 2 details

## Support

For issues or questions, refer to the main repository documentation.
