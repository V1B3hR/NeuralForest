# NeuralForest

![NeuralForest banner](banner.png)

---

## 💖 Sponsoruj mnie

Jeśli projekt NeuralForest Ci się podoba i chcesz wesprzeć jego rozwój, możesz zostać moim sponsorem na [GitHub Sponsors](https://github.com/sponsors/V1B3hR)!

---

**NeuralForest v2** is an experimental neural network ecosystem combining tree experts, prioritized experience replay, coreset memory, gating/routing, drift detection, and visualization. 
It is designed for continual learning, with robustness against data drift and dynamic expert growth/pruning.

## ✨ New in Phase 5, 6 & 7

### Phase 5: Visualization and Monitoring ✅
- **Fitness & Diversity Charts** - Real-time visualization of evolution metrics
- **Architecture Distribution** - t-SNE/PCA visualization of tree architectures
- **Performance Heatmaps** - Tree performance across multiple dimensions
- **Species Tree Plots** - Genealogy and ancestry visualization
- **Evolution Dashboard** - Comprehensive multi-panel monitoring

### Phase 6: Cooperation & Environmental Adaptation ✅
- **Tree Cooperation** - Communication channels and message passing
- **Federated Learning** - Collaborative training with fitness-based weighting
- **Transfer Learning** - Knowledge distillation across tree species
- **Environmental Simulation** - Dynamic climate and stressor modeling
- **Distribution Shifts** - Gradual, sudden, and cyclical data changes

### Phase 6+: Self-Evolution & Meta-Learning ✅
- **Forest Consciousness** - Meta-controller for autonomous monitoring and improvement
- **Goal Management** - Learning objectives with progress tracking
- **Architecture Search** - Evolutionary optimization of tree structures
- **Self-Improvement Loop** - Continuous autonomous enhancement

### Phase 7: Production & Scaling ✅
- **Production API** - Production-oriented interface for deployment (`NeuralForestAPI`)
- **Checkpoint Management** - Robust save/load with validation
- **Performance Benchmarks** - Comprehensive latency, throughput, memory, and accuracy testing
- **Docker & Kubernetes** - Production-ready deployment configurations

## Main Features
- Prioritized experience replay (weighted sampling without full sorting)
- Anchor coreset memory — representative “skill anchors” from previous data
- Routing: top-k tree experts per input
- Distillation (LwF-style “Learning without Forgetting”)
- Drift detection, with dynamic tree growing/pruning
- Optimizer state preservation during structural changes
- Visualization: interactive graphs of network and function fitting

## Requirements
- Python >= 3.10
- PyTorch >= 1.12
- NumPy
- Matplotlib
- NetworkX

## Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Basic Training
Run in terminal:
```bash
python NeuralForest.py
```
Or, in a Jupyter notebook:
```python
%run NeuralForest.py
```

### Phase Demos
Explore different capabilities:
```bash
# Phase 1-4: Root system, groves, canopy, seasonal cycles
python phase1_demo.py  # Root system
python phase2_demo.py  # Specialized groves
python phase3_demo.py  # Canopy routing
python phase4_demo.py  # Seasonal cycles

# Phase 5: Multi-modal understanding & Visualization
python phase5_demo.py  # Multi-modal tasks, evolution visualization

# Phase 6: Cooperation, environmental adaptation, self-evolution
python phase6_demo.py  # Tree cooperation, environmental simulation, consciousness

# Phase 7: Production & Scaling  
python phase7_demo.py  # API, checkpoints, benchmarks, production-oriented deployment
```

## 💻 CPU Training

NeuralForest fully supports CPU training with identical results to GPU training.

### Quick Start

```bash
# 100 epochs (4-5 hours on modern CPU)
python training_demos/cifar10_hybrid_training.py --epochs 100 --batch_size 16 --device cpu

# Or use GitHub Actions for automatic training
# See .github/workflows/cpu_training.yml
```

### Performance

- **100 epochs**: ~4-5 hours (CPU) vs ~40 min (GPU)
- **Accuracy**: 85-88% (identical results)
- **Use case**: Overnight training, no GPU available, batch experiments

For details, see [docs/CPU_TRAINING.md](docs/CPU_TRAINING.md)

### Production API
```python
from api import NeuralForestAPI

# Load trained forest
api = NeuralForestAPI(checkpoint_path='checkpoints/forest.pt')

# Make predictions
result = api.predict({'input': data})
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")

# Online learning
api.train_online({'input': new_data}, targets)

# Check health
status = api.get_forest_status()
health = api.health_check()
```

### Security Notes
- Checkpoints are trusted-only artifacts and should only be loaded from known-good sources.
- The production API should still undergo security audits, load testing, and input-sanitization review before production use.
- Validate checkpoints before deployment and keep checkpoint handling in the trusted artifact boundary.

### Self-Evolution
```python
from consciousness import ForestConsciousness
from evolution import SelfImprovementLoop

# Create consciousness
consciousness = ForestConsciousness(forest)

# Set learning goals
consciousness.goals.create_goal(
    name="Achieve High Fitness",
    target_metric="forest_fitness",
    target_value=7.0,
    priority=3
)

# Run evolution
result = consciousness.evolve()

# Self-improvement
improvement = SelfImprovementLoop(forest, consciousness)
cycle_result = improvement.run_cycle()
```

### Visualization
```python
from evolution import ForestVisualizer

# Create visualizer
visualizer = ForestVisualizer(save_dir='./visualizations')

# Plot fitness trends
visualizer.plot_fitness_trends(history, show=True)

# Plot architecture distribution
visualizer.plot_architecture_distribution(trees_data, method='pca')

# Create comprehensive dashboard
visualizer.create_evolution_dashboard(history, trees_data, genealogy_data)

# Export all plots
saved_files = visualizer.export_all_plots(history, trees_data, genealogy_data)
```

### Tree Cooperation
```python
from evolution import CooperationSystem

# Create cooperation system
cooperation = CooperationSystem()

# Enable communication
for tree_id in range(5):
    cooperation.enable_tree_communication(tree_id)

# Send messages between trees
cooperation.communication.send_message(
    sender_id=0, receiver_id=1,
    message_type='knowledge',
    content={'tip': 'Try lower learning rate'}
)

# Coordinate federated learning
result = cooperation.coordinate_learning(
    participating_trees,
    coordination_type='federated'
)
```

### Environmental Simulation
```python
from evolution import EnvironmentalSimulator, ClimateType, StressorType

# Create environmental simulator
env = EnvironmentalSimulator(initial_climate=ClimateType.TEMPERATE)

# Simulate environmental steps
for _ in range(10):
    state = env.step()
    print(f"Resources: {state.resource_availability:.2f}")
    print(f"Data quality: {state.data_quality:.2f}")

# Trigger environmental stressors
env.trigger_stressor(StressorType.DROUGHT)

# Apply environmental effects to data
modified_x, modified_y = env.apply_to_data(data_x, data_y)
```

### Benchmarking
```python
from benchmarks import PerformanceBenchmark

# Create benchmark suite
benchmark = PerformanceBenchmark(api)

# Run full benchmark
results = benchmark.run_full_benchmark(input_dim=2)

# Individual benchmarks
latency = benchmark.benchmark_latency(input_dim=2, num_samples=1000)
throughput = benchmark.benchmark_throughput(input_dim=2, duration_seconds=10)
memory = benchmark.benchmark_memory(input_dim=2)
```

## Visualization
The script opens a matplotlib interactive window showing a graph network of trees and a plot of model fit at regular intervals.

## Deployment

### Docker
```bash
# Build image
docker build -t neuralforest:latest -f deployment/docker/Dockerfile .

# Run container
docker run -v $(pwd)/checkpoints:/app/checkpoints neuralforest:latest

# Or use docker-compose
cd deployment/docker
docker-compose up
```

### Kubernetes
```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/hpa.yaml

# Check status
kubectl get pods -l app=neuralforest
kubectl logs -f deployment/neuralforest
```

## Project Structure
```
NeuralForest/
├── NeuralForest.py          # Core forest implementation
├── metrics.py               # Evaluation metrics
├── phase1_demo.py - phase7_demo.py  # Phase demonstrations
│
├── soil/                    # Media processors (Phase 1)
├── roots/                   # Unified backbone (Phase 1)
├── groves/                  # Specialized experts (Phase 2)
├── canopy/                  # Routing & attention (Phase 3)
├── mycelium/                # Knowledge transfer (Phase 2)
├── seasons/                 # Training regimes (Phase 4)
├── tasks/                   # Multi-modal tasks (Phase 5)
│   ├── vision/             # Image tasks
│   ├── audio/              # Audio tasks
│   ├── text/               # Text tasks
│   ├── video/              # Video tasks
│   └── cross_modal/        # Multi-modal tasks
│
├── consciousness/           # Meta-controller (Phase 6+)
│   ├── meta_controller.py  # Forest consciousness
│   └── goal_manager.py     # Learning objectives
│
├── evolution/               # Evolution & Cooperation (Phases 3-6)
│   ├── architecture_search.py  # Neural architecture search
│   ├── self_improvement.py     # Self-improvement loop
│   ├── visualization.py        # Evolution visualization (Phase 5)
│   ├── cooperation.py          # Tree cooperation (Phase 6)
│   ├── environmental_sim.py    # Environmental simulation (Phase 6)
│   ├── genealogy.py            # Genealogy tracking
│   ├── monitoring.py           # Real-time monitoring
│   ├── humus_nursery.py        # Eliminated tree archive (primary; tree_graveyard shim retained)
│   ├── tree_graveyard.py       # Backward-compatibility shim
│   └── ...                     # Other evolution modules
│
├── api/                     # Production API (Phase 7)
│   └── forest_api.py       # NeuralForestAPI
│
├── benchmarks/              # Performance tests (Phase 7)
│   └── performance_tests.py
│
├── deployment/              # Deployment configs (Phase 7)
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── kubernetes/
│       ├── deployment.yaml
│       └── hpa.yaml
│
└── tests/                   # Unit tests
```

## Author
[V1B3hR](https://github.com/V1B3hR)

## License
MIT
