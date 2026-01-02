# NeuralForest

**NeuralForest v2** is an experimental neural network ecosystem combining tree experts, prioritized experience replay, coreset memory, gating/routing, drift detection, and visualization. 
It is designed for continual learning, with robustness against data drift and dynamic expert growth/pruning.

## ✨ New in Phase 6 & 7

### Phase 6: Self-Evolution & Meta-Learning ✅
- **Forest Consciousness** - Meta-controller for autonomous monitoring and improvement
- **Goal Management** - Learning objectives with progress tracking
- **Architecture Search** - Evolutionary optimization of tree structures
- **Self-Improvement Loop** - Continuous autonomous enhancement

### Phase 7: Production & Scaling ✅
- **Production API** - Clean interface for deployment (`NeuralForestAPI`)
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
# Phase 1-5: Soil processors, groves, canopy, seasons, multi-modal tasks
python phase1_demo.py  # Root system
python phase2_demo.py  # Specialized groves
python phase3_demo.py  # Canopy routing
python phase4_demo.py  # Seasonal cycles
python phase5_demo.py  # Multi-modal understanding

# Phase 6: Self-evolution & Meta-learning
python phase6_demo.py  # Consciousness, goals, architecture search, self-improvement

# Phase 7: Production & Scaling  
python phase7_demo.py  # API, checkpoints, benchmarks, deployment readiness
```

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
├── consciousness/           # Meta-controller (Phase 6)
│   ├── meta_controller.py  # Forest consciousness
│   └── goal_manager.py     # Learning objectives
│
├── evolution/               # Architecture search (Phase 6)
│   ├── architecture_search.py
│   └── self_improvement.py
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
