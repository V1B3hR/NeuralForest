# NeuralForest Training Demonstrations

Comprehensive live training demonstrations showing NeuralForest learning on real datasets with full cognitive AI evaluation.

## 📋 Overview

This directory contains four main training demonstrations:

1. **CIFAR-10 Hybrid Training** (`cifar10_hybrid_training.py`) - **NEW!** Advanced training with layer-wise optimizer
2. **CIFAR-10 Full Training** (`cifar10_full_training.py`) - Complete 100-epoch training
3. **Continual Learning** (`continual_learning_demo.py`) - Multi-stage learning across 3 datasets
4. **Few-Shot Learning** (`few_shot_demo.py`) - Rapid adaptation with minimal examples

## 🚀 Quick Start

### Installation

Ensure you have the required dependencies:

```bash
pip install torch torchvision numpy matplotlib networkx
```

### Run Smoke Test

Verify the infrastructure is working:

```bash
python training_demos/test_smoke.py
```

### Run Demonstrations

#### 0. CIFAR-10 Hybrid Training (NEW!) (~50-60 minutes on GPU)

**🌟 Advanced training with layer-wise optimizer system**

**Quick Start:**

```bash
python training_demos/cifar10_hybrid_training.py
```

**Custom Configuration:**

```bash
python training_demos/cifar10_hybrid_training.py \
    --epochs 100 \
    --batch_size 64 \
    --base_lr 0.01 \
    --hidden_dim 512 \
    --head_hidden_dim 64 \
    --half_life 60.0 \
    --fitness_aware \
    --warmup_epochs 5 \
    --max_trees 12 \
    --initial_trees 6
```

**Available Arguments:**

- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 64 for better competition)
- `--base_lr`: Base learning rate (default: 0.01)
- `--min_lr`: Minimum learning rate (default: 0.0001)
- `--hidden_dim`: Forest hidden dimension (default: 512)
- `--head_hidden_dim`: Task head hidden dimension (default: 64)
- `--half_life`: Age decay half-life in epochs (default: 60.0)
- `--fitness_scale`: Target fitness for scaling (default: 5.0)
- `--fitness_aware`: Enable fitness-aware LR adjustment
- `--warmup_epochs`: Number of warmup epochs (default: 5)
- `--schedule`: LR schedule (cosine/step/none, default: cosine)
- `--max_trees`: Maximum trees (default: 12)
- `--initial_trees`: Initial trees (default: 6)
- `--head_activation`: Task head activation (relu/gelu/leaky_relu)
- `--use_skip`: Use skip connection in task head
- `--output_dir`: Results directory (default: training_demos/results/cifar10_hybrid)

**What it does:**
- Implements layer-wise learning rates (early layers slow, late layers fast)
- Applies exponential age decay (older trees → lower LR)
- Uses fitness-aware LR adjustment (high fitness → lower LR)
- Enhanced task head with refinement layer (128→64→10)
- Batch size 64 for better competition dynamics
- Unique tree seeds for initialization diversity
- Dynamic optimizer recreation per epoch
- Complete 100-epoch CIFAR-10 training

**Expected Results:**
- Test Accuracy: 85-88% (target: >80%)
- Training Time: 50-60 minutes (GPU)
- Trees: 10-12 (evolved from 6)
- Fitness: +250-300% improvement
- Architecture Diversity: 5-7 unique types

**Results location:** `training_demos/results/cifar10_hybrid/`

#### 1. CIFAR-10 Full Training (~20-30 minutes on GPU, longer on CPU)

**Quick Start (100 epochs):**

```bash
# Using default configuration
python training_demos/cifar10_full_training.py --epochs 100

# Or use the convenience script
./training_demos/run_full_training.sh
```

**Custom Configuration:**

```bash
python training_demos/cifar10_full_training.py \
    --epochs 100 \
    --batch_size 256 \
    --learning_rate 0.0005 \
    --max_trees 20 \
    --competition_fairness 0.2 \
    --output_dir training_demos/results/custom_run
```

**Available Arguments:**

- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 128)
- `--learning_rate`: Learning rate (default: 0.001)
- `--checkpoint_every`: Checkpoint frequency (default: 20)
- `--max_trees`: Maximum trees (default: 15)
- `--competition_fairness`: Fairness factor 0-1 (default: 0.3)
- `--selection_threshold`: Selection threshold (default: 0.25)
- `--prune_every`: Prune every N epochs (default: 10)
- `--plant_every`: Plant every N epochs (default: 15)
- `--input_dim`: Input dimension (default: 3072)
- `--hidden_dim`: Hidden dimension (default: 128)
- `--num_classes`: Number of classes (default: 10)
- `--dropout`: Dropout rate (default: 0.3)
- `--output_dir`: Results directory (default: training_demos/results/cifar10_full_100ep)

See `--help` for all options:

```bash
python training_demos/cifar10_full_training.py --help
```

**What it does:**
- Trains NeuralForest on CIFAR-10 for specified epochs
- Saves checkpoints at regular intervals
- Tracks comprehensive metrics (accuracy, loss, trees, fitness, etc.)
- Generates learning curves and final report

**Results location:** Specified by `--output_dir` (default: `training_demos/results/cifar10_full_100ep/`)

#### 2. Continual Learning (~30-40 minutes)

```bash
python training_demos/continual_learning_demo.py
```

**What it does:**
- Stage 1: MNIST (epochs 1-30)
- Stage 2: Fashion-MNIST (epochs 31-60)
- Stage 3: CIFAR-10 (epochs 61-100)
- Analyzes memory retention and catastrophic forgetting
- Generates stage-specific visualizations

**Results location:** `training_demos/results/continual_learning/`

#### 3. Few-Shot Learning (~10-15 minutes)

```bash
python training_demos/few_shot_demo.py
```

**What it does:**
- Pre-trains on 9 CIFAR-10 classes (30 epochs)
- Adapts to 10th class with only 10 examples (10 epochs)
- Demonstrates rapid adaptation and knowledge retention
- Tracks adaptation curve

**Results location:** `training_demos/results/few_shot/`

## 📊 Expected Results

### CIFAR-10 Full Training
- **Target accuracy**: >75% (ideal: 80-85%)
- **Final trees**: 10-15 (evolved from 6)
- **Fitness improvement**: >200%
- **Architecture diversity**: 4-6 unique types

### Continual Learning
- **Catastrophic forgetting**: <10% average
- **Final retention**: All three datasets retained
- **Memory system**: PrioritizedMulch + AnchorCoreset active

### Few-Shot Learning
- **Adaptation**: >50% accuracy on new class with 10 examples
- **Knowledge retention**: >70% on original 9 classes
- **Sample efficiency**: Rapid learning demonstrated

## 📁 Results Structure

```
training_demos/results/
├── cifar10_full/
│   ├── checkpoints/
│   │   ├── epoch_20.pt
│   │   ├── epoch_40.pt
│   │   ├── epoch_60.pt
│   │   ├── epoch_80.pt
│   │   └── epoch_100.pt
│   ├── best_model.pt
│   ├── learning_curves.png
│   ├── metrics.json
│   └── final_report.md
├── continual_learning/
│   ├── stage1_mnist.png
│   ├── stage2_fashion_mnist.png
│   ├── stage3_cifar_10.png
│   ├── retention_analysis.png
│   └── continual_report.md
└── few_shot/
    ├── adaptation_curve.png
    └── few_shot_report.md
```

## 🎯 Key Features

### Dataset Loaders (`utils.py`)
- **CIFAR-10**: 50K train, 10K test (32×32 RGB, 10 classes)
- **MNIST**: 60K train, 10K test (28×28→32×32 grayscale, 10 digits)
- **Fashion-MNIST**: 60K train, 10K test (28×28→32×32 grayscale, 10 categories)
- Automatic download and normalization
- Consistent preprocessing pipeline

### Metrics Tracking (`utils.py`)
- Training/test loss and accuracy
- Number of trees over time
- Average tree fitness
- Architecture diversity
- Memory usage (PrioritizedMulch size)
- Automatic plotting and JSON export

### Forest Integration
- **ForestEcosystem**: Adaptive tree population with per-tree architectures
- **EcosystemSimulator**: Competition, selection, and evolution
- **Task Head**: ImageClassification with multi-layer design
- **Memory Systems**: PrioritizedMulch (replay) + AnchorCoreset (retention)

## 🔬 Cognitive AI Features Demonstrated

### 1. Transfer Learning
- Cross-domain knowledge transfer (MNIST → Fashion → CIFAR)
- Shared representations across tasks
- Adaptive routing to specialized trees

### 2. Memory & Retention
- Experience replay with importance sampling
- Representative anchors for knowledge preservation
- Catastrophic forgetting prevention

### 3. Few-Shot Adaptation
- Quick learning with minimal examples
- Meta-learning capabilities
- Knowledge preservation during adaptation

### 4. Architecture Evolution
- Per-tree NAS (Neural Architecture Search)
- Fitness-based competition
- Dynamic pruning and planting
- Architecture diversity emergence

### 5. Robustness
- Handles data scarcity (drought)
- Handles noisy data (flood)
- Ecosystem resilience

### 6. Hybrid Optimizer System (NEW!)
- **Layer-wise learning rates**: Early layers learn slowly, late layers learn fast
- **Exponential age decay**: `age_factor = exp(-epoch_age / half_life)`
- **Fitness-aware adjustment**: High-fitness trees learn slower (more refined)
- **Per-layer warmup**: Different warmup rates for different layer depths
- **Dynamic optimizer recreation**: Optimizer recreated each epoch with updated age/fitness
- **Enhanced task head**: Refinement layer (128→64→10) with layer normalization

**Learning Rate Evolution Example:**
```
Epoch 0 (new tree, age=0, fitness=5.0):
  Early layer: 0.001 (0.1× base_lr)
  Late layer:  0.01  (1.0× base_lr)
  Task head:   0.02  (2.0× base_lr)

Epoch 50 (old tree, age=50, fitness=8.0):
  Early layer: 0.0006  (age & fitness decay)
  Late layer:  0.006
  Task head:   0.015

Epoch 100 (very old tree, age=100, fitness=12.0):
  Early layer: 0.0002  (heavy decay)
  Late layer:  0.002
  Task head:   0.005
```

## 🛠️ Configuration

Each demo has a `CONFIG` dictionary you can modify:

### Common Parameters
- `batch_size`: Batch size for training (default: 128)
- `learning_rate`: Learning rate for optimizer (default: 0.001)
- `input_dim`: Flattened input dimension (3072 for 32×32×3)
- `hidden_dim`: Forest hidden dimension (default: 128)
- `max_trees`: Maximum trees in forest (default: 15-20)

### Ecosystem Parameters
- `competition_fairness`: Balance between fitness and equality (0-1, default: 0.3)
- `selection_threshold`: Fitness threshold for pruning (default: 0.25)
- `prune_every`: Epochs between pruning (default: 10)
- `plant_every`: Epochs between planting (default: 10-15)

### Task Head Parameters
- `num_classes`: Number of output classes (10 for all demos)
- `dropout`: Dropout rate (default: 0.3)

## 📝 Reports Generated

Each demo generates a comprehensive Markdown report with:
- Configuration details
- Training metrics and final results
- Embedded visualizations (learning curves, etc.)
- Analysis and insights
- Cognitive AI evaluation
- Comparison with baselines (where applicable)

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size` (e.g., 64 or 32)
- Reduce `max_trees` (e.g., 10)
- Reduce `hidden_dim` (e.g., 64)

### Slow Training
- Training on CPU is significantly slower than GPU
- Consider reducing epochs for testing (e.g., 20 instead of 100)
- Reduce dataset size for quick validation

### Dataset Download Issues
- Datasets download automatically to `./data/`
- If download fails, check internet connection
- You can manually download datasets and place in `./data/`

## 🎓 Understanding the Results

### Learning Curves
- **Loss**: Should decrease over time (train and test)
- **Accuracy**: Should increase over time
- **Gap between train/test**: Indicates overfitting if large

### Tree Evolution
- **Number of trees**: Should stabilize around 10-15
- **Fitness**: Should improve significantly (>200%)
- **Diversity**: Should maintain 4-6 unique architectures

### Memory System
- **PrioritizedMulch size**: Should grow to ~10K samples
- **AnchorCoreset**: Should maintain ~256 representative samples
- Both enable continual learning and prevent forgetting

## 📚 See Also

- **Main Report**: `LIVE_TRAINING_REPORT.md` - Comprehensive analysis
- **NeuralForest Core**: `NeuralForest.py` - Forest implementation
- **Ecosystem**: `ecosystem_simulation.py` - Competition and evolution
- **Tasks**: `tasks/vision/classification.py` - Task heads

## 🤝 Contributing

To add new demonstrations:
1. Create a new Python file in `training_demos/`
2. Import utilities from `utils.py`
3. Follow the structure of existing demos
4. Add results directory in `results/`
5. Generate comprehensive report

## 📄 License

Same as NeuralForest main repository.
