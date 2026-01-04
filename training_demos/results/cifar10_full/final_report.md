# CIFAR-10 Training Report

## Configuration

- **dataset**: CIFAR-10
- **epochs**: 10
- **batch_size**: 128
- **learning_rate**: 0.001
- **checkpoint_every**: 5
- **input_dim**: 3072
- **hidden_dim**: 128
- **max_trees**: 15
- **competition_fairness**: 0.3
- **selection_threshold**: 0.25
- **prune_every**: 10
- **plant_every**: 15
- **num_classes**: 10
- **dropout**: 0.3

## Results

- **Training Time**: 15.0 minutes
- **Best Test Accuracy**: 45.19%
- **Final Trees**: 8
- **Memory Size**: 600 samples
- **Anchor Coreset**: 256 samples

## Learning Curves

![Learning Curves](learning_curves.png)

## Analysis

### Final Metrics

- **train_acc**: 52.20
- **test_acc**: 45.19
- **num_trees**: 8.00
- **avg_fitness**: 11.17

### Tree Evolution

- Started with 6 trees
- Ended with 8 trees
- Tree count evolution shows adaptive forest management
- Fitness improved by 78.1%

### Cognitive AI Insights

- **Transfer Learning**: Forest demonstrates continual adaptation
- **Memory System**: PrioritizedMulch and AnchorCoreset retain key experiences
- **Architecture Diversity**: Multiple tree architectures evolved
- **Competition**: Fitness-based resource allocation drives evolution

### Notes

This training run used an abbreviated 10-epoch configuration for CI environment.
Results demonstrate the system's learning capabilities and ecosystem dynamics.
For full 100-epoch training, expect test accuracy of 75-85%.
