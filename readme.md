# NeuralForest

**NeuralForest v2** is an experimental neural network ecosystem combining tree experts, prioritized experience replay, coreset memory, gating/routing, drift detection, and visualization. 
                    It is designed for continual learning, with robustness against data drift and dynamic expert growth/pruning.

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
Run in terminal:
```bash
python NeuralForest.py
```
Or, in a Jupyter notebook:
```python
%run NeuralForest.py
```

## Visualization
The script opens a matplotlib interactive window showing a graph network of trees and a plot of model fit at regular intervals.

## Author
[V1B3hR](https://github.com/V1B3hR)

## License
MIT
