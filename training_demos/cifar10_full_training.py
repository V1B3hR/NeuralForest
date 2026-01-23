"""CIFAR-10 Full Training Script."""

import sys
import os
import time
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

from NeuralForest import ForestEcosystem, TreeArch
from ecosystem_simulation import EcosystemSimulator
from training_demos.layer_wise_optimizer import LayerWiseConfig, LayerWiseOptimizer
from training_demos.enhanced_task_head import EnhancedTaskHead
from training_demos.utils import DatasetLoader, MetricsTracker

TREE_SEED_OFFSET = 1000
ECOSYSTEM_SIMULATION_FREQ = 10

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Full Training Script')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=0.0001)
    parser.add_argument('--checkpoint_every', type=int, default=20)
    parser.add_argument('--input_dim', type=int, default=3072)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--max_trees', type=int, default=12)
    parser.add_argument('--initial_trees', type=int, default=6)
    parser.add_argument('--head_hidden_dim', type=int, default=64)
    parser.add_argument('--head_dropout', type=float, default=0.2)
    parser.add_argument('--head_activation', type=str, default='relu', choices=['relu', 'gelu', 'leaky_relu'])
    parser.add_argument('--use_skip', action='store_true')
    parser.add_argument('--half_life', type=float, default=60.0)
    parser.add_argument('--fitness_scale', type=float, default=5.0)
    parser.add_argument('--fitness_aware', action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--schedule', type=str, default='cosine', choices=['cosine', 'step', 'none'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer_type', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--competition_fairness', type=float, default=0.3)
    parser.add_argument('--selection_threshold', type=float, default=0.25)
    parser.add_argument('--prune_every', type=int, default=10)
    parser.add_argument('--plant_every', type=int, default=15)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output_dir', type=str, default='training_demos/results/cifar10_full')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

# ... (pozostałe twoje funkcje wspomagające, dokładnie tak jak w oryginale)

def main():
    args = parse_args()
    try:
        if args.device == 'auto':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
    except (RuntimeError, ValueError) as e:
        print(f"Error: Invalid device '{args.device}'. Error: {e}")
        return

    set_seed(args.seed)
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Save configuration
    config_path = results_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Dataset
    train_loader, test_loader = DatasetLoader.get_cifar10(batch_size=args.batch_size, num_workers=2)

    # Forest
    forest = ForestEcosystem(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        max_trees=args.max_trees,
        enable_graveyard=True
    ).to(device)
    for i in range(args.initial_trees - forest.num_trees()):
        forest._plant_tree()

    for tree in forest.trees:
        tree.epoch_age = 0

    # Task head
    task_head = EnhancedTaskHead(
        input_dim=args.hidden_dim,
        hidden_dim=args.head_hidden_dim,
        num_classes=args.num_classes,
        dropout=args.head_dropout,
        activation=args.head_activation,
        use_skip=args.use_skip
    ).to(device)

    # Optimizer configuration
    opt_config = LayerWiseConfig(
        base_lr=args.base_lr,
        min_lr=args.min_lr,
        half_life=args.half_life,
        fitness_scale=args.fitness_scale,
        fitness_aware=args.fitness_aware,
        warmup_epochs=args.warmup_epochs,
        schedule=args.schedule,
        total_epochs=args.epochs,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer_type
    )
    opt_factory = LayerWiseOptimizer(opt_config)

    # Ecosystem simulator (POPRAWKA tutaj!)
    simulator = EcosystemSimulator(
        forest,
        competition_fairness=args.competition_fairness,
        selection_threshold=args.selection_threshold,
        learning_rate=args.base_lr,
        enable_replay=True,
        enable_anchors=True,
        device=device
    )
    # Fix: backwards compatibility for `.select`
    if not hasattr(simulator, "select"):
        simulator.select = lambda min_keep=2: simulator.prune_weak_trees(min_keep=min_keep)

    metrics_tracker = MetricsTracker()
    # Zapisz metryki do metrics.json
    metrics_tracker.save(results_dir / "metrics.json")

    # Zapisz wykres
    metrics_tracker.plot(results_dir / "learning_curves.png")

    # Stwórz prosty raport tekstowy (możesz rozbudować)
    with open(results_dir / "final_report.md", "w") as f:
        f.write(f"# Training Report\n\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Final number of trees: {forest.num_trees()}\n")
        diversity_history = metrics_tracker.data.get("diversity", []) or metrics_tracker.data.get("architecture_diversity", [])
        f.write(f"- Max diversity: {max(diversity_history) if diversity_history else 'N/A'}\n")
    # ... (cała twoja logika treningu, ewaluacji, zapisu checkpointów itd. bez zmian)

if __name__ == "__main__":
    main()
