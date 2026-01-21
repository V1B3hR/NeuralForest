import argparse
import yaml
from pathlib import Path
import random
import numpy as np
import torch
from training_demos.utils import DatasetLoader, MetricsTracker

# === LIBERAL EVOLUTIONARY UTILS ===

def reward_tree(tree, metrics, config):
    """Natural reward system for trees."""
    bonus = 0
    if metrics.get('test_accuracy', 0) > tree.best_test_accuracy:
        bonus += config.get('reward_system', {}).get('sun_bonus', True) * 1
        tree.best_test_accuracy = metrics['test_accuracy']
    if getattr(tree, 'did_mutate', False):
        bonus += config.get('reward_system', {}).get('rain_bonus', True) * 0.5
    if getattr(tree, 'age', 0) > 0 and getattr(tree, 'age', 0) % 10 == 0:
        bonus += config.get('reward_system', {}).get('mineral_bonus', True) * 0.2
    if getattr(tree, 'recycled', False):
        bonus += config.get('reward_system', {}).get('soil_enrichment', True) * 0.2
    tree.fitness += bonus


def adaptive_mutation(tree, forest, config):
    """Apply mutations depending on forest diversity."""
    import copy
    diversity_metric = getattr(forest, "compute_diversity", lambda: 1.0)()
    scope = config.get('mutation_scope', 'adaptive')
    if scope == 'adaptive':
        mutation_prob = max(0.1, 1.0 - diversity_metric)
    else:
        mutation_prob = 0.2

    changed = False
    if random.random() < mutation_prob:
        old_dim = tree.hidden_dim
        tree.hidden_dim = int(old_dim * (random.uniform(0.8, 1.2)))
        changed = True
    if random.random() < mutation_prob * 0.7:
        old_drop = tree.head_dropout
        tree.head_dropout = min(0.5, max(0.1, old_drop + random.uniform(-0.05, 0.05)))
        changed = True
    if random.random() < mutation_prob * 0.5:
        tree.head_activation = random.choice(['relu', 'gelu', 'leaky_relu'])
        changed = True
    if changed:
        tree.did_mutate = True

# Placeholder ForestEcosystem and Tree
class Tree:
    def __init__(self, config):
        self.hidden_dim = config['hidden_dim']
        self.head_dropout = config.get('head_dropout', 0.2)
        self.head_activation = config.get('head_activation', 'relu')
        self.fitness = 1.0
        self.age = 0
        self.best_test_accuracy = 0.0
        self.did_mutate = False
        self.recycled = False

class ForestEcosystem:
    def __init__(self, config):
        self.trees = [Tree(config) for _ in range(config['initial_trees'])]
        self.config = config
        self.epoch = 0

    def grow_forest(self):
        config = self.config
        # Pollination if diversity drops
        if config.get('pollination_on_low_diversity', False) and self.compute_diversity() < 2:
            parent = random.choice(self.trees)
            for _ in range(2):
                t = Tree(config)
                t.hidden_dim = parent.hidden_dim
                t.head_dropout = parent.head_dropout + random.uniform(-0.03, 0.03)
                t.head_activation = parent.head_activation
                self.trees.append(t)
        # Plant new trees
        if len(self.trees) < config['max_trees']:
            t = Tree(config)
            adaptive_mutation(t, self, config)
            self.trees.append(t)
        # Prune
        if len(self.trees) > config['min_trees']:
            sorted_trees = sorted(self.trees, key=lambda x: x.fitness)
            for i in range(int(0.2*len(self.trees))):
                if sorted_trees[i].age > 8:
                    self.trees.remove(sorted_trees[i])
        # Age and reward all
        for t in self.trees:
            t.age += 1
            reward_tree(t, {'test_accuracy': random.uniform(0,1)}, config)

    def compute_diversity(self):
        # Simple diversity: count unique head_activation
        activations = {t.head_activation for t in self.trees}
        return len(activations)


# === MAIN TRAINING LOGIC ===

def load_config(args):
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    # Override from CLI precisely
    for k,v in vars(args).items():
        if v is not None:
            config[k] = v
    # Add defaults if missing:
    config.setdefault('initial_trees', 10)
    config.setdefault('max_trees', 20)
    config.setdefault('min_trees', 6)
    config.setdefault('epochs', 100)
    config.setdefault('batch_size', 64)
    config.setdefault('optimizer_type', 'adam')
    config.setdefault('hidden_dim', 512)
    config.setdefault('head_dropout', 0.2)
    config.setdefault('head_activation', 'relu')
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='YAML config path')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--initial_trees', type=int, default=None)
    parser.add_argument('--max_trees', type=int, default=None)
    parser.add_argument('--min_trees', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='training_demos/results/liberal_forest')
    args = parser.parse_args()

    config = load_config(args)
    print("\n--- Using Configuration ---")
    for k,v in config.items():
        print(f"{k}: {v}")
    print("---------------------------\n")

    # Set random seeds (optionally)
    if 'seed' in config:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
    # Data loaders (adjust as needed)
    train_loader, test_loader = DatasetLoader.get_cifar10(batch_size=config['batch_size'])

    # Metrics Tracker
    tracker = MetricsTracker()

    # Forest ecosystem
    forest = ForestEcosystem(config)
    print(f"Initial trees: {len(forest.trees)}")

    for epoch in range(config['epochs']):
        # Train: (mock demo logic -- replace with real train step)
        for t in forest.trees:
            t.fitness *= 1 + random.uniform(-0.03, 0.05)
            reward_tree(t, {'test_accuracy': random.uniform(0.3,0.9)}, config)
        # Periodic evolutionary steps
        if (epoch+1) % config['plant_every'] == 0 or (epoch+1) % config['prune_every'] == 0:
            forest.grow_forest()
            print(f"[Epoch {epoch+1}] Trees: {len(forest.trees)}, Diversity: {forest.compute_diversity()}")

        # Track (mock): document diversity, fitness, num trees
        avg_fitness = np.mean([t.fitness for t in forest.trees])
        tracker.update(epoch+1, {
            'epoch': epoch+1,
            'num_trees': len(forest.trees),
            'architecture_diversity': forest.compute_diversity(),
            'avg_fitness': avg_fitness,
            'train_accuracy': 0.75 + random.uniform(-0.05, 0.08),
            'test_accuracy': 0.5 + random.uniform(-0.1, 0.09),
        })

    print("\n--- Training Complete! ---")
    tracker.save(Path(config['output_dir']) / "metrics.json")
    tracker.plot(Path(config['output_dir']) / "learning_curves.png")


if __name__ == "__main__":
    main()
