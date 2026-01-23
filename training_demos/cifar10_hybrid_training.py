import sys
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from pathlib import Path

# ==============================
# --- Trivial Data Simulation ---
# (symuluje partiami "CIFAR-10" loader na potrzeby przykładu)
# ==============================
class DummyLoader:
    def __init__(self, batch_size, batches=50):
        self.batch_size = batch_size
        self.batches = batches

    def __iter__(self):
        for _ in range(self.batches):
            # 32x32x3 obrazki + etykiety
            data = torch.randn(self.batch_size, 3, 32, 32)
            targets = torch.randint(0, 10, size=(self.batch_size,))
            yield data, targets

    def __len__(self):
        return self.batches

# ===================
# --- METRICS ---
# ===================
class MetricsTracker:
    def __init__(self):
        self.history = {}

    def update(self, epoch, dictlike):
        for k, v in dictlike.items():
            self.history.setdefault(k, []).append(v)

    @property
    def data(self):
        return self.history

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot(self, path):
        keys = list(self.history.keys())
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        panels = [k for k in keys if k != 'epoch']
        for idx, k in enumerate(panels[:4]):
            axes[idx//2, idx%2].plot(self.history['epoch'], self.history[k], label=k)
            axes[idx//2, idx%2].set_title(k)
            axes[idx//2, idx%2].legend()
        fig.tight_layout()
        plt.savefig(path)
        plt.close()

# ==============================
# --- Ewolucyjny las -----------
# ==============================
def reward_tree(tree, metrics, config, forest):
    bonus = 0
    # Bazowy bonus za accuracy
    if metrics.get('test_accuracy', 0) > tree.best_test_accuracy:
        bonus += config.get('reward_system', {}).get('sun_bonus', True) * 1
        tree.best_test_accuracy = metrics['test_accuracy']
    if getattr(tree, 'did_mutate', False):
        bonus += config.get('reward_system', {}).get('rain_bonus', True) * 0.5
    if getattr(tree, 'age', 0) > 0 and getattr(tree, 'age', 0) % 10 == 0:
        bonus += config.get('reward_system', {}).get('mineral_bonus', True) * 0.2
    if getattr(tree, 'recycled', False):
        bonus += config.get('reward_system', {}).get('soil_enrichment', True) * 0.2
    # Nowy BONUS: za unikalny typ aktywacji
    head_acts = [t.head_activation for t in forest.trees]
    if head_acts.count(tree.head_activation) == 1:
        bonus += config.get('reward_system', {}).get('diversity_bonus', 1.0) * 0.5
    tree.fitness += bonus

def adaptive_mutation(tree, forest, config):
    diversity_metric = forest.compute_diversity()
    # Mocniej zachęcamy do mutacji:
    if config.get('mutation_scope', 'adaptive') == 'adaptive':
        mutation_prob = max(0.3, 1.0 - 0.6 * diversity_metric)
    else:
        mutation_prob = 0.5

    changed = False
    # Mutacja rozmiaru warstwy
    if random.random() < mutation_prob:
        tree.hidden_dim = int(tree.hidden_dim * random.uniform(0.8, 1.2))
        changed = True
    # Mutacja dropout
    if random.random() < mutation_prob * 0.7:
        tree.head_dropout = min(0.5, max(0.1, tree.head_dropout + random.uniform(-0.05, 0.05)))
        changed = True
    # Mutacja aktywacji – SZANSA 0.9×
    if random.random() < mutation_prob * 0.9:
        tree.head_activation = random.choice(['relu', 'gelu', 'leaky_relu'])
        changed = True
    if changed:
        tree.did_mutate = True

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
        # Zawsze próbuj stworzyć nowe drzewo o zmutowanej architekturze:
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
        # Age i bonusy
        for t in self.trees:
            t.age += 1
            reward_tree(t, {'test_accuracy': random.uniform(0, 1)}, config, self)

    def compute_diversity(self):
        acts = {t.head_activation for t in self.trees}
        return len(acts)

    def get_state_dict(self):
        return {
            'epoch': self.epoch,
            'trees': [
                {
                    'hidden_dim': t.hidden_dim,
                    'head_dropout': t.head_dropout,
                    'head_activation': t.head_activation,
                    'fitness': t.fitness,
                    'age': t.age,
                    'best_test_accuracy': t.best_test_accuracy,
                    'did_mutate': t.did_mutate,
                    'recycled': t.recycled
                } for t in self.trees
            ],
            'config': self.config
        }

# =============
# --- MAIN ---
# =============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--initial_trees', type=int, default=10)
    parser.add_argument('--max_trees', type=int, default=20)
    parser.add_argument('--min_trees', type=int, default=6)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--checkpoint_every', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--head_dropout', type=float, default=0.2)
    parser.add_argument('--head_activation', type=str, default='relu')
    parser.add_argument('--plant_every', type=int, default=2)
    parser.add_argument('--prune_every', type=int, default=2)
    args = parser.parse_args()

    config = vars(args)
    print("\n--- Using Configuration ---")
    for k,v in config.items():
        print(f"{k}: {v}")
    print("---------------------------\n")

    os.makedirs(config['output_dir'], exist_ok=True)
    checkpoints_dir = os.path.join(config['output_dir'], 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    train_loader = DummyLoader(batch_size=args.batch_size, batches=50)
    tracker = MetricsTracker()
    forest = ForestEcosystem(config)
    print(f"Initial trees: {len(forest.trees)}")

    for epoch in range(args.epochs):
        forest.epoch = epoch + 1
        # Ewolucja i mock trening:
        for t in forest.trees:
            t.fitness *= 1 + random.uniform(-0.03, 0.05)
            reward_tree(t, {'test_accuracy': random.uniform(0.3,0.9)}, config, forest)
        # Mutacje/przycinanie/rozrost co plant_every/prune_every
        if (epoch+1) % args.plant_every == 0 or (epoch+1) % args.prune_every == 0:
            forest.grow_forest()
            print(f"[Epoch {epoch+1}] Trees: {len(forest.trees)}, Diversity: {forest.compute_diversity()}")

        avg_fitness = np.mean([t.fitness for t in forest.trees])
        diversity = forest.compute_diversity()
        mock_train_loss = 1.0 - min(epoch / (args.epochs or 1), 1.0) + random.uniform(0, 0.1)
        tracker.update(epoch+1, {
            'epoch': epoch+1,
            'num_trees': len(forest.trees),
            'architecture_diversity': diversity,
            'avg_fitness': avg_fitness,
            'train_loss': mock_train_loss,
            'train_accuracy': 0.75 + random.uniform(-0.05, 0.08),
            'test_accuracy': 0.5 + random.uniform(-0.1, 0.09),
        })

        if args.checkpoint_every and ((epoch + 1) % args.checkpoint_every == 0 or (epoch + 1 == args.epochs)):
            checkpoint_path = os.path.join(checkpoints_dir, f'forest_checkpoint_epoch{epoch+1}.pt')
            torch.save(forest.get_state_dict(), checkpoint_path)
            print(f"[Checkpoint] Saved ecosystem state: {checkpoint_path}")

    print("\n--- Training Complete! ---")
    metrics_path = Path(config['output_dir']) / "metrics.json"
    tracker.save(metrics_path)
    curves_path = Path(config['output_dir']) / "learning_curves.png"
    tracker.plot(curves_path)
    report_path = os.path.join(config['output_dir'], "final_report.md")
    with open(report_path, "w") as f:
        f.write(f"# Training Report\n\n")
        f.write(f"- Epochs: {config['epochs']}\n")
        f.write(f"- Batch size: {config['batch_size']}\n")
        f.write(f"- Final number of trees: {len(forest.trees)}\n")
        f.write(f"- Max diversity: {max(tracker.data['architecture_diversity']) if tracker.data['architecture_diversity'] else None}\n")

    print(f"Metrics: {metrics_path}\nCurves: {curves_path}\nReport: {report_path}")

if __name__ == "__main__":
    main()
