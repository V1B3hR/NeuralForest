import sys
import os
import argparse
import yaml
from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# -- UTILS (Tracker) --
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
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot(self, path):
        keys = list(self.history.keys())
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        idx = 0
        for k in keys:
            if k == 'epoch': continue
            axes[idx//2, idx%2].plot(self.history['epoch'], self.history[k], label=k)
            axes[idx//2, idx%2].set_title(k)
            axes[idx//2, idx%2].legend()
            idx += 1
            if idx >= 4: break
        fig.tight_layout()
        plt.savefig(path)
        plt.close()

# -- UTILS (Mock DatasetLoader, CIFAR-10 must be available for true use) --
class DatasetLoader:
    @staticmethod
    def get_cifar10(batch_size=64):
        try:
            import torchvision
            import torchvision.transforms as transforms
        except ImportError:
            # fallback: empty mock
            return [], []
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        return trainloader, testloader

# -- LIBERAL EVOLUTIONARY UTILS --

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
        if config.get('pollination_on_low_diversity', False) and self.compute_diversity() < 2:
            parent = random.choice(self.trees)
            for _ in range(2):
                t = Tree(config)
                t.hidden_dim = parent.hidden_dim
                t.head_dropout = parent.head_dropout + random.uniform(-0.03, 0.03)
                t.head_activation = parent.head_activation
                self.trees.append(t)
        if len(self.trees) < config['max_trees']:
            t = Tree(config)
            adaptive_mutation(t, self, config)
            self.trees.append(t)
        if len(self.trees) > config['min_trees']:
            sorted_trees = sorted(self.trees, key=lambda x: x.fitness)
            for i in range(int(0.2*len(self.trees))):
                if sorted_trees[i].age > 8:
                    self.trees.remove(sorted_trees[i])
        for t in self.trees:
            t.age += 1
            reward_tree(t, {'test_accuracy': random.uniform(0,1)}, config)

    def compute_diversity(self):
        activations = {t.head_activation for t in self.trees}
        return len(activations)

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

def load_config(args):
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    for k,v in vars(args).items():
        if v is not None:
            config[k] = v
    config.setdefault('initial_trees', 10)
    config.setdefault('max_trees', 20)
    config.setdefault('min_trees', 6)
    config.setdefault('epochs', 100)
    config.setdefault('batch_size', 64)
    config.setdefault('optimizer_type', 'adam')
    config.setdefault('hidden_dim', 512)
    config.setdefault('head_dropout', 0.2)
    config.setdefault('head_activation', 'relu')
    config.setdefault('plant_every', 5)
    config.setdefault('prune_every', 5)
    config.setdefault('checkpoint_every', 20)
    config.setdefault('device', 'cpu')
    return config

def main():
    # Ensure repo root in sys.path (for workflow/offline use)
    top = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if top not in sys.path:
        sys.path.insert(0, top)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='YAML config path')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--initial_trees', type=int, default=None)
    parser.add_argument('--max_trees', type=int, default=None)
    parser.add_argument('--min_trees', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='training_demos/results/liberal_forest')
    parser.add_argument('--checkpoint_every', type=int, default=None, help='Co ile epok zapisywać checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Urządzenie docelowe (cpu, cuda)')
    args = parser.parse_args()

    config = load_config(args)
    print("\n--- Using Configuration ---")
    for k,v in config.items():
        print(f"{k}: {v}")
    print("---------------------------\n")

    if 'seed' in config:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])

    train_loader, test_loader = DatasetLoader.get_cifar10(batch_size=config['batch_size'])
    tracker = MetricsTracker()
    forest = ForestEcosystem(config)
    print(f"Initial trees: {len(forest.trees)}")
    device = config.get('device', 'cpu')
    print(f"Training device target: {device}")

    os.makedirs(config['output_dir'], exist_ok=True)
    checkpoints_dir = os.path.join(config['output_dir'], 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    for epoch in range(config['epochs']):
        forest.epoch = epoch + 1
        # Mock "trening" ekosystemu i mutacje
        for t in forest.trees:
            t.fitness *= 1 + random.uniform(-0.03, 0.05)
            reward_tree(t, {'test_accuracy': random.uniform(0.3,0.9)}, config)
        if (epoch+1) % config['plant_every'] == 0 or (epoch+1) % config['prune_every'] == 0:
            forest.grow_forest()
            print(f"[Epoch {epoch+1}] Trees: {len(forest.trees)}, Diversity: {forest.compute_diversity()}")

        avg_fitness = np.mean([t.fitness for t in forest.trees])
        # --- MOCKUJEMY stratę (train_loss) do wykresu ---
        mock_train_loss = 1.0 - min(epoch / (config['epochs'] or 1), 1.0) + random.uniform(0, 0.1)
        tracker.update(epoch+1, {
            'epoch': epoch+1,
            'num_trees': len(forest.trees),
            'architecture_diversity': forest.compute_diversity(),
            'avg_fitness': avg_fitness,
            'train_loss': mock_train_loss,
            'train_accuracy': 0.75 + random.uniform(-0.05, 0.08),
            'test_accuracy': 0.5 + random.uniform(-0.1, 0.09),
        })

        if 'checkpoint_every' in config and config['checkpoint_every']:
            if ((epoch + 1) % config['checkpoint_every'] == 0) or (epoch + 1 == config['epochs']):
                checkpoint_path = os.path.join(checkpoints_dir, f'forest_checkpoint_epoch{epoch+1}.pt')
                torch.save(forest.get_state_dict(), checkpoint_path)
                print(f"[Checkpoint] Zapisano stan ekosystemu: {checkpoint_path}")

    print("\n--- Training Complete! ---")
    tracker.save(Path(config['output_dir']) / "metrics.json")
    tracker.plot(Path(config['output_dir']) / "learning_curves.png")
    report_path = os.path.join(config['output_dir'], "final_report.md")
    with open(report_path, "w") as f:
        f.write(f"# Training Report\n\n")
        f.write(f"- Epochs: {config['epochs']}\n")
        f.write(f"- Batch size: {config['batch_size']}\n")
        f.write(f"- Final number of trees: {len(forest.trees)}\n")
        f.write(f"- Max diversity: {max(tracker.data['architecture_diversity']) if tracker.data['architecture_diversity'] else None}\n")

if __name__ == "__main__":
    main()
