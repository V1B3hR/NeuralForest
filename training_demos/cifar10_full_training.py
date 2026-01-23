import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Utils ---
class MetricsTracker:
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'num_trees': [],
            'avg_fitness': [],
            'architecture_diversity': [],
            'memory_size': [],
        }

    def update(self, epoch, metrics):
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot(self, save_path):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['epoch'], self.history['test_loss'], label='Test')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 1].plot(self.history['epoch'], self.history['train_accuracy'], label='Train')
        axes[0, 1].plot(self.history['epoch'], self.history['test_accuracy'], label='Test')
        axes[0, 1].set_title('Accuracy Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 2].plot(self.history['epoch'], self.history['avg_fitness'])
        axes[0, 2].set_title('Average Tree Fitness')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Fitness')
        axes[0, 2].grid(True)
        axes[1, 0].plot(self.history['epoch'], self.history['num_trees'])
        axes[1, 0].set_title('Number of Trees')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True)
        axes[1, 1].plot(self.history['epoch'], self.history['architecture_diversity'])
        axes[1, 1].set_title('Architecture Diversity')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Unique Architectures')
        axes[1, 1].grid(True)
        axes[1, 2].plot(self.history['epoch'], self.history['memory_size'])
        axes[1, 2].set_title('Memory Usage')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Samples Stored')
        axes[1, 2].grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @property
    def data(self):
        return self.history

# --- Dummy Evolutive Forest: minimal representative for script demo ---
class DummyTree:
    def __init__(self):
        self.fitness = np.random.uniform(1, 10)
        self.architecture = np.random.choice(['A', 'B', 'C', 'D', 'E'])

class DummyForest:
    def __init__(self, num=6):
        self.trees = [DummyTree() for _ in range(num)]

    def num_trees(self):
        return len(self.trees)

    def diversity(self):
        return len(set(t.architecture for t in self.trees))

    def grow(self):
        # Simulates random mutational expansion
        if np.random.rand() > 0.5 and self.num_trees() < 20:
            self.trees.append(DummyTree())
        # Simulate pruning
        if self.num_trees() > 6 and np.random.rand() > 0.9:
            self.trees.pop(np.random.randint(self.num_trees()))
        # Randomly mutate
        for t in self.trees:
            if np.random.rand() < 0.2:
                t.fitness += np.random.normal(0, 0.2)
            if np.random.rand() < 0.15:
                t.architecture = np.random.choice(['A', 'B', 'C', 'D', 'E'])

    def avg_fitness(self):
        if not self.trees: return 0
        return float(np.mean([t.fitness for t in self.trees]))

# --- Main training script (analog: cifar10_full_training.py) ---
def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Full Training Script')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--checkpoint_every', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='training_demos/results/cifar10_full')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device(args.device)
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    with open(results_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    # === Dataset
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # === Dummy "Forest" + tracker
    forest = DummyForest(num=6)
    metrics_tracker = MetricsTracker()

    # === Dummy model for the demo (to avoid NeuralForest class dependencies)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*32*32, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # === Training loop (simplified for demo)
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss, train_acc, n = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            train_acc += (pred == y).sum().item()
            n += x.size(0)
            if n > args.batch_size*30: break # <- Full epoch = would be slow on dummy
        train_loss, train_acc = train_loss/n, 100*train_acc/n

        # Eval (short)
        model.eval()
        test_loss, test_acc, nval = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                test_loss += loss.item() * x.size(0)
                _, pred = out.max(1)
                test_acc += (pred == y).sum().item()
                nval += x.size(0)
                if nval > args.batch_size*15: break
        test_loss, test_acc = test_loss/nval, 100*test_acc/nval

        # Dummy evolutionary step
        forest.grow()

        metrics_tracker.update(epoch, {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "num_trees": forest.num_trees(),
            "avg_fitness": forest.avg_fitness(),
            "architecture_diversity": forest.diversity(),
            "memory_size": n
        })

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), checkpoints_dir / f"model_epoch{epoch}.pt")

    # === Save metrics, plot and final report
    metrics_tracker.save(results_dir / "metrics.json")
    metrics_tracker.plot(results_dir / "learning_curves.png")
    diversity_history = metrics_tracker.data.get("architecture_diversity", [])
    with open(results_dir / "final_report.md", "w") as f:
        f.write("# Training Report\n\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Final number of trees: {forest.num_trees()}\n")
        f.write(f"- Max diversity: {max(diversity_history) if diversity_history else 'N/A'}\n")

    print("Done! All outputs saved to:", results_dir)

if __name__ == "__main__":
    main()
