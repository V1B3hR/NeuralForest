import matplotlib
matplotlib.use('Agg')

import sys
import os
import argparse
import json
import time
from pathlib import Path
import traceback
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ===== Reproducibility =====
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

# ========== Mulch Buffer ==========
class PrioritizedMulch:
    def __init__(self, capacity=2000):
        self.buffer = []
        self.capacity = capacity
    def add(self, x, y, priority):
        if len(self.buffer) < self.capacity:
            self.buffer.append((x.detach().cpu(), y.detach().cpu(), priority))
        else:  # FIFO
            self.buffer.pop(0)
            self.buffer.append((x.detach().cpu(), y.detach().cpu(), priority))
    def __len__(self):
        return len(self.buffer)

# ========== Tree/Fores Simulation ==========
class TreeArch:
    def __init__(self, num_layers, hidden_dim, activation='relu', dropout=0.0, normalization='none', residual=False):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.residual = residual

class DummyTree(nn.Module):
    _ids = 0
    def __init__(self, input_dim, arch):
        super().__init__()
        self.id = DummyTree._ids
        DummyTree._ids += 1
        self.arch = arch
        self.fc = nn.Linear(input_dim, arch.hidden_dim)
        self.out = nn.Linear(arch.hidden_dim, 1)
        self.epoch_age = 0
        self.fitness = 5.0
        self.age = 0
    def forward(self, x):
        act = torch.relu if self.arch.activation == 'relu' else torch.tanh
        h = act(self.fc(x))
        return self.out(h)
    def update_fitness(self, v):
        self.fitness = max(0.8 * self.fitness + 0.2 * v, 0.)

class ForestEcosystem(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_trees, **kwargs):
        super().__init__()
        self._input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_trees = max_trees
        self.trees = nn.ModuleList()
        self.mulch = PrioritizedMulch(capacity=2000)
    def num_trees(self):
        return len(self.trees)
    def _plant_tree(self, arch):
        if len(self.trees) < self.max_trees:
            self.trees.append(DummyTree(self._input_dim, arch))
    def _prune_trees(self, ids, min_keep=3):
        keep = [t for t in self.trees if (t.id not in ids)]
        while len(keep) < min_keep and self.trees:
            keep.append(self.trees[0])
        self.trees = nn.ModuleList(keep)
    def apply_bark_gradient_mask(self):
        pass
    def state_dict(self):
        return {}
    def train(self, mode=True):
        super().train(mode)
    def eval(self):
        super().eval()
    def update_ages(self):
        for t in self.trees:
            t.age += 1

# ========== Task Head ==========
class EnhancedTaskHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2, activation='relu', use_skip=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# ========== Metrics Tracker ==========
class MetricsTracker:
    def __init__(self):
        self.metrics = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
            "num_trees": [],
            "avg_fitness": [],
            "architecture_diversity": [],
            "memory_size": [],
        }
    def update(self, epoch, d):
        self.metrics["epoch"].append(epoch)
        for k in d:
            self.metrics[k].append(d[k])
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
    def plot(self, path):
        import matplotlib.pyplot as plt
        plt.figure()
        try:
            plt.plot(self.metrics["epoch"], self.metrics["train_accuracy"], label="Train Acc")
            plt.plot(self.metrics["epoch"], self.metrics["test_accuracy"], label="Test Acc")
            plt.legend()
        except:
            plt.text(0.5, 0.5, "No Data", ha='center', va='center')
        plt.savefig(path)
        plt.close()

# ========== DATA ==========
def get_cifar10(batch_size=16, num_workers=0):
    import torchvision
    import torchvision.transforms as transforms
    trans = transforms.Compose([transforms.ToTensor(),])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trans)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def safe_tree_features(forest, x, output_dim_per_tree):
    tree_outputs = []
    for tree in forest.trees:
        out = tree(x)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        if out.shape[1] == 1 and output_dim_per_tree > 1:
            out = out.repeat(1, output_dim_per_tree)
        if out.shape[1] < output_dim_per_tree:
            pad = torch.zeros(out.shape[0], output_dim_per_tree - out.shape[1], device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=1)
        if out.shape[1] > output_dim_per_tree:
            out = out[:, :output_dim_per_tree]
        tree_outputs.append(out)
    return torch.cat(tree_outputs, dim=1)

def min_arch_diversity(forest):
    arch_signatures = set()
    for tree in forest.trees:
        arch = tree.arch
        sig = (arch.num_layers, arch.hidden_dim)
        arch_signatures.add(sig)
    return len(arch_signatures)

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Full Training Script (Stable Single File)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint_every', type=int, default=2)
    parser.add_argument('--max_trees', type=int, default=6)
    parser.add_argument('--output_dim_per_tree', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='training_demos/results/cifar10_full')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device(args.device)
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    metrics_json = results_dir / "metrics.json"
    learning_png = results_dir / "learning_curves.png"
    report_md = results_dir / "final_report.md"
    
    try:
        print(f"\n==== NeuralForest CIFAR-10 Training SINGLECELL ====")
        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {device}  Epochs: {args.epochs}  Batch size: {args.batch_size}")
        print(f"Max trees: {args.max_trees}, Output dim per tree: {args.output_dim_per_tree}")
        print(f"Output dir: {args.output_dir} (abs: {results_dir.resolve()})")
        with open(results_dir / "config.json", 'w') as f:
            json.dump(vars(args), f, indent=2)

        train_loader, test_loader = get_cifar10(
            batch_size=args.batch_size,
            num_workers=0
        )
        print(f"✓ Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

        forest = ForestEcosystem(
            input_dim=3072,
            hidden_dim=64,
            max_trees=args.max_trees
        ).to(device)

        initial_trees = min(3, args.max_trees)
        for i in range(initial_trees - forest.num_trees()):
            arch = TreeArch(
                num_layers=np.random.randint(2, 4),
                hidden_dim=int(np.random.choice([32, 64])),
                activation='relu',
                dropout=0.1,
                normalization='none',
                residual=False
            )
            forest._plant_tree(arch)
        for tree in forest.trees:
            tree.epoch_age = 0

        current_num_trees = forest.num_trees()
        task_head_input_dim = current_num_trees * args.output_dim_per_tree

        task_head = EnhancedTaskHead(
            input_dim=task_head_input_dim,
            hidden_dim=32,
            num_classes=10,
            dropout=0.2,
            activation='relu',
            use_skip=False
        ).to(device)
        print("✓ Models ready.")

        metrics_tracker = MetricsTracker()
        last_diversity_increase_epoch = 0 
        last_diversity_warning_epoch = -50 

        def flatten_images(images):
            return images.view(images.size(0), -1)

        best_test_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            if forest.num_trees() != current_num_trees:
                current_num_trees = forest.num_trees()
                task_head_input_dim = current_num_trees * args.output_dim_per_tree
                old_task_head = task_head
                task_head = EnhancedTaskHead(
                    input_dim=task_head_input_dim,
                    hidden_dim=32,
                    num_classes=10,
                    dropout=0.2,
                    activation='relu',
                    use_skip=False
                ).to(device)
                try:
                    if hasattr(old_task_head, 'fc2'):
                        task_head.fc2.load_state_dict(old_task_head.fc2.state_dict())
                except:
                    pass
                print(f"✓ Task head recreated for {current_num_trees} trees")

            optimizer = torch.optim.Adam(list(forest.parameters()) + list(task_head.parameters()), lr=0.003)

            forest.train()
            task_head.train()
            total_loss, correct, total = 0.0, 0, 0

            # TRAIN
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                x_flat = flatten_images(images)
                tree_features = safe_tree_features(forest, x_flat, args.output_dim_per_tree)
                logits = task_head(tree_features)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                with torch.no_grad():
                    for tree in forest.trees:
                        tree.update_fitness(loss.item() * np.random.uniform(0.9, 1.1))
                # --- FIX: mulch.capacity, not max_size!
                should_add_to_mulch = (len(forest.mulch) < forest.mulch.capacity) or (batch_idx % 5 == 0)
                if should_add_to_mulch:
                    with torch.no_grad():
                        for i in range(min(len(x_flat), 5)):
                            priority = loss.item()
                            forest.mulch.add(x_flat[i], labels[i].float().unsqueeze(0), priority)
                if batch_idx > 10: break  # demo run: you can remove for full train

            forest.update_ages()
            train_loss = total_loss / (batch_idx+1)
            train_acc = 100.0 * correct / total

            # TEST
            forest.eval()
            task_head.eval()
            total_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for test_idx, (images, labels) in enumerate(test_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    x_flat = flatten_images(images)
                    tree_features = safe_tree_features(forest, x_flat, args.output_dim_per_tree)
                    logits = task_head(tree_features)
                    loss = F.cross_entropy(logits, labels)
                    total_loss += loss.item()
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    if test_idx > 3: break  # demo run
            test_loss = total_loss / (test_idx+1)
            test_acc = 100.0 * correct / total
            avg_fitness = sum(t.fitness for t in forest.trees) / len(forest.trees)
            num_trees = forest.num_trees()
            arch_diversity = min_arch_diversity(forest)
            memory_size = len(forest.mulch)

            # --- DIVERSITY TRACKING (SAFE!) ---
            arch_div_history = metrics_tracker.metrics.get("architecture_diversity", [])
            if epoch > 1:
                prev_diversity = arch_div_history[-1] if arch_div_history else arch_diversity
                if arch_diversity > prev_diversity:
                    last_diversity_increase_epoch = epoch
                elif epoch - last_diversity_increase_epoch > 50 and epoch - last_diversity_warning_epoch >= 50:
                    print(f"⚠️  Warning: Architecture diversity has not increased for {epoch - last_diversity_increase_epoch} epochs (current: {arch_diversity})")
                    last_diversity_warning_epoch = epoch

            metrics_tracker.update(epoch, {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "num_trees": num_trees,
                "avg_fitness": avg_fitness,
                "architecture_diversity": arch_diversity,
                "memory_size": memory_size
            })

            print(f"[EPOCH {epoch}] Train: {train_acc:.2f}%  Test: {test_acc:.2f}%  Trees: {num_trees}  ArchDiv:{arch_diversity}")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'forest': {}, # forest.state_dict()
                    'task_head': task_head.state_dict(),
                    'test_acc': test_acc,
                    'num_trees': num_trees,
                }, checkpoints_dir / "best_model.pt")

            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                torch.save({
                    'epoch': epoch,
                    'forest': {}, # forest.state_dict()
                    'task_head': task_head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'num_trees': num_trees,
                }, checkpoints_dir / f"model_epoch{epoch}.pt")

        metrics_tracker.save(metrics_json)
        metrics_tracker.plot(learning_png)

        with open(report_md, "w") as f:
            f.write("# NeuralForest CIFAR-10 Training Report\n\n")
            f.write(f"**Best test accuracy**: {best_test_acc:.2f}%\n\n")
        print(f"✓ Training done and all results saved to {results_dir}.")

    except Exception as e:
        error_msg = f"ERROR: Training failed!\n{traceback.format_exc()}"
        print(error_msg)
        with open(results_dir / "error.log", 'w') as f:
            f.write(error_msg)
        with open(report_md, 'w') as f:
            f.write("# Training Failed\n\n")
            f.write(f"## Error\n\n```\n{str(e)}\n```\n\n")
            f.write("See error.log for full traceback.\n")
        # --- dummy metrics etc, like before ---
        dummy = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
            "num_trees": [],
            "avg_fitness": [],
            "architecture_diversity": [],
            "memory_size": [],
        }
        with open(metrics_json, "w") as f:
            json.dump(dummy, f, indent=2)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.text(0.5, 0.5, "No Data", ha='center', va='center')
        plt.axis('off')
        plt.savefig(learning_png)
        plt.close()
        print("✓ dummy metrics.json and learning_curves.png saved (failure)")
        raise

if __name__ == "__main__":
    main()
