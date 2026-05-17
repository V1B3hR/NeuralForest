import os
import sys
import json
import time
import argparse
import random
from collections import deque
from pathlib import Path
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== REPRODUCIBILITY ====================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

# ==================== MULCH ====================
class PrioritizedMulch:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    def add(self, x, y, priority):
        self.buffer.append((x.detach().cpu(), y.detach().cpu(), priority))
    def __len__(self):
        return len(self.buffer)

# ==================== TREE & FOREST ====================
class TreeArch:
    def __init__(self, num_layers, hidden_dim, activation, dropout, normalization, residual):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.residual = residual

class TreeNet(nn.Module):
    _ids = 0
    def __init__(self, input_dim, arch):
        super().__init__()
        self.id = TreeNet._ids
        TreeNet._ids += 1
        self.arch = arch
        layers = []
        last_dim = input_dim
        for i in range(arch.num_layers):
            layers.append(nn.Linear(last_dim, arch.hidden_dim))
            if arch.normalization == 'layer':
                layers.append(nn.LayerNorm(arch.hidden_dim))
            if arch.normalization == 'batch':
                layers.append(nn.BatchNorm1d(arch.hidden_dim))
            if arch.activation == 'relu':
                layers.append(nn.ReLU())
            elif arch.activation == 'tanh':
                layers.append(nn.Tanh())
            elif arch.activation == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            if arch.dropout > 0:
                layers.append(nn.Dropout(arch.dropout))
            last_dim = arch.hidden_dim
        self.net = nn.Sequential(*[l for l in layers if callable(l) or isinstance(l, nn.Module)])
        self.out = nn.Linear(last_dim, 1)
        self.epoch_age = 0
        self.fitness = 5.0
        self.age = 0
    def forward(self, x):
        h = x
        for layer in self.net:
            h = layer(h) if isinstance(layer, nn.Module) else layer(h)
        return self.out(h)
    def update_fitness(self, v):
        self.fitness = max(0.8 * self.fitness + 0.2 * v, 0.0)

class ForestEcosystem(nn.Module):
    def __init__(self, input_dim, max_trees):
        super().__init__()
        self.input_dim = input_dim
        self.max_trees = max_trees
        self.trees = nn.ModuleList()
        self.mulch = PrioritizedMulch(capacity=2000)
    def num_trees(self):
        return len(self.trees)
    def _plant_tree(self, arch):
        if len(self.trees) < self.max_trees:
            self.trees.append(TreeNet(self.input_dim, arch))
    def _prune_trees(self, ids, min_keep=3):
        self.trees = nn.ModuleList([t for t in self.trees if t.id not in ids])
        while len(self.trees) < min_keep:
            self.trees.append(TreeNet(self.input_dim, TreeArch(2, 256, 'relu', 0.1, 'none', False)))
    def apply_bark_gradient_mask(self):
        pass
    def state_dict(self):
        return super().state_dict()
    def update_ages(self):
        for t in self.trees:
            t.age += 1

# ==================== TASK HEAD ====================
class EnhancedTaskHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = getattr(F, activation, F.relu)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)

# ==================== METRICS ====================
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
        plt.figure()
        plt.plot(self.metrics["epoch"], self.metrics["train_accuracy"], label="Train Acc")
        plt.plot(self.metrics["epoch"], self.metrics["test_accuracy"], label="Test Acc")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("NeuralForest CIFAR-10 Accuracy")
        plt.savefig(path)
        plt.close()

def min_arch_diversity(forest):
    arch_signatures = set()
    for tree in forest.trees:
        arch = tree.arch
        sig = (arch.num_layers, arch.hidden_dim, arch.activation, arch.normalization)
        arch_signatures.add(sig)
    return len(arch_signatures)

def safe_tree_features(forest, x, output_dim_per_tree):
    tree_outputs = []
    for tree in forest.trees:
        out = tree(x)
        if out.ndim == 1: out = out.unsqueeze(1)
        if out.shape[1] == 1 and output_dim_per_tree > 1:
            out = out.repeat(1, output_dim_per_tree)
        if out.shape[1] < output_dim_per_tree:
            pad = torch.zeros(out.shape[0], output_dim_per_tree - out.shape[1], device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=1)
        if out.shape[1] > output_dim_per_tree:
            out = out[:, :output_dim_per_tree]
        tree_outputs.append(out)
    return torch.cat(tree_outputs, dim=1)

def get_cifar10(batch_size=32, num_workers=2):
    import torchvision
    import torchvision.transforms as transforms
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
    ])
    trans_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=trans_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Full Training Script (NeuralForest)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--checkpoint_every', type=int, default=10)
    parser.add_argument('--max_trees', type=int, default=8)
    parser.add_argument('--output_dim_per_tree', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='training_demos/results/cifar10_full')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    return args

def rebuild_task_head(forest, args, device):
    task_head_input_dim = forest.num_trees() * args.output_dim_per_tree
    return EnhancedTaskHead(
        input_dim=task_head_input_dim,
        hidden_dim=256,
        num_classes=10,
        dropout=0.3,
        activation='relu'
    ).to(device)

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

    with open(results_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"Device: {device}, Epochs: {args.epochs}, Batch: {args.batch_size}, Trees: {args.max_trees}, OutDimPerTree: {args.output_dim_per_tree}")
    train_loader, test_loader = get_cifar10(args.batch_size, num_workers=2)
    input_dim = 32 * 32 * 3

    forest = ForestEcosystem(input_dim=input_dim, max_trees=args.max_trees).to(device)
    initial_trees = min(8, args.max_trees)
    for _ in range(initial_trees):
        arch = TreeArch(
            num_layers=np.random.randint(2, 5),
            hidden_dim=int(np.random.choice([256, 512, 768, 1024])),
            activation=str(np.random.choice(['relu', 'tanh'])),
            dropout=float(np.random.uniform(0.05, 0.3)),
            normalization=str(np.random.choice(['none', 'layer', 'batch'])),
            residual=False,
        )
        forest._plant_tree(arch)
    for tree in forest.trees:
        tree.epoch_age = 0

    task_head = rebuild_task_head(forest, args, device)

    def build_optimizer():
        return torch.optim.Adam(
            list(forest.parameters()) + list(task_head.parameters()), lr=0.001, weight_decay=5e-4
        )

    optimizer = build_optimizer()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.6)
    criterion = nn.CrossEntropyLoss()
    metrics_tracker = MetricsTracker()
    best_test_acc = 0.0

    def flatten_images(images):
        return images.view(images.size(0), -1)

    for epoch in range(1, args.epochs + 1):
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
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            forest.apply_bark_gradient_mask()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            with torch.no_grad():
                for tree in forest.trees:
                    tree.update_fitness(loss.item() * np.random.uniform(0.9, 1.1))
            should_add_to_mulch = (len(forest.mulch) < forest.mulch.capacity) or (batch_idx % 10 == 0)
            if should_add_to_mulch:
                with torch.no_grad():
                    for i in range(min(len(x_flat), 8)):
                        priority = loss.item()
                        forest.mulch.add(x_flat[i], labels[i].float().unsqueeze(0), priority)
        forest.update_ages()
        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # TEST
        forest.eval()
        task_head.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                x_flat = flatten_images(images)
                tree_features = safe_tree_features(forest, x_flat, args.output_dim_per_tree)
                logits = task_head(tree_features)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_loss = total_loss / len(test_loader)
        test_acc = 100.0 * correct / total
        avg_fitness = sum(t.fitness for t in forest.trees) / len(forest.trees)
        arch_diversity = min_arch_diversity(forest)
        num_trees = forest.num_trees()
        memory_size = len(forest.mulch)
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
        print(f"[Epoch {epoch}] Train acc: {train_acc:.2f}% | Test acc: {test_acc:.2f}% | Trees: {num_trees} | ArchDiv: {arch_diversity}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'forest': forest.state_dict(),
                'task_head': task_head.state_dict(),
                'test_acc': test_acc,
                'num_trees': num_trees
            }, checkpoints_dir / "best_model.pt")

        if (epoch % args.checkpoint_every == 0) or (epoch == args.epochs):
            torch.save({
                'epoch': epoch,
                'forest': forest.state_dict(),
                'task_head': task_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_acc': test_acc,
                'num_trees': num_trees
            }, checkpoints_dir / f"model_epoch{epoch}.pt")

        # Pruning (co 7 epok, zostaw minimum 3 drzewa, jeśli mamy >5 i mało fit)
        if (epoch % 7 == 0 and forest.num_trees() > 5):
            weak_trees = [t.id for t in forest.trees if t.age > 10 and t.fitness < 2.5]
            if weak_trees and arch_diversity > 2:
                num_to_prune = min(3, len(weak_trees))
                forest._prune_trees(weak_trees[:num_to_prune], min_keep=3)
                task_head = rebuild_task_head(forest, args, device)
                optimizer = build_optimizer()
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.6)
                print(f"Pruned {num_to_prune} weak trees (arch.div:{arch_diversity})")
        # Planting new trees for diversity
        if ((epoch % 13 == 0 and forest.num_trees() < args.max_trees) or arch_diversity <= 2):
            arch = TreeArch(
                num_layers=int(np.random.randint(2, 5)),
                hidden_dim=int(np.random.choice([256, 512, 768, 1024])),
                activation=str(np.random.choice(['relu', 'tanh'])),
                dropout=float(np.random.uniform(0.05, 0.3)),
                normalization=str(np.random.choice(['none', 'layer', 'batch'])),
                residual=False
            )
            forest._plant_tree(arch)
            task_head = rebuild_task_head(forest, args, device)
            optimizer = build_optimizer()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.6)
            print(f"Planted new tree (total: {forest.num_trees()}, arch.div: {min_arch_diversity(forest)})")

        scheduler.step()

    metrics_tracker.save(metrics_json)
    metrics_tracker.plot(learning_png)
    with open(report_md, "w") as f:
        f.write(f"# NeuralForest CIFAR-10 Training Report\n\n")
        f.write(f"**Best test accuracy**: {best_test_acc:.2f}%\n\n")
    print(f"Done. Best Test Acc: {best_test_acc:.2f}%. Report in {results_dir}.")

if __name__ == "__main__":
    main()
