"""CIFAR-10 Full Training Script with Real Training Loop."""

import sys
import os
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from NeuralForest import ForestEcosystem, TreeArch, DEVICE
from ecosystem_simulation import EcosystemSimulator
from training_demos.layer_wise_optimizer import LayerWiseConfig, LayerWiseOptimizer
from training_demos.enhanced_task_head import EnhancedTaskHead
from training_demos.utils import DatasetLoader, MetricsTracker

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Full Training Script')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint_every', type=int, default=20)
    parser.add_argument('--max_trees', type=int, default=75)
    parser.add_argument('--output_dir', type=str, default='training_demos/results/cifar10_full')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)

def flatten_images(images):
    """Flatten CIFAR-10 images from [B, 3, 32, 32] to [B, 3072]."""
    return images.view(images.size(0), -1)

def train_epoch(forest, task_head, simulator, optimizer, train_loader, device, epoch):
    """Train for one epoch."""
    forest.train()
    task_head.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Flatten images for forest
        x_flat = flatten_images(images)
        
        # Forward pass through forest
        forest_out, weights, _ = forest.forward_forest(x_flat, top_k=3)
        
        # Forward pass through task head
        logits = task_head(forest_out)
        
        # Calculate loss
        loss = F.cross_entropy(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Apply bark gradient mask
        forest.apply_bark_gradient_mask()
        
        # Optimizer step
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Store experiences in memory (every 10 batches)
        if batch_idx % 10 == 0:
            with torch.no_grad():
                for i in range(min(len(x_flat), 5)):
                    priority = loss.item()
                    forest.mulch.add(x_flat[i], labels[i].float().unsqueeze(0), priority)
    
    # Update tree ages
    forest.update_ages()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def evaluate(forest, task_head, test_loader, device):
    """Evaluate on test set."""
    forest.eval()
    task_head.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Flatten images
            x_flat = flatten_images(images)
            
            # Forward pass
            forest_out, _, _ = forest.forward_forest(x_flat, top_k=3)
            logits = task_head(forest_out)
            
            # Calculate loss
            loss = F.cross_entropy(logits, labels)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def count_unique_architectures(forest):
    """Count unique tree architectures."""
    arch_signatures = set()
    for tree in forest.trees:
        if hasattr(tree, 'arch'):
            sig = (
                tree.arch.num_layers,
                tree.arch.hidden_dim,
                tree.arch.activation,
                tree.arch.dropout,
            )
            arch_signatures.add(sig)
    return len(arch_signatures)

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device(args.device)
    
    print(f"🌲 NeuralForest CIFAR-10 Training")
    print(f"=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max trees: {args.max_trees}")
    print(f"=" * 70)
    
    # Create results directory
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Save config
    with open(results_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load dataset
    print("\n📦 Loading CIFAR-10 dataset...")
    train_loader, test_loader = DatasetLoader.get_cifar10(
        batch_size=args.batch_size, 
        num_workers=2
    )
    print(f"✓ Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Create forest
    print("\n🌱 Creating NeuralForest...")
    forest = ForestEcosystem(
        input_dim=3072,  # 32x32x3 flattened
        hidden_dim=512,
        max_trees=args.max_trees,
        enable_graveyard=True
    ).to(device)
    
    # Plant initial trees
    initial_trees = min(6, args.max_trees)
    for i in range(initial_trees - forest.num_trees()):
        forest._plant_tree()
    
    # Initialize epoch_age for all trees
    for tree in forest.trees:
        tree.epoch_age = 0
    
    print(f"✓ Forest created with {forest.num_trees()} trees")
    
    # Create task head
    print("\n🎯 Creating task head...")
    task_head = EnhancedTaskHead(
        input_dim=512,
        hidden_dim=64,
        num_classes=10,
        dropout=0.2,
        activation='relu',
        use_skip=False
    ).to(device)
    print("✓ Task head created")
    
    # Create optimizer configuration
    opt_config = LayerWiseConfig(
        base_lr=0.01,
        min_lr=0.0001,
        half_life=60.0,
        fitness_scale=5.0,
        fitness_aware=False,
        warmup_epochs=5,
        schedule='cosine',
        total_epochs=args.epochs,
        weight_decay=1e-4,
        optimizer_type='adam'
    )
    opt_factory = LayerWiseOptimizer(opt_config)
    
    # Create ecosystem simulator
    simulator = EcosystemSimulator(
        forest,
        competition_fairness=0.3,
        selection_threshold=0.25,
        learning_rate=0.01,
        enable_replay=True,
        enable_anchors=True,
        device=device
    )
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    print("\n🚀 Starting training...")
    print("=" * 70)
    
    best_test_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Create optimizer for this epoch
        optimizer = opt_factory.create_optimizer(forest, task_head, epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            forest, task_head, simulator, optimizer, train_loader, device, epoch
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(forest, task_head, test_loader, device)
        
        # Calculate fitness metrics
        avg_fitness = sum(t.fitness for t in forest.trees) / len(forest.trees)
        num_trees = forest.num_trees()
        arch_diversity = count_unique_architectures(forest)
        memory_size = len(forest.mulch)
        
        # Update metrics
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
        
        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_acc:5.2f}% loss={train_loss:.4f} | "
              f"Test: {test_acc:5.2f}% loss={test_loss:.4f} | "
              f"Trees: {num_trees:2d} | Fit: {avg_fitness:.2f}")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'forest': forest.state_dict(),
                'task_head': task_head.state_dict(),
                'test_acc': test_acc,
            }, checkpoints_dir / "best_model.pt")
        
        # Save checkpoint
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save({
                'epoch': epoch,
                'forest': forest.state_dict(),
                'task_head': task_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_acc': test_acc,
            }, checkpoints_dir / f"model_epoch{epoch}.pt")
            print(f"💾 Checkpoint saved at epoch {epoch}")
        
        # Tree management
        if epoch % 10 == 0:
            # Prune weak trees
            if forest.num_trees() > 4:
                weak_trees = [t.id for t in forest.trees if t.age > 30 and t.fitness < 2.0]
                if weak_trees:
                    forest._prune_trees(weak_trees[:2], min_keep=3)
                    print(f"🪓 Pruned {len(weak_trees[:2])} weak trees")
        
        if epoch % 15 == 0:
            # Plant new tree
            if forest.num_trees() < args.max_trees:
                arch = TreeArch(
                    num_layers=np.random.randint(2, 5),
                    hidden_dim=512,
                    activation=np.random.choice(['relu', 'gelu', 'tanh']),
                    dropout=np.random.uniform(0.0, 0.2),
                    normalization=np.random.choice(['none', 'layer']),
                    residual=np.random.choice([True, False])
                )
                forest._plant_tree(arch)
                print(f"🌱 Planted new tree (total: {forest.num_trees()})")
        
        # Update tree ages
        opt_factory.update_tree_ages(forest)
    
    print("\n" + "=" * 70)
    print("✅ Training completed!")
    print("=" * 70)
    
    # Save final metrics
    print("\n💾 Saving results...")
    metrics_tracker.save(results_dir / "metrics.json")
    metrics_tracker.plot(results_dir / "learning_curves.png")
    
    # Generate final report
    with open(results_dir / "final_report.md", "w") as f:
        f.write("# NeuralForest CIFAR-10 Training Report\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- **Epochs**: {args.epochs}\n")
        f.write(f"- **Batch size**: {args.batch_size}\n")
        f.write(f"- **Max trees**: {args.max_trees}\n")
        f.write(f"- **Device**: {args.device}\n\n")
        f.write("## Final Results\n\n")
        f.write(f"- **Best test accuracy**: {best_test_acc:.2f}%\n")
        f.write(f"- **Final train accuracy**: {train_acc:.2f}%\n")
        f.write(f"- **Final test accuracy**: {test_acc:.2f}%\n")
        f.write(f"- **Final number of trees**: {num_trees}\n")
        f.write(f"- **Architecture diversity**: {arch_diversity} unique types\n")
        f.write(f"- **Memory size**: {memory_size} samples\n")
        f.write(f"- **Average fitness**: {avg_fitness:.2f}\n\n")
        f.write("## Learning Curves\n\n")
        f.write("![Learning Curves](learning_curves.png)\n\n")
        f.write("## Tree Evolution\n\n")
        f.write(f"Started with {initial_trees} trees, ended with {num_trees} trees.\n")
        f.write(f"Developed {arch_diversity} unique architectural patterns.\n\n")
        f.write("---\n")
        f.write("*Generated by NeuralForest Training System*\n")
    
    print(f"✓ Results saved to: {results_dir}")
    print(f"✓ Best test accuracy: {best_test_acc:.2f}%")
    print("\n🎉 All done!")

if __name__ == "__main__":
    main()
