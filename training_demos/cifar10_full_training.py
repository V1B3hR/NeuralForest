import matplotlib
matplotlib.use('Agg')

import sys
import os
import argparse
import json
import time
from pathlib import Path
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Full Training Script')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint_every', type=int, default=25)
    parser.add_argument('--max_trees', type=int, default=30)
    parser.add_argument('--output_dim_per_tree', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='training_demos/results/cifar10_full')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)

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
        if hasattr(tree, 'arch'):
            sig = (tree.arch.num_layers, tree.arch.hidden_dim, tree.arch.activation)
            arch_signatures.add(sig)
    return len(arch_signatures)

def write_dummy_metrics(path):
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
    with open(path, "w") as f:
        json.dump(dummy, f, indent=2)

def write_dummy_png(path):
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.text(0.5, 0.5, "No Data", ha='center', va='center')
        plt.axis('off')
        plt.savefig(path)
        plt.close()
    except Exception:
        minimal_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!bc\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with open(path, "wb") as f:
            f.write(minimal_png)

def list_result_files(output_dir, stage=""):
    try:
        out_path = Path(output_dir)
        print(f"[V7-DEBUG] {stage} Output files in {out_path.resolve()}:")
        for root, dirs, files in os.walk(out_path):
            for name in files:
                print("  -", os.path.join(root, name))
    except Exception as e:
        print(f"[V7-DEBUG] Error listing files: {e}")

def main():
    args = parse_args()
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    error_log_path = results_dir / "error.log"
    metrics_json = results_dir / "metrics.json"
    learning_png = results_dir / "learning_curves.png"
    report_md = results_dir / "final_report.md"

    try:
        set_seed(42)
        device = torch.device(args.device)
        print(f"\n==== NeuralForest CIFAR-10 Training V7 ====")
        print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {device}  Epochs: {args.epochs}  Batch size: {args.batch_size}")
        print(f"Max trees: {args.max_trees}, Output dim per tree: {args.output_dim_per_tree}")
        print(f"Output dir: {args.output_dir} (abs: {results_dir.resolve()})")
        with open(results_dir / "config.json", 'w') as f:
            json.dump(vars(args), f, indent=2)

        list_result_files(results_dir, "After config saved --")

        from NeuralForest import ForestEcosystem, TreeArch, DEVICE
        from ecosystem_simulation import EcosystemSimulator
        from training_demos.layer_wise_optimizer import LayerWiseConfig, LayerWiseOptimizer
        from training_demos.enhanced_task_head import EnhancedTaskHead
        from training_demos.utils import DatasetLoader, MetricsTracker
        print("✓ All modules imported")

        train_loader, test_loader = DatasetLoader.get_cifar10(
            batch_size=args.batch_size,
            num_workers=0
        )
        print(f"✓ Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

        forest = ForestEcosystem(
            input_dim=3072,
            hidden_dim=512,
            max_trees=args.max_trees,
            enable_graveyard=True
        ).to(device)

        initial_trees = min(14, args.max_trees)
        for i in range(initial_trees - forest.num_trees()):
            arch = TreeArch(
                num_layers=np.random.randint(2, 7),
                hidden_dim=np.random.choice([256, 512, 768, 1024]),
                activation=str(np.random.choice(['relu', 'gelu', 'tanh'])),
                dropout=float(np.random.uniform(0.0, 0.4)),
                normalization=str(np.random.choice(['none', 'layer'])),
                residual=bool(np.random.choice([True, False]))
            )
            forest._plant_tree(arch)
        for tree in forest.trees:
            tree.epoch_age = 0

        current_num_trees = forest.num_trees()
        print(f"✓ Forest created with {current_num_trees} trees")
        task_head_input_dim = current_num_trees * args.output_dim_per_tree

        task_head = EnhancedTaskHead(
            input_dim=task_head_input_dim,
            hidden_dim=64,
            num_classes=10,
            dropout=0.2,
            activation='relu',
            use_skip=False
        ).to(device)
        print("✓ Task head created")

        opt_config = LayerWiseConfig(
            base_lr=0.003,
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

        simulator = EcosystemSimulator(
            forest,
            competition_fairness=0.33,
            selection_threshold=0.21,
            learning_rate=0.01,
            enable_replay=True,
            enable_anchors=True,
            device=device
        )
        metrics_tracker = MetricsTracker()
        last_diversity_increase_epoch = 0  # Track when diversity last increased
        last_diversity_warning_epoch = -50  # Track when we last warned (initialized to allow first warning at epoch 50+)

        def flatten_images(images):
            return images.view(images.size(0), -1)

        print("\n==== Starting training ====")
        best_test_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            print(f"\n[V7-TRACE] >>> EPOCH {epoch} START ({time.strftime('%H:%M:%S')})")
            if forest.num_trees() != current_num_trees:
                print(f"⚠️  Tree count changed: {current_num_trees} → {forest.num_trees()}")
                current_num_trees = forest.num_trees()
                task_head_input_dim = current_num_trees * args.output_dim_per_tree
                old_task_head = task_head
                task_head = EnhancedTaskHead(
                    input_dim=task_head_input_dim,
                    hidden_dim=64,
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

            optimizer = opt_factory.create_optimizer(forest, task_head, epoch)

            forest.train()
            task_head.train()
            total_loss, correct, total = 0.0, 0, 0

            # TRAIN
            for batch_idx, (images, labels) in enumerate(train_loader):
                print(f"[V7-TRACE] E{epoch} TRAIN batch {batch_idx+1}/{len(train_loader)} SHAPE images {images.shape} labels {labels.shape} ({time.strftime('%H:%M:%S')})")
                images = images.to(device)
                labels = labels.to(device)
                x_flat = flatten_images(images)
                tree_features = safe_tree_features(forest, x_flat, args.output_dim_per_tree)
                logits = task_head(tree_features)
                loss = F.cross_entropy(logits, labels)
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
                # Add more samples to mulch buffer more frequently
                should_add_to_mulch = (len(forest.mulch) < forest.mulch.max_size) or (batch_idx % 5 == 0)
                if should_add_to_mulch:
                    with torch.no_grad():
                        for i in range(min(len(x_flat), 10)):
                            priority = loss.item()
                            forest.mulch.add(x_flat[i], labels[i].float().unsqueeze(0), priority)
                print(f"[V7-TRACE] E{epoch} TRAIN batch {batch_idx+1}/{len(train_loader)} DONE ({time.strftime('%H:%M:%S')})")

            forest.update_ages()
            train_loss = total_loss / len(train_loader)
            train_acc = 100.0 * correct / total

            # TEST
            forest.eval()
            task_head.eval()
            total_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for test_idx, (images, labels) in enumerate(test_loader):
                    print(f"[V7-TRACE] E{epoch} TEST batch {test_idx+1}/{len(test_loader)} SHAPE images {images.shape} labels {labels.shape} ({time.strftime('%H:%M:%S')})")
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
                    print(f"[V7-TRACE] E{epoch} TEST batch {test_idx+1}/{len(test_loader)} DONE ({time.strftime('%H:%M:%S')})")
            test_loss = total_loss / len(test_loader)
            test_acc = 100.0 * correct / total
            avg_fitness = sum(t.fitness for t in forest.trees) / len(forest.trees)
            num_trees = forest.num_trees()
            arch_diversity = min_arch_diversity(forest)
            memory_size = len(forest.mulch)
            
            # Track diversity changes
            if epoch > 1:
                arch_div_history = metrics_tracker.metrics.get("architecture_diversity", [])
                if arch_div_history:
                    prev_diversity = arch_div_history[-1]
                    if arch_diversity > prev_diversity:
                        last_diversity_increase_epoch = epoch
                    elif epoch - last_diversity_increase_epoch > 50 and epoch - last_diversity_warning_epoch >= 50:
                        # Print warning only once every 50 epochs without increase
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

            print(f"[V7-TRACE] EPOCH {epoch} END | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Trees: {num_trees} | ({time.strftime('%H:%M:%S')})")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'forest': forest.state_dict(),
                    'task_head': task_head.state_dict(),
                    'test_acc': test_acc,
                    'num_trees': num_trees,
                }, checkpoints_dir / "best_model.pt")
                print(f"✓ Best model checkpoint saved at epoch {epoch}")

            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                torch.save({
                    'epoch': epoch,
                    'forest': forest.state_dict(),
                    'task_head': task_head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'num_trees': num_trees,
                }, checkpoints_dir / f"model_epoch{epoch}.pt")
                print(f"💾 Checkpoint saved at epoch {epoch}")

            # More frequent and decisive pruning (every 5 epochs instead of 10)
            if epoch % 5 == 0 and forest.num_trees() > 4:
                weak_trees = [t.id for t in forest.trees if t.age > 20 and t.fitness < 2.5]
                if weak_trees and arch_diversity > 2:
                    # Prune up to 3 weak trees at a time
                    num_to_prune = min(3, len(weak_trees))
                    forest._prune_trees(weak_trees[:num_to_prune], min_keep=3)
                    print(f"🪓 Pruned {num_to_prune} weak trees (arch.div: {arch_diversity})")
            # Plant new trees with greater diversity
            if (epoch % 15 == 0 and forest.num_trees() < args.max_trees) or arch_diversity <= 2:
                arch = TreeArch(
                    num_layers=int(np.random.randint(2, 7)),
                    hidden_dim=int(np.random.choice([256, 512, 768, 1024])),
                    activation=str(np.random.choice(['relu', 'gelu', 'tanh'])),
                    dropout=float(np.random.uniform(0.0, 0.4)),
                    normalization=str(np.random.choice(['none', 'layer', 'batch'])),
                    residual=bool(np.random.choice([True, False]))
                )
                forest._plant_tree(arch)
                print(f"🌱 Planted new tree (total: {forest.num_trees()}, arch.div: {min_arch_diversity(forest)})")
            opt_factory.update_tree_ages(forest)

        print("\n" + "=" * 60)
        print("✅ Training completed — saving results…")
        list_result_files(results_dir, stage="Pre-metrics save")
        start = time.time()
        metrics_tracker.save(metrics_json)
        print(f"✓ metrics.json saved ({metrics_json}) ({int(time.time()-start)}s)")
        start = time.time()
        metrics_tracker.plot(learning_png)
        print(f"✓ learning_curves.png saved ({learning_png}) ({int(time.time()-start)}s)")

        with open(report_md, "w") as f:
            f.write("# NeuralForest CIFAR-10 Training Report\n\n")
            f.write("## Configuration\n\n")
            f.write(f"- **Epochs**: {args.epochs}\n")
            f.write(f"- **Batch size**: {args.batch_size}\n")
            f.write(f"- **Max trees**: {args.max_trees}\n")
            f.write(f"- **Output dim per tree**: {args.output_dim_per_tree}\n")
            f.write(f"- **Device**: {args.device}\n\n")
            f.write("## Final Results\n\n")
            f.write(f"- **Best test accuracy**: {best_test_acc:.2f}%\n")
            f.write(f"- **Final train accuracy**: {train_acc:.2f}%\n")
            f.write(f"- **Final test accuracy**: {test_acc:.2f}%\n")
            f.write(f"- **Final number of trees**: {num_trees}\n")
            f.write(f"- **Task head input dimension**: {num_trees} trees × {args.output_dim_per_tree} = {num_trees * args.output_dim_per_tree}\n")
            f.write(f"- **Architecture diversity**: {arch_diversity} unique types\n")
            f.write(f"- **Memory size**: {memory_size} samples\n")
            f.write(f"- **Average fitness**: {avg_fitness:.2f}\n\n")
            f.write("## Architecture Philosophy\n\n")
            f.write(f"Each tree produces a {args.output_dim_per_tree}-dimensional expert opinion. Task head aggregates these {num_trees * args.output_dim_per_tree} expert opinions.\n\n")
            f.write("## Learning Curves\n\n![Learning Curves](learning_curves.png)\n\n")
            f.write("## Tree Evolution\n\n")
            f.write(f"Started with {initial_trees} trees, ended with {num_trees} trees. Developed {arch_diversity} unique architectural patterns.\n\n---\n*Generated by NeuralForest Training System*\n")
        print(f"✓ final_report.md saved")
        list_result_files(results_dir, stage="AFTER ALL SAVES")
        print(f"✓ Results ready in: {results_dir.resolve()}")
        print("\n🎉 All done!")

    except Exception as e:
        error_msg = f"ERROR: Training failed!\n{traceback.format_exc()}"
        print(error_msg)
        with open(error_log_path, 'w') as f:
            f.write(error_msg)
        with open(report_md, 'w') as f:
            f.write("# Training Failed\n\n")
            f.write(f"## Error\n\n```\n{str(e)}\n```\n\n")
            f.write("See error.log for full traceback.\n")
        if not metrics_json.exists():
            write_dummy_metrics(metrics_json)
            print("✓ dummy metrics.json saved (failure)")
        if not learning_png.exists():
            write_dummy_png(learning_png)
            print("✓ dummy learning_curves.png saved (failure)")
        list_result_files(results_dir, stage="AFTER EXCEPTION")
        raise

if __name__ == "__main__":
    main()
