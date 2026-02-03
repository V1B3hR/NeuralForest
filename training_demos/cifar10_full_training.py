import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_seed(seed=42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

class SimpleCIFARConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # [B, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class MetricsTracker:
    def __init__(self):
        self.metrics = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
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
        plt.title("CIFAR-10 Accuracy")
        plt.savefig(path)
        plt.close()

def train(model, device, train_loader, optimizer, sched, criterion, epoch, tracker):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    sched.step()
    acc = 100. * correct / total
    tracker.update(epoch, {"train_loss": train_loss / len(train_loader), "train_accuracy": acc})
    return acc

def test(model, device, test_loader, criterion, epoch, tracker):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    acc = 100. * correct / total
    tracker.metrics["test_loss"].append(test_loss / len(test_loader))
    tracker.metrics["test_accuracy"].append(acc)
    return acc

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--output_dir', type=str, default='training_demos/results/cifar10_full')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    model = SimpleCIFARConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=20, gamma=0.2)
    tracker = MetricsTracker()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        acc_train = train(model, device, train_loader, optimizer, scheduler, criterion, epoch, tracker)
        acc_test = test(model, device, test_loader, criterion, epoch, tracker)
        if acc_test > best_acc:
            best_acc = acc_test
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pt'))
        print(f'[Epoch {epoch}] Train acc: {acc_train:.2f}%  Test acc: {acc_test:.2f}% (Best: {best_acc:.2f}%)')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(results_dir, f'model_epoch{epoch}.pt'))

    tracker.save(os.path.join(results_dir, "metrics.json"))
    tracker.plot(os.path.join(results_dir, "learning_curves.png"))
    with open(os.path.join(results_dir, "final_report.md"), "w") as f:
        f.write(f"# NeuralForest CIFAR-10 Training Report\n\n")
        f.write(f"**Best test accuracy**: {best_acc:.2f}%\n\n")
    print(f"\nDone! Best test accuracy: {best_acc:.2f}% . Results in {results_dir}")

if __name__ == "__main__":
    main()
