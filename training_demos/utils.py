"""
Utilities for NeuralForest training demonstrations.
Includes dataset loaders, metrics tracking, and visualization helpers.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

class DatasetLoader:
    """Load and prepare datasets for training."""
    
    @staticmethod
    def get_cifar10(batch_size=128, num_workers=4):
        """Load CIFAR-10 dataset."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, test_loader
    
    @staticmethod
    def get_mnist(batch_size=128, num_workers=4):
        """Load MNIST dataset."""
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to match CIFAR
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, test_loader
    
    @staticmethod
    def get_fashion_mnist(batch_size=128, num_workers=4):
        """Load Fashion-MNIST dataset."""
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, test_loader

class MetricsTracker:
    """Track and log training metrics."""
    
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
        """Update metrics for current epoch."""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save(self, path):
        """Save metrics to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot(self, save_path):
        """Generate comprehensive plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['epoch'], self.history['test_loss'], label='Test')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history['epoch'], self.history['train_accuracy'], label='Train')
        axes[0, 1].plot(self.history['epoch'], self.history['test_accuracy'], label='Test')
        axes[0, 1].set_title('Accuracy Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Fitness
        axes[0, 2].plot(self.history['epoch'], self.history['avg_fitness'])
        axes[0, 2].set_title('Average Tree Fitness')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Fitness')
        axes[0, 2].grid(True)
        
        # Trees Count
        axes[1, 0].plot(self.history['epoch'], self.history['num_trees'])
        axes[1, 0].set_title('Number of Trees')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True)
        
        # Architecture Diversity
        axes[1, 1].plot(self.history['epoch'], self.history['architecture_diversity'])
        axes[1, 1].set_title('Architecture Diversity')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Unique Architectures')
        axes[1, 1].grid(True)
        
        # Memory Usage
        axes[1, 2].plot(self.history['epoch'], self.history['memory_size'])
        axes[1, 2].set_title('Memory Usage')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Samples Stored')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
