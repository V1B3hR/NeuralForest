"""
Enhanced Task Head for NeuralForest.

Implements task head with refinement layer:
- Input (128) → Linear(128, 64) → LayerNorm → ReLU → Dropout → Linear(64, 10)
- Optional skip connection (128→10 direct)
- Multiple activation options
- Proper weight initialization
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


class EnhancedTaskHead(nn.Module):
    """
    Enhanced task head with refinement layer.
    
    Architecture:
        Input (input_dim)
          ↓
        Linear(input_dim, hidden_dim)
        LayerNorm(hidden_dim)
        Activation
        Dropout
          ↓
        Linear(hidden_dim, num_classes)
          ↓
        Output logits (num_classes)
    
    Features:
    - Layer normalization for training stability
    - Configurable dropout
    - Optional skip connection
    - Multiple activation functions
    - Kaiming initialization
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 10,
        dropout: float = 0.2,
        activation: Literal['relu', 'gelu', 'leaky_relu'] = 'relu',
        use_skip: bool = False,
        task_name: str = "enhanced_classification"
    ):
        """
        Initialize enhanced task head.
        
        Args:
            input_dim: Input feature dimension (default: 128)
            hidden_dim: Hidden layer dimension (default: 64)
            num_classes: Number of output classes (default: 10)
            dropout: Dropout probability (default: 0.2)
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
            use_skip: Whether to use skip connection (default: False)
            task_name: Task identifier
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_skip = use_skip
        self.task_name = task_name
        
        # First layer: input_dim → hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layer: hidden_dim → num_classes
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Optional skip connection
        if use_skip:
            self.skip = nn.Linear(input_dim, num_classes, bias=False)
        else:
            self.skip = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        # Kaiming initialization for layers with ReLU-like activations
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        
        if self.skip is not None:
            nn.init.kaiming_normal_(self.skip.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through task head.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Class logits [batch_size, num_classes]
        """
        # Store input for skip connection
        identity = x
        
        # Main path
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        
        # Add skip connection if enabled
        if self.skip is not None:
            x = x + self.skip(identity)
        
        return x
    
    def get_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Cross-entropy loss
        """
        return nn.functional.cross_entropy(predictions, targets)
    
    def predict(
        self, 
        x: torch.Tensor, 
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        Make predictions on input.
        
        Args:
            x: Input features [batch_size, input_dim]
            return_probs: If True, return probabilities instead of class indices
            
        Returns:
            Predicted classes or probabilities
        """
        logits = self.forward(x)
        
        if return_probs:
            return torch.softmax(logits, dim=-1)
        return torch.argmax(logits, dim=-1)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate features (before final classification).
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Hidden features [batch_size, hidden_dim]
        """
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.activation(x)
        return x


class MultiHeadTaskHead(nn.Module):
    """
    Multi-task variant with multiple output heads.
    
    Useful for multi-task learning scenarios.
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_classes_list: list = [10, 10],
        dropout: float = 0.2,
        activation: str = 'relu',
        task_names: Optional[list] = None
    ):
        """
        Initialize multi-head task head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Shared hidden dimension
            num_classes_list: List of number of classes for each task
            dropout: Dropout probability
            activation: Activation function name
            task_names: Optional list of task names
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = len(num_classes_list)
        self.num_classes_list = num_classes_list
        
        if task_names is None:
            task_names = [f"task_{i}" for i in range(self.num_tasks)]
        self.task_names = task_names
        
        # Shared feature extractor
        self.shared_fc = nn.Linear(input_dim, hidden_dim)
        self.shared_norm = nn.LayerNorm(hidden_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for num_classes in num_classes_list
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_normal_(self.shared_fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.shared_fc.bias)
        
        for head in self.task_heads:
            nn.init.kaiming_normal_(head.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(head.bias)
    
    def forward(self, x: torch.Tensor, task_idx: int = 0) -> torch.Tensor:
        """
        Forward pass through specific task head.
        
        Args:
            x: Input features [batch_size, input_dim]
            task_idx: Which task head to use (default: 0)
            
        Returns:
            Class logits for specified task
        """
        # Shared features
        x = self.shared_fc(x)
        x = self.shared_norm(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        
        # Task-specific output
        x = self.task_heads[task_idx](x)
        
        return x
    
    def forward_all(self, x: torch.Tensor) -> list:
        """
        Forward pass through all task heads.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            List of logits for each task
        """
        # Shared features
        h = self.shared_fc(x)
        h = self.shared_norm(h)
        h = self.activation(h)
        h = self.dropout_layer(h)
        
        # All task outputs
        outputs = [head(h) for head in self.task_heads]
        
        return outputs
    
    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        task_idx: int = 0
    ) -> torch.Tensor:
        """Compute loss for specific task."""
        return nn.functional.cross_entropy(predictions, targets)


if __name__ == "__main__":
    """Example usage and testing."""
    print("Testing Enhanced Task Head")
    print("=" * 70)
    
    # Test single-task head
    print("\n1. Single-Task Head (128→64→10)")
    head = EnhancedTaskHead(
        input_dim=128,
        hidden_dim=64,
        num_classes=10,
        dropout=0.2,
        activation='relu',
        use_skip=False
    )
    
    print(f"   Input dim: 128")
    print(f"   Hidden dim: 64")
    print(f"   Output classes: 10")
    print(f"   Parameters: {sum(p.numel() for p in head.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 128)
    logits = head(x)
    print(f"\n   Input shape: {x.shape}")
    print(f"   Output shape: {logits.shape}")
    assert logits.shape == (batch_size, 10), "Output shape mismatch!"
    print("   ✅ Shape test passed")
    
    # Test loss computation
    targets = torch.randint(0, 10, (batch_size,))
    loss = head.get_loss(logits, targets)
    print(f"\n   Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive!"
    print("   ✅ Loss computation passed")
    
    # Test prediction
    preds = head.predict(x)
    probs = head.predict(x, return_probs=True)
    print(f"\n   Predictions shape: {preds.shape}")
    print(f"   Probabilities shape: {probs.shape}")
    assert preds.shape == (batch_size,), "Predictions shape mismatch!"
    assert probs.shape == (batch_size, 10), "Probabilities shape mismatch!"
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size)), "Probs should sum to 1!"
    print("   ✅ Prediction test passed")
    
    # Test feature extraction
    features = head.get_features(x)
    print(f"\n   Features shape: {features.shape}")
    assert features.shape == (batch_size, 64), "Features shape mismatch!"
    print("   ✅ Feature extraction passed")
    
    # Test with skip connection
    print("\n2. Single-Task Head with Skip Connection")
    head_skip = EnhancedTaskHead(
        input_dim=128,
        hidden_dim=64,
        num_classes=10,
        use_skip=True
    )
    
    logits_skip = head_skip(x)
    print(f"   Output shape: {logits_skip.shape}")
    assert logits_skip.shape == (batch_size, 10), "Output shape mismatch!"
    print("   ✅ Skip connection test passed")
    
    # Test multi-head variant
    print("\n3. Multi-Task Head")
    multi_head = MultiHeadTaskHead(
        input_dim=128,
        hidden_dim=64,
        num_classes_list=[10, 5, 3],
        task_names=['cifar10', 'task2', 'task3']
    )
    
    print(f"   Tasks: {multi_head.num_tasks}")
    print(f"   Parameters: {sum(p.numel() for p in multi_head.parameters()):,}")
    
    # Test single task forward
    logits_task0 = multi_head(x, task_idx=0)
    print(f"\n   Task 0 output shape: {logits_task0.shape}")
    assert logits_task0.shape == (batch_size, 10), "Task 0 shape mismatch!"
    
    # Test all tasks forward
    all_logits = multi_head.forward_all(x)
    print(f"   All tasks outputs: {len(all_logits)} tensors")
    for i, logits in enumerate(all_logits):
        expected_classes = multi_head.num_classes_list[i]
        print(f"   Task {i} shape: {logits.shape}")
        assert logits.shape == (batch_size, expected_classes), f"Task {i} shape mismatch!"
    print("   ✅ Multi-task test passed")
    
    # Test different activations
    print("\n4. Different Activations")
    for act in ['relu', 'gelu', 'leaky_relu']:
        head_act = EnhancedTaskHead(
            input_dim=128,
            hidden_dim=64,
            num_classes=10,
            activation=act
        )
        logits_act = head_act(x)
        print(f"   {act:12s}: output shape {logits_act.shape}")
        assert logits_act.shape == (batch_size, 10), f"{act} output shape mismatch!"
    print("   ✅ Activation test passed")
    
    # Test gradient flow
    print("\n5. Gradient Flow Test")
    head.train()
    x_grad = torch.randn(batch_size, 128, requires_grad=True)
    logits_grad = head(x_grad)
    loss_grad = head.get_loss(logits_grad, targets)
    loss_grad.backward()
    
    assert x_grad.grad is not None, "Gradients should flow to input!"
    assert all(p.grad is not None for p in head.parameters() if p.requires_grad), \
        "All parameters should have gradients!"
    print("   ✅ Gradient flow test passed")
    
    print("\n" + "=" * 70)
    print("✅ All Enhanced Task Head tests passed!")
