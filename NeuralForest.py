import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
from IPython.display import clear_output  # For smooth animation in Notebooks

# ==========================================
# 1. BIOLOGICAL COMPONENTS (CLASSES)
# ==========================================

class ForestMulch:
    """
    METAPHOR: Forest Mulch (The decaying leaves)
    TECHNICAL: Experience Replay / Memory Buffer
    FUNCTION: Prevents 'Catastrophic Forgetting' by mixing past data with new data.
    """
    def __init__(self, capacity=2000):
        self.memory = deque(maxlen=capacity)
    
    def leaf_drop(self, sun_input, target):
        # Store a single experience (input, target)
        self.memory.append((sun_input, target))
    
    def absorb_nutrients(self, batch_size):
        # Retrieve random past experiences
        if len(self.memory) < batch_size:
            return None, None
        
        batch = random.sample(self.memory, batch_size)
        batch_x = torch.stack([item[0] for item in batch])
        batch_y = torch.stack([item[1] for item in batch])
        return batch_x, batch_y

class Tree(nn.Module):
    """
    METAPHOR: A Living Tree
    TECHNICAL: A Node / Expert (Small MLP Neural Network)
    FUNCTION: Processes local data. As it ages, it develops 'Bark' (lower plasticity).
    """
    def __init__(self, input_dim, hidden_dim, tree_id):
        super(Tree, self).__init__()
        self.id = tree_id
        self.age = 0
        self.bark_thickness = 0.0  # 0.0 = Young (Flexible), 1.0 = Old (Fixed)
        
        # The neural structure (The trunk and crown)
        self.trunk = nn.Linear(input_dim, hidden_dim)
        self.crown = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh() # Tanh often works better for wave-like functions
        
    def photosynthesis(self, x):
        # Forward pass
        sap = self.activation(self.trunk(x))
        fruit = self.crown(sap)
        return fruit
    
    def grow(self):
        # Aging process
        self.age += 1
        # After age 20, bark starts to thicken, protecting knowledge
        if self.age > 20:
            self.bark_thickness = min(0.95, self.bark_thickness + 0.02)

class LivingForest:
    """
    METAPHOR: The Ecosystem
    TECHNICAL: Dynamic Graph Neural Network (Manager)
    FUNCTION: Manages topology, growth, connections, and the lifecycle loop.
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.graph = nx.Graph()          # The root network (Visualization/Topology)
        self.trees = nn.ModuleList()     # The actual models
        self.mulch = ForestMulch()       # Memory
        self.tree_counter = 0
        
        # Plant the first seed
        self.plant_tree()

    def plant_tree(self):
        new_tree = Tree(self.input_dim, 32, self.tree_counter)
        self.trees.append(new_tree)
        self.graph.add_node(self.tree_counter, color='lime') # Lime = Young
        
        # Mycorrhizal Connection: Find the most similar existing tree to connect to
        if len(self.trees) > 1:
            similar_tree = self.find_similar_tree(new_tree)
            # Add a "binding root" edge
            self.graph.add_edge(new_tree.id, similar_tree.id, weight=2.0, type='binding')
        
        self.tree_counter += 1

    def find_similar_tree(self, new_tree):
        """
        Calculates weight similarity between the new tree and existing ones.
        Connects trees that 'think' alike (Clustering).
        """
        best_match = self.trees[0]
        min_dist = float('inf')
        
        for other_tree in self.trees:
            if other_tree.id == new_tree.id: continue
            
            # Euclidean distance between parameters (simplified)
            dist = 0
            for p1, p2 in zip(new_tree.parameters(), other_tree.parameters()):
                dist += (p1 - p2).norm().item()
            
            if dist < min_dist:
                min_dist = dist
                best_match = other_tree
                
        return best_match

    def life_cycle(self, current_sun, current_target, optimizer):
        # --- A. Leaf Drop (Memory Storage) ---
        for i in range(len(current_sun)):
            self.mulch.leaf_drop(current_sun[i], current_target[i])

        # --- B. Nutrient Absorption (Experience Replay) ---
        # Mix fresh data (Sun) with old data (Mulch)
        old_x, old_y = self.mulch.absorb_nutrients(len(current_sun))
        
        if old_x is not None:
            train_x = torch.cat([current_sun, old_x])
            train_y = torch.cat([current_target, old_y])
        else:
            train_x, train_y = current_sun, current_target

        # --- C. Photosynthesis (Forward Pass) ---
        optimizer.zero_grad()
        forest_output = torch.zeros_like(train_y)
        
        # Ensemble prediction (Average of all trees)
        for tree in self.trees:
            forest_output += tree.photosynthesis(train_x)
        
        forest_output /= len(self.trees)
        
        # --- D. Stress Response (Loss Calculation) ---
        loss = nn.MSELoss()(forest_output, train_y)
        loss.backward()

        # --- E. Bark Protection (Gradient Masking) ---
        # Older trees with thick bark resist change (lower plasticity)
        for tree in self.trees:
            if tree.bark_thickness > 0:
                for param in tree.parameters():
                    if param.grad is not None:
                        # Scale gradient: Thick bark = tiny changes
                        param.grad *= (1.0 - tree.bark_thickness)
            tree.grow() # Increase age

        optimizer.step()

        # --- F. Growth (Neurogenesis) ---
        # If stress (loss) is high, plant a new tree
        if loss.item() > 0.02 and len(self.trees) < 15:
            # Only grow if a random chance is met (simulating nature's randomness)
            if random.random() < 0.15:
                self.plant_tree()
        
        return loss.item(), train_x, forest_output

    def visualize(self, epoch, loss):
        clear_output(wait=True) # Clear previous frame
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: The Network Structure (Roots)
        plt.subplot(1, 2, 1)
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Color nodes by Age (Green = Young, Dark Green = Old)
        colors = []
        sizes = []
        for tree in self.trees:
            darkness = min(1.0, tree.bark_thickness)
            colors.append((0, 1.0 - (darkness * 0.7), 0)) 
            sizes.append(300 + tree.age * 10)
            
        nx.draw(self.graph, pos, node_color=colors, node_size=sizes, with_labels=True, font_color='white')
        plt.title(f"Root System (GNN)\nTrees: {len(self.trees)} | Epoch: {epoch}")

        # Subplot 2: The Knowledge (Function Approximation)
        plt.subplot(1, 2, 2)
        
        # Generate a full view of what the forest currently "knows"
        test_x = torch.linspace(0, 10, 200).reshape(-1, 1)
        with torch.no_grad():
            full_pred = torch.zeros_like(test_x)
            for tree in self.trees:
                full_pred += tree.photosynthesis(test_x)
            full_pred /= len(self.trees)
            
        plt.plot(test_x.numpy(), torch.sin(test_x).numpy(), 'k--', alpha=0.5, label='True Nature (Sin)')
        plt.plot(test_x.numpy(), full_pred.numpy(), 'g-', linewidth=2, label='Forest Knowledge')
        plt.title(f"Photosynthesis Result\nStress (Loss): {loss:.4f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ==========================================
# 2. RUNNING THE SIMULATION
# ==========================================

# Data: A Sine wave (The Sun)
X = torch.linspace(0, 10, 200).reshape(-1, 1)
Y = torch.sin(X)

# Initialize Ecosystem
forest = LivingForest(input_dim=1)
optimizer = optim.Adam(forest.trees.parameters(), lr=0.05)

print("🌱 The forest is beginning to grow...")

# Time Loop
for epoch in range(150):
    
    # Simulate Seasons: The forest only sees a PART of the data at a time.
    # Without Mulch (Memory), it would forget the beginning when seeing the end.
    window_start = (epoch * 5) % 150
    window_end = window_start + 50
    
    sun_batch = X[window_start:window_end]
    target_batch = Y[window_start:window_end]
    
    # Execute Life Cycle
    loss, _, _ = forest.life_cycle(sun_batch, target_batch, optimizer)
    
    # Update Optimizer if a new tree was planted (parameters changed)
    if len(list(forest.trees.parameters())) > len(optimizer.param_groups[0]['params']):
        optimizer = optim.Adam(forest.trees.parameters(), lr=0.05)

    # Visualize every 2 epochs
    if epoch % 2 == 0:
        forest.visualize(epoch, loss)

print("✅ Simulation Complete.")