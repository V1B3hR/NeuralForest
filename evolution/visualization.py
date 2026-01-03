"""
Visualization and Charting System for Forest Evolution.

Provides comprehensive visualization tools including:
- Fitness and diversity charts over time
- Architecture distribution visualization (t-SNE/PCA)
- Heatmaps for tree performance
- Species tree plots
- Generation progression visualizations
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from collections import defaultdict

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. t-SNE/PCA visualizations will be disabled.")


class ForestVisualizer:
    """
    Comprehensive visualization system for forest evolution.
    
    Features:
    - Fitness and diversity trend charts
    - Architecture distribution plots
    - Performance heatmaps
    - Species/genealogy tree visualization
    - Multi-panel dashboards
    """
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize forest visualizer.
        
        Args:
            save_dir: Directory to save generated plots
        """
        self.save_dir = Path(save_dir) if save_dir else Path("./visualizations")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'fitness': '#2E7D32',  # Green
            'diversity': '#1565C0',  # Blue
            'mutations': '#F57C00',  # Orange
            'crossovers': '#6A1B9A',  # Purple
            'births': '#00897B',  # Teal
            'deaths': '#C62828',  # Red
        }
    
    def plot_fitness_trends(
        self,
        history: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot fitness trends over generations.
        
        Args:
            history: List of generation statistics
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if not history:
            print("No history data available for plotting.")
            return
        
        generations = [h.get('generation', i) for i, h in enumerate(history)]
        avg_fitness = [h.get('avg_fitness', 0) for h in history]
        max_fitness = [h.get('max_fitness', 0) for h in history]
        min_fitness = [h.get('min_fitness', 0) for h in history]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot lines
        ax.plot(generations, avg_fitness, 'o-', color=self.colors['fitness'], 
                linewidth=2, markersize=4, label='Average Fitness')
        ax.plot(generations, max_fitness, '^--', color='darkgreen', 
                linewidth=1.5, markersize=4, label='Max Fitness')
        ax.plot(generations, min_fitness, 'v--', color='lightgreen', 
                linewidth=1.5, markersize=4, label='Min Fitness')
        
        # Fill between for range visualization
        ax.fill_between(generations, min_fitness, max_fitness, 
                        alpha=0.2, color=self.colors['fitness'])
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title('Fitness Evolution Over Generations', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Fitness trends plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_diversity_metrics(
        self,
        history: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot diversity metrics over generations.
        
        Args:
            history: List of generation statistics
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if not history:
            print("No history data available for plotting.")
            return
        
        generations = [h.get('generation', i) for i, h in enumerate(history)]
        arch_diversity = [h.get('architecture_diversity', 0) for h in history]
        fitness_diversity = [h.get('fitness_diversity', 0) for h in history]
        num_trees = [h.get('num_trees', 0) for h in history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Diversity metrics
        ax1.plot(generations, arch_diversity, 'o-', color=self.colors['diversity'], 
                linewidth=2, markersize=4, label='Architecture Diversity')
        ax1.plot(generations, fitness_diversity, 's-', color=self.colors['mutations'], 
                linewidth=2, markersize=4, label='Fitness Diversity')
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Diversity Score', fontsize=12)
        ax1.set_title('Diversity Metrics Over Generations', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Population size
        ax2.plot(generations, num_trees, 'o-', color=self.colors['births'], 
                linewidth=2, markersize=4, label='Population Size')
        ax2.fill_between(generations, 0, num_trees, alpha=0.3, color=self.colors['births'])
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Number of Trees', fontsize=12)
        ax2.set_title('Population Size Over Generations', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Diversity metrics plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_evolution_events(
        self,
        history: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot evolutionary events (mutations, crossovers, births, deaths).
        
        Args:
            history: List of generation statistics
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if not history:
            print("No history data available for plotting.")
            return
        
        generations = [h.get('generation', i) for i, h in enumerate(history)]
        mutations = [h.get('mutations_count', 0) for h in history]
        crossovers = [h.get('crossovers_count', 0) for h in history]
        births = [h.get('births_count', 0) for h in history]
        deaths = [h.get('deaths_count', 0) for h in history]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Stacked bar chart
        width = 0.8
        ax.bar(generations, mutations, width, label='Mutations', 
               color=self.colors['mutations'], alpha=0.8)
        ax.bar(generations, crossovers, width, bottom=mutations,
               label='Crossovers', color=self.colors['crossovers'], alpha=0.8)
        ax.bar(generations, births, width, 
               bottom=np.array(mutations) + np.array(crossovers),
               label='Births', color=self.colors['births'], alpha=0.8)
        
        # Deaths on secondary axis
        ax2 = ax.twinx()
        ax2.plot(generations, deaths, 'o-', color=self.colors['deaths'], 
                linewidth=2, markersize=6, label='Deaths')
        
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Evolutionary Events (Stacked)', fontsize=12)
        ax2.set_ylabel('Deaths', fontsize=12, color=self.colors['deaths'])
        ax.set_title('Evolutionary Events Over Generations', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Evolution events plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_architecture_distribution(
        self,
        trees_data: List[Dict[str, Any]],
        method: str = 'pca',
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Visualize architecture distribution using dimensionality reduction.
        
        Args:
            trees_data: List of tree data with architecture information
            method: 'pca' or 'tsne' for dimensionality reduction
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available. Skipping architecture distribution plot.")
            return
        
        if not trees_data:
            print("No tree data available for plotting.")
            return
        
        # Extract architecture features
        features = []
        fitnesses = []
        ages = []
        
        for tree in trees_data:
            arch = tree.get('architecture', {})
            features.append([
                arch.get('num_layers', 3),
                arch.get('hidden_dim', 64),
                arch.get('dropout', 0.1) * 100,  # Scale for better visualization
                hash(arch.get('activation', 'relu')) % 100,  # Encode activation
                1 if arch.get('use_residual', False) else 0,
                hash(arch.get('normalization', 'none')) % 100,  # Encode normalization
            ])
            fitnesses.append(tree.get('fitness', 0))
            ages.append(tree.get('age', 0))
        
        features = np.array(features)
        fitnesses = np.array(fitnesses)
        ages = np.array(ages)
        
        # Apply dimensionality reduction
        if method == 'tsne' and len(features) > 5:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
            reduced = reducer.fit_transform(features)
            title = 't-SNE Architecture Distribution'
        else:
            reducer = PCA(n_components=2, random_state=42)
            reduced = reducer.fit_transform(features)
            title = 'PCA Architecture Distribution'
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Color by fitness
        scatter1 = ax1.scatter(reduced[:, 0], reduced[:, 1], 
                              c=fitnesses, s=100, cmap='viridis', 
                              alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax1.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax1.set_title(f'{title} (colored by fitness)', fontsize=14, fontweight='bold')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Fitness', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Color by age
        scatter2 = ax2.scatter(reduced[:, 0], reduced[:, 1], 
                              c=ages, s=100, cmap='plasma', 
                              alpha=0.7, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax2.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax2.set_title(f'{title} (colored by age)', fontsize=14, fontweight='bold')
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Age (generations)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Architecture distribution plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_performance_heatmap(
        self,
        trees_data: List[Dict[str, Any]],
        metric: str = 'fitness',
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Create a heatmap of tree performance across different metrics.
        
        Args:
            trees_data: List of tree data
            metric: Primary metric to visualize ('fitness', 'age', etc.)
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        if not trees_data:
            print("No tree data available for plotting.")
            return
        
        # Organize data
        tree_ids = [t.get('id', i) for i, t in enumerate(trees_data)]
        metrics_data = {
            'Fitness': [t.get('fitness', 0) for t in trees_data],
            'Age': [t.get('age', 0) for t in trees_data],
            'Bark': [t.get('bark', 1.0) for t in trees_data],
            'Layers': [t.get('architecture', {}).get('num_layers', 3) for t in trees_data],
            'Hidden Dim': [t.get('architecture', {}).get('hidden_dim', 64) for t in trees_data],
            'Dropout': [t.get('architecture', {}).get('dropout', 0.1) * 100 for t in trees_data],
        }
        
        # Create matrix
        metric_names = list(metrics_data.keys())
        data_matrix = np.array([metrics_data[m] for m in metric_names])
        
        # Normalize each row for better visualization
        data_matrix_norm = np.zeros_like(data_matrix, dtype=float)
        for i in range(len(metric_names)):
            row = data_matrix[i]
            if row.max() > row.min():
                data_matrix_norm[i] = (row - row.min()) / (row.max() - row.min())
            else:
                data_matrix_norm[i] = row
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(tree_ids) * 0.5), 8))
        
        im = ax.imshow(data_matrix_norm, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(tree_ids)))
        ax.set_yticks(np.arange(len(metric_names)))
        ax.set_xticklabels([f"Tree {tid}" for tid in tree_ids])
        ax.set_yticklabels(metric_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value', fontsize=11)
        
        # Add values in cells
        for i in range(len(metric_names)):
            for j in range(len(tree_ids)):
                value = data_matrix[i, j]
                text = ax.text(j, i, f"{value:.2f}",
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Tree Performance Heatmap (Normalized)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Performance heatmap saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_species_tree(
        self,
        genealogy_data: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Visualize species/genealogy tree showing ancestry relationships.
        
        Args:
            genealogy_data: Dictionary with genealogy information
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        try:
            import networkx as nx
        except ImportError:
            print("networkx not available. Skipping species tree plot.")
            return
        
        if not genealogy_data:
            print("No genealogy data available for plotting.")
            return
        
        # Build graph from genealogy data
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for tree_id, data in genealogy_data.items():
            G.add_node(tree_id, 
                      fitness=data.get('fitness', 0),
                      generation=data.get('generation', 0),
                      age=data.get('age', 0))
            
            # Add edges from parents
            parents = data.get('parents', [])
            for parent_id in parents:
                G.add_edge(parent_id, tree_id)
        
        if len(G.nodes) == 0:
            print("No nodes in genealogy graph.")
            return
        
        # Create layout
        try:
            pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        except:
            pos = nx.random_layout(G, seed=42)
        
        # Get node attributes for coloring
        fitnesses = [G.nodes[node].get('fitness', 0) for node in G.nodes]
        generations = [G.nodes[node].get('generation', 0) for node in G.nodes]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Draw edges (ancestry connections)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', 
                              arrows=True, arrowsize=15, 
                              arrowstyle='->', alpha=0.5, width=1.5)
        
        # Draw nodes colored by fitness
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax,
                                      node_color=fitnesses,
                                      node_size=500,
                                      cmap='viridis',
                                      vmin=min(fitnesses) if fitnesses else 0,
                                      vmax=max(fitnesses) if fitnesses else 1,
                                      edgecolors='black',
                                      linewidths=1.5)
        
        # Draw labels
        labels = {node: f"{node}\nG{G.nodes[node].get('generation', 0)}" 
                 for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, 
                               font_size=8, font_weight='bold')
        
        # Add colorbar
        if fitnesses:
            cbar = plt.colorbar(nodes, ax=ax)
            cbar.set_label('Fitness', fontsize=12)
        
        ax.set_title('Species/Genealogy Tree\n(Arrows show parent → child relationships)', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Species tree plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_evolution_dashboard(
        self,
        history: List[Dict[str, Any]],
        trees_data: List[Dict[str, Any]],
        genealogy_data: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Create a comprehensive multi-panel dashboard showing all evolution metrics.
        
        Args:
            history: List of generation statistics
            trees_data: Current tree data
            genealogy_data: Optional genealogy information
            save_path: Path to save the dashboard
            show: Whether to display the dashboard
        """
        if not history and not trees_data:
            print("No data available for dashboard.")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Fitness trends (top left)
        if history:
            ax1 = fig.add_subplot(gs[0, 0])
            generations = [h.get('generation', i) for i, h in enumerate(history)]
            avg_fitness = [h.get('avg_fitness', 0) for h in history]
            max_fitness = [h.get('max_fitness', 0) for h in history]
            
            ax1.plot(generations, avg_fitness, 'o-', color=self.colors['fitness'], 
                    linewidth=2, markersize=3, label='Avg')
            ax1.plot(generations, max_fitness, '^--', color='darkgreen', 
                    linewidth=1.5, markersize=3, label='Max')
            ax1.set_xlabel('Generation', fontsize=10)
            ax1.set_ylabel('Fitness', fontsize=10)
            ax1.set_title('Fitness Trends', fontsize=11, fontweight='bold')
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)
        
        # 2. Diversity metrics (top center)
        if history:
            ax2 = fig.add_subplot(gs[0, 1])
            arch_diversity = [h.get('architecture_diversity', 0) for h in history]
            fitness_diversity = [h.get('fitness_diversity', 0) for h in history]
            
            ax2.plot(generations, arch_diversity, 'o-', color=self.colors['diversity'], 
                    linewidth=2, markersize=3, label='Arch')
            ax2.plot(generations, fitness_diversity, 's-', color=self.colors['mutations'], 
                    linewidth=2, markersize=3, label='Fitness')
            ax2.set_xlabel('Generation', fontsize=10)
            ax2.set_ylabel('Diversity', fontsize=10)
            ax2.set_title('Diversity Metrics', fontsize=11, fontweight='bold')
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        # 3. Population size (top right)
        if history:
            ax3 = fig.add_subplot(gs[0, 2])
            num_trees = [h.get('num_trees', 0) for h in history]
            
            ax3.plot(generations, num_trees, 'o-', color=self.colors['births'], 
                    linewidth=2, markersize=3)
            ax3.fill_between(generations, 0, num_trees, alpha=0.3, color=self.colors['births'])
            ax3.set_xlabel('Generation', fontsize=10)
            ax3.set_ylabel('Trees', fontsize=10)
            ax3.set_title('Population Size', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 4. Evolution events (middle left)
        if history:
            ax4 = fig.add_subplot(gs[1, 0])
            mutations = [h.get('mutations_count', 0) for h in history]
            crossovers = [h.get('crossovers_count', 0) for h in history]
            births = [h.get('births_count', 0) for h in history]
            
            width = 0.8
            ax4.bar(generations, mutations, width, label='Mutations', 
                   color=self.colors['mutations'], alpha=0.8)
            ax4.bar(generations, crossovers, width, bottom=mutations,
                   label='Crossovers', color=self.colors['crossovers'], alpha=0.8)
            ax4.set_xlabel('Generation', fontsize=10)
            ax4.set_ylabel('Events', fontsize=10)
            ax4.set_title('Evolution Events', fontsize=11, fontweight='bold')
            ax4.legend(loc='best', fontsize=8)
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Current tree fitness distribution (middle center)
        if trees_data:
            ax5 = fig.add_subplot(gs[1, 1])
            fitnesses = [t.get('fitness', 0) for t in trees_data]
            
            ax5.hist(fitnesses, bins=15, color=self.colors['fitness'], 
                    alpha=0.7, edgecolor='black')
            ax5.axvline(np.mean(fitnesses), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(fitnesses):.2f}')
            ax5.set_xlabel('Fitness', fontsize=10)
            ax5.set_ylabel('Count', fontsize=10)
            ax5.set_title('Current Fitness Distribution', fontsize=11, fontweight='bold')
            ax5.legend(loc='best', fontsize=8)
            ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Architecture parameters (middle right)
        if trees_data:
            ax6 = fig.add_subplot(gs[1, 2])
            layers = [t.get('architecture', {}).get('num_layers', 3) for t in trees_data]
            hidden_dims = [t.get('architecture', {}).get('hidden_dim', 64) for t in trees_data]
            
            ax6.scatter(layers, hidden_dims, s=100, alpha=0.6, 
                       c=fitnesses if trees_data else None,
                       cmap='viridis', edgecolors='black', linewidth=0.5)
            ax6.set_xlabel('Number of Layers', fontsize=10)
            ax6.set_ylabel('Hidden Dimension', fontsize=10)
            ax6.set_title('Architecture Parameters', fontsize=11, fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        # 7. Tree ages (bottom left)
        if trees_data:
            ax7 = fig.add_subplot(gs[2, 0])
            ages = [t.get('age', 0) for t in trees_data]
            tree_ids = [t.get('id', i) for i, t in enumerate(trees_data)]
            
            bars = ax7.bar(range(len(tree_ids)), ages, color=self.colors['births'], 
                          alpha=0.7, edgecolor='black')
            ax7.set_xlabel('Tree ID', fontsize=10)
            ax7.set_ylabel('Age (generations)', fontsize=10)
            ax7.set_title('Tree Ages', fontsize=11, fontweight='bold')
            ax7.set_xticks(range(len(tree_ids)))
            ax7.set_xticklabels([str(tid) for tid in tree_ids], rotation=45, fontsize=8)
            ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Performance summary (bottom center)
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.axis('off')
        
        # Calculate summary statistics
        summary_text = "=== Forest Summary ===\n\n"
        
        if history:
            latest = history[-1]
            summary_text += f"Generation: {latest.get('generation', 0)}\n"
            summary_text += f"Avg Fitness: {latest.get('avg_fitness', 0):.3f}\n"
            summary_text += f"Max Fitness: {latest.get('max_fitness', 0):.3f}\n"
            summary_text += f"Population: {latest.get('num_trees', 0)}\n"
            summary_text += f"Arch Diversity: {latest.get('architecture_diversity', 0):.3f}\n\n"
        
        if trees_data:
            fitnesses = [t.get('fitness', 0) for t in trees_data]
            ages = [t.get('age', 0) for t in trees_data]
            summary_text += f"Trees: {len(trees_data)}\n"
            summary_text += f"Mean Age: {np.mean(ages):.1f}\n"
            summary_text += f"Oldest Tree: {max(ages) if ages else 0}\n"
            summary_text += f"Fitness Std: {np.std(fitnesses):.3f}\n"
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
        
        # 9. Status indicator (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        status_items = []
        if history:
            latest = history[-1]
            
            # Fitness status
            avg_fit = latest.get('avg_fitness', 0)
            if avg_fit > 5:
                status_items.append(('Fitness', '✓ Excellent', 'green'))
            elif avg_fit > 3:
                status_items.append(('Fitness', '✓ Good', 'orange'))
            else:
                status_items.append(('Fitness', '⚠ Low', 'red'))
            
            # Diversity status
            diversity = latest.get('architecture_diversity', 0)
            if diversity > 0.5:
                status_items.append(('Diversity', '✓ Healthy', 'green'))
            elif diversity > 0.2:
                status_items.append(('Diversity', '⚠ Moderate', 'orange'))
            else:
                status_items.append(('Diversity', '⚠ Low', 'red'))
            
            # Population status
            pop = latest.get('num_trees', 0)
            if pop >= 5:
                status_items.append(('Population', '✓ Stable', 'green'))
            elif pop >= 3:
                status_items.append(('Population', '⚠ Small', 'orange'))
            else:
                status_items.append(('Population', '⚠ Critical', 'red'))
        
        y_pos = 0.9
        for label, status, color in status_items:
            ax9.text(0.1, y_pos, f"{label}:", transform=ax9.transAxes,
                    fontsize=10, fontweight='bold')
            ax9.text(0.5, y_pos, status, transform=ax9.transAxes,
                    fontsize=10, color=color, fontweight='bold')
            y_pos -= 0.15
        
        # Main title
        fig.suptitle('NeuralForest Evolution Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Evolution dashboard saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def export_all_plots(
        self,
        history: List[Dict[str, Any]],
        trees_data: List[Dict[str, Any]],
        genealogy_data: Optional[Dict[str, Any]] = None,
        prefix: str = "forest"
    ) -> Dict[str, str]:
        """
        Generate and save all visualization plots.
        
        Args:
            history: List of generation statistics
            trees_data: Current tree data
            genealogy_data: Optional genealogy information
            prefix: Prefix for saved file names
            
        Returns:
            Dictionary mapping plot types to file paths
        """
        saved_files = {}
        
        # Fitness trends
        path = self.save_dir / f"{prefix}_fitness_trends.png"
        self.plot_fitness_trends(history, save_path=str(path), show=False)
        saved_files['fitness_trends'] = str(path)
        
        # Diversity metrics
        path = self.save_dir / f"{prefix}_diversity_metrics.png"
        self.plot_diversity_metrics(history, save_path=str(path), show=False)
        saved_files['diversity_metrics'] = str(path)
        
        # Evolution events
        path = self.save_dir / f"{prefix}_evolution_events.png"
        self.plot_evolution_events(history, save_path=str(path), show=False)
        saved_files['evolution_events'] = str(path)
        
        # Architecture distribution (PCA)
        if SKLEARN_AVAILABLE and trees_data:
            path = self.save_dir / f"{prefix}_architecture_pca.png"
            self.plot_architecture_distribution(trees_data, method='pca', 
                                               save_path=str(path), show=False)
            saved_files['architecture_pca'] = str(path)
            
            # Architecture distribution (t-SNE)
            if len(trees_data) > 5:
                path = self.save_dir / f"{prefix}_architecture_tsne.png"
                self.plot_architecture_distribution(trees_data, method='tsne', 
                                                   save_path=str(path), show=False)
                saved_files['architecture_tsne'] = str(path)
        
        # Performance heatmap
        if trees_data:
            path = self.save_dir / f"{prefix}_performance_heatmap.png"
            self.plot_performance_heatmap(trees_data, save_path=str(path), show=False)
            saved_files['performance_heatmap'] = str(path)
        
        # Species tree
        if genealogy_data:
            path = self.save_dir / f"{prefix}_species_tree.png"
            self.plot_species_tree(genealogy_data, save_path=str(path), show=False)
            saved_files['species_tree'] = str(path)
        
        # Evolution dashboard
        path = self.save_dir / f"{prefix}_dashboard.png"
        self.create_evolution_dashboard(history, trees_data, genealogy_data,
                                       save_path=str(path), show=False)
        saved_files['dashboard'] = str(path)
        
        print(f"\n✓ All plots exported to {self.save_dir}")
        print(f"  Generated {len(saved_files)} visualizations")
        
        return saved_files
