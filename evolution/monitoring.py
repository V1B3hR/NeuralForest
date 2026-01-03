"""
Real-time Monitoring System for Evolutionary Progress.

Provides live tracking and visualization of forest evolution,
including fitness trends, diversity metrics, and population dynamics.
"""

from __future__ import annotations

import time
import json
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvolutionSnapshot:
    """A snapshot of evolutionary state at a point in time."""
    
    timestamp: float
    generation: int
    step: int
    
    # Population metrics
    num_trees: int
    alive_trees: int
    dead_trees: int
    
    # Fitness metrics
    avg_fitness: float
    max_fitness: float
    min_fitness: float
    fitness_std: float
    
    # Diversity metrics
    architecture_diversity: float
    fitness_diversity: float
    
    # Evolution metrics
    mutations_count: int = 0
    crossovers_count: int = 0
    births_count: int = 0
    deaths_count: int = 0
    
    # Season (if applicable)
    season: Optional[str] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class EvolutionMonitor:
    """
    Real-time monitoring of evolutionary progress.
    
    Features:
    - Live metrics tracking
    - Trend analysis
    - Alert system
    - Export capabilities
    - CLI display
    """
    
    def __init__(
        self,
        window_size: int = 100,
        alert_thresholds: Optional[Dict[str, float]] = None,
        save_dir: Optional[Path] = None
    ):
        """
        Initialize evolution monitor.
        
        Args:
            window_size: Number of snapshots to keep in memory
            alert_thresholds: Thresholds for triggering alerts
            save_dir: Directory to save monitoring data
        """
        self.window_size = window_size
        self.snapshots = deque(maxlen=window_size)
        self.alerts = []
        self.save_dir = Path(save_dir) if save_dir else None
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "min_diversity": 0.1,  # Alert if diversity drops below
            "max_fitness_stagnation": 10,  # Alert if no improvement for N snapshots
            "min_alive_trees": 2,  # Alert if population too small
            "fitness_drop_rate": 0.3,  # Alert if fitness drops > 30%
        }
        
        # Tracking
        self.start_time = time.time()
        self.last_max_fitness = 0.0
        self.stagnation_counter = 0
        self.total_snapshots = 0
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, Dict], None]] = []
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def record_snapshot(
        self,
        generation: int,
        step: int,
        forest_state: Dict[str, Any],
        evolution_state: Optional[Dict[str, Any]] = None,
        season: Optional[str] = None
    ):
        """
        Record a snapshot of the current evolutionary state.
        
        Args:
            generation: Current generation number
            step: Current training step
            forest_state: State of the forest (trees, fitness, etc.)
            evolution_state: Evolution statistics (mutations, crossovers, etc.)
            season: Current season if applicable
        """
        evolution_state = evolution_state or {}
        
        snapshot = EvolutionSnapshot(
            timestamp=time.time(),
            generation=generation,
            step=step,
            num_trees=forest_state.get("num_trees", 0),
            alive_trees=forest_state.get("alive_trees", 0),
            dead_trees=forest_state.get("dead_trees", 0),
            avg_fitness=forest_state.get("avg_fitness", 0.0),
            max_fitness=forest_state.get("max_fitness", 0.0),
            min_fitness=forest_state.get("min_fitness", 0.0),
            fitness_std=forest_state.get("fitness_std", 0.0),
            architecture_diversity=forest_state.get("architecture_diversity", 0.0),
            fitness_diversity=forest_state.get("fitness_diversity", 0.0),
            mutations_count=evolution_state.get("mutations", 0),
            crossovers_count=evolution_state.get("crossovers", 0),
            births_count=evolution_state.get("births", 0),
            deaths_count=evolution_state.get("deaths", 0),
            season=season,
            custom_metrics=forest_state.get("custom_metrics", {}),
        )
        
        self.snapshots.append(snapshot)
        self.total_snapshots += 1
        
        # Check for alerts
        self._check_alerts(snapshot)
        
        # Auto-save periodically
        if self.save_dir and self.total_snapshots % 100 == 0:
            self.save_snapshots()
    
    def _check_alerts(self, snapshot: EvolutionSnapshot):
        """Check if any alert conditions are triggered."""
        alerts_triggered = []
        
        # Diversity alert
        if snapshot.architecture_diversity < self.alert_thresholds["min_diversity"]:
            alerts_triggered.append({
                "type": "low_diversity",
                "message": f"⚠️ Low diversity: {snapshot.architecture_diversity:.3f}",
                "severity": "warning",
                "generation": snapshot.generation,
                "step": snapshot.step,
            })
        
        # Population alert
        if snapshot.alive_trees < self.alert_thresholds["min_alive_trees"]:
            alerts_triggered.append({
                "type": "low_population",
                "message": f"⚠️ Low population: {snapshot.alive_trees} trees alive",
                "severity": "critical",
                "generation": snapshot.generation,
                "step": snapshot.step,
            })
        
        # Fitness stagnation
        if snapshot.max_fitness <= self.last_max_fitness:
            self.stagnation_counter += 1
        else:
            self.last_max_fitness = snapshot.max_fitness
            self.stagnation_counter = 0
        
        if self.stagnation_counter >= self.alert_thresholds["max_fitness_stagnation"]:
            alerts_triggered.append({
                "type": "fitness_stagnation",
                "message": f"⚠️ Fitness stagnant for {self.stagnation_counter} snapshots",
                "severity": "warning",
                "generation": snapshot.generation,
                "step": snapshot.step,
            })
        
        # Fitness drop
        if len(self.snapshots) >= 2:
            prev_snapshot = list(self.snapshots)[-2]
            if prev_snapshot.avg_fitness > 0:
                fitness_change = (snapshot.avg_fitness - prev_snapshot.avg_fitness) / prev_snapshot.avg_fitness
                if fitness_change < -self.alert_thresholds["fitness_drop_rate"]:
                    alerts_triggered.append({
                        "type": "fitness_drop",
                        "message": f"⚠️ Fitness dropped by {abs(fitness_change)*100:.1f}%",
                        "severity": "warning",
                        "generation": snapshot.generation,
                        "step": snapshot.step,
                    })
        
        # Record and trigger callbacks
        for alert in alerts_triggered:
            self.alerts.append(alert)
            logger.warning(alert["message"])
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert["type"], alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def register_alert_callback(self, callback: Callable[[str, Dict], None]):
        """
        Register a callback function for alerts.
        
        Args:
            callback: Function that takes (alert_type, alert_dict)
        """
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics."""
        if not self.snapshots:
            return {}
        
        latest = self.snapshots[-1]
        return latest.to_dict()
    
    def get_trend(self, metric: str, window: Optional[int] = None) -> List[float]:
        """
        Get trend for a specific metric.
        
        Args:
            metric: Metric name
            window: Number of recent snapshots (None for all)
            
        Returns:
            List of metric values
        """
        snapshots_to_use = list(self.snapshots)
        if window:
            snapshots_to_use = snapshots_to_use[-window:]
        
        values = []
        for snapshot in snapshots_to_use:
            snapshot_dict = snapshot.to_dict()
            if metric in snapshot_dict:
                values.append(snapshot_dict[metric])
            elif metric in snapshot.custom_metrics:
                values.append(snapshot.custom_metrics[metric])
        
        return values
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall monitoring statistics."""
        if not self.snapshots:
            return {
                "total_snapshots": 0,
                "runtime_seconds": 0,
                "alerts_count": len(self.alerts),
            }
        
        runtime = time.time() - self.start_time
        latest = self.snapshots[-1]
        oldest = self.snapshots[0]
        
        # Calculate rates
        generation_span = latest.generation - oldest.generation
        time_span = latest.timestamp - oldest.timestamp
        
        generations_per_second = generation_span / time_span if time_span > 0 else 0
        
        # Fitness improvement
        fitness_improvement = latest.max_fitness - oldest.max_fitness
        
        # Alert breakdown
        alert_counts = defaultdict(int)
        for alert in self.alerts:
            alert_counts[alert["type"]] += 1
        
        return {
            "total_snapshots": self.total_snapshots,
            "runtime_seconds": runtime,
            "current_generation": latest.generation,
            "current_step": latest.step,
            "generations_per_second": generations_per_second,
            "fitness_improvement": fitness_improvement,
            "current_max_fitness": latest.max_fitness,
            "current_diversity": latest.architecture_diversity,
            "alive_trees": latest.alive_trees,
            "alerts_count": len(self.alerts),
            "alerts_by_type": dict(alert_counts),
            "stagnation_counter": self.stagnation_counter,
        }
    
    def print_status(self, detailed: bool = False):
        """
        Print current monitoring status to console.
        
        Args:
            detailed: Whether to print detailed metrics
        """
        if not self.snapshots:
            print("📊 Evolution Monitor: No data yet")
            return
        
        latest = self.snapshots[-1]
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("📊 EVOLUTION MONITOR - LIVE STATUS")
        print("="*60)
        
        # Basic info
        print(f"\n⏱️  Runtime: {stats['runtime_seconds']:.1f}s")
        print(f"🧬 Generation: {latest.generation} | Step: {latest.step}")
        if latest.season:
            print(f"🌍 Season: {latest.season}")
        
        # Population
        print(f"\n👥 Population: {latest.alive_trees} alive / {latest.num_trees} total")
        
        # Fitness
        print(f"\n💪 Fitness:")
        print(f"   Max: {latest.max_fitness:.4f}")
        print(f"   Avg: {latest.avg_fitness:.4f}")
        print(f"   Min: {latest.min_fitness:.4f}")
        print(f"   Std: {latest.fitness_std:.4f}")
        
        # Diversity
        print(f"\n🌈 Diversity:")
        print(f"   Architecture: {latest.architecture_diversity:.4f}")
        print(f"   Fitness: {latest.fitness_diversity:.4f}")
        
        # Evolution activity
        if latest.mutations_count or latest.crossovers_count:
            print(f"\n🔬 Evolution:")
            print(f"   Mutations: {latest.mutations_count}")
            print(f"   Crossovers: {latest.crossovers_count}")
            print(f"   Births: {latest.births_count}")
            print(f"   Deaths: {latest.deaths_count}")
        
        # Alerts
        if self.alerts:
            recent_alerts = self.alerts[-3:]
            print(f"\n⚠️  Recent Alerts ({len(self.alerts)} total):")
            for alert in recent_alerts:
                print(f"   {alert['message']}")
        
        # Trends (if enough data)
        if len(self.snapshots) >= 10 and detailed:
            print(f"\n📈 Trends (last 10 snapshots):")
            
            fitness_trend = self.get_trend("avg_fitness", 10)
            if len(fitness_trend) >= 2:
                trend_direction = "↗️" if fitness_trend[-1] > fitness_trend[0] else "↘️"
                print(f"   Fitness: {trend_direction} {fitness_trend[0]:.3f} → {fitness_trend[-1]:.3f}")
            
            diversity_trend = self.get_trend("architecture_diversity", 10)
            if len(diversity_trend) >= 2:
                trend_direction = "↗️" if diversity_trend[-1] > diversity_trend[0] else "↘️"
                print(f"   Diversity: {trend_direction} {diversity_trend[0]:.3f} → {diversity_trend[-1]:.3f}")
        
        print("\n" + "="*60 + "\n")
    
    def export_metrics(self, output_path: Path):
        """
        Export all metrics to a file.
        
        Args:
            output_path: Path to save metrics
        """
        data = {
            "statistics": self.get_statistics(),
            "snapshots": [s.to_dict() for s in self.snapshots],
            "alerts": self.alerts,
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to {output_path}")
    
    def save_snapshots(self, path: Optional[Path] = None):
        """
        Save snapshots to disk.
        
        Args:
            path: Path to save file
        """
        save_path = path or (self.save_dir / f"snapshots_{int(time.time())}.json" if self.save_dir else None)
        
        if not save_path:
            logger.warning("No save path specified")
            return
        
        data = {
            "snapshots": [s.to_dict() for s in self.snapshots],
            "start_time": self.start_time,
            "total_snapshots": self.total_snapshots,
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Snapshots saved to {save_path}")
    
    def clear_alerts(self):
        """Clear all recorded alerts."""
        self.alerts.clear()
        logger.info("Alerts cleared")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get formatted data for dashboard display.
        
        Returns:
            Dictionary with dashboard-ready data
        """
        stats = self.get_statistics()
        
        # Get trends for visualization
        fitness_trend = self.get_trend("avg_fitness", 50)
        max_fitness_trend = self.get_trend("max_fitness", 50)
        diversity_trend = self.get_trend("architecture_diversity", 50)
        population_trend = self.get_trend("alive_trees", 50)
        
        return {
            "summary": {
                "generation": stats.get("current_generation", 0),
                "step": stats.get("current_step", 0),
                "runtime": stats.get("runtime_seconds", 0),
                "max_fitness": stats.get("current_max_fitness", 0),
                "diversity": stats.get("current_diversity", 0),
                "alive_trees": stats.get("alive_trees", 0),
            },
            "trends": {
                "fitness": fitness_trend,
                "max_fitness": max_fitness_trend,
                "diversity": diversity_trend,
                "population": population_trend,
            },
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "statistics": stats,
        }


class MonitoringDashboard:
    """
    Simple CLI dashboard for live monitoring.
    """
    
    def __init__(self, monitor: EvolutionMonitor, refresh_interval: float = 1.0):
        """
        Initialize dashboard.
        
        Args:
            monitor: EvolutionMonitor instance
            refresh_interval: Seconds between refreshes
        """
        self.monitor = monitor
        self.refresh_interval = refresh_interval
        self.running = False
    
    def start(self):
        """Start the dashboard (blocking)."""
        import sys
        
        self.running = True
        print("\n🌲 NeuralForest Evolution Monitor 🌲")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                # Clear screen (simple approach)
                sys.stdout.write("\033[2J\033[H")
                sys.stdout.flush()
                
                # Print status
                self.monitor.print_status(detailed=True)
                
                # Wait
                time.sleep(self.refresh_interval)
        
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")
            self.running = False
    
    def stop(self):
        """Stop the dashboard."""
        self.running = False
