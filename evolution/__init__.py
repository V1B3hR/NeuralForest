"""
Evolution module for NeuralForest Phase 3 & Phase 4: Evolution & Self-Improvement

This module implements neural architecture search, self-improvement systems,
tree legacy management, genealogy tracking, seasonal evolution integration,
real-time monitoring, and AutoML capabilities that enable the forest to
discover optimal tree architectures, continuously improve its structure and
performance, and maintain a memory of eliminated trees for evolutionary insights.
"""

from .architecture_search import TreeArchitectureSearch
from .self_improvement import SelfImprovementLoop
from .tree_graveyard import TreeGraveyard, TreeRecord, GraveyardStats
from .genealogy import GenealogyTracker, TreeLineage
from .season_integration import SeasonalEvolution, integrate_season_with_nas
from .monitoring import EvolutionMonitor, EvolutionSnapshot, MonitoringDashboard
from .automl import (
    AutoMLOrchestrator,
    ContinuousGeneralizationTester,
    RegressionValidator,
    MetricAlerter,
    TestResult,
    BenchmarkResult,
)

__all__ = [
    "TreeArchitectureSearch",
    "SelfImprovementLoop",
    "TreeGraveyard",
    "TreeRecord",
    "GraveyardStats",
    "GenealogyTracker",
    "TreeLineage",
    "SeasonalEvolution",
    "integrate_season_with_nas",
    "EvolutionMonitor",
    "EvolutionSnapshot",
    "MonitoringDashboard",
    "AutoMLOrchestrator",
    "ContinuousGeneralizationTester",
    "RegressionValidator",
    "MetricAlerter",
    "TestResult",
    "BenchmarkResult",
]
