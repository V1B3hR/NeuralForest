"""
Evolution module for NeuralForest Phase 3, 4, 5, & 6: Evolution, Self-Improvement, Visualization & Cooperation

This module implements neural architecture search, self-improvement systems,
tree legacy management, genealogy tracking, seasonal evolution integration,
real-time monitoring, AutoML capabilities, comprehensive visualization tools,
tree cooperation mechanisms, and environmental simulation that enable the forest
to discover optimal tree architectures, continuously improve its structure and
performance, maintain a memory of eliminated trees for evolutionary insights,
visualize the evolutionary process, cooperate across trees, and adapt to
dynamic environmental conditions.
"""

from .architecture_search import TreeArchitectureSearch
from .self_improvement import SelfImprovementLoop
from .tree_graveyard import TreeGraveyard, TreeRecord, GraveyardStats
from .genealogy import GenealogyTracker, TreeLineage
from .season_integration import SeasonalEvolution, integrate_season_with_nas
from .monitoring import EvolutionMonitor, EvolutionSnapshot, MonitoringDashboard
from .visualization import ForestVisualizer
from .cooperation import (
    CooperationSystem,
    CommunicationChannel,
    FederatedLearning,
    TransferLearning,
    CommunicationMessage,
)
from .environmental_sim import (
    EnvironmentalSimulator,
    DataDistributionShift,
    ClimateType,
    StressorType,
    EnvironmentalState,
)
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
    "ForestVisualizer",
    "CooperationSystem",
    "CommunicationChannel",
    "FederatedLearning",
    "TransferLearning",
    "CommunicationMessage",
    "EnvironmentalSimulator",
    "DataDistributionShift",
    "ClimateType",
    "StressorType",
    "EnvironmentalState",
    "AutoMLOrchestrator",
    "ContinuousGeneralizationTester",
    "RegressionValidator",
    "MetricAlerter",
    "TestResult",
    "BenchmarkResult",
]
