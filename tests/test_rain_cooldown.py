"""
Tests for the adaptive "rain" cooldown system in SelfImprovementLoop.

Verifies:
- Cooldowns are reduced when fitness > threshold and memory < threshold
- Cooldowns remain normal under stressed conditions
- Minimum cooldown of 1 cycle is always maintained
- rain_active field appears in run_cycle results
- ImprovementConfig rain parameters are configurable
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "evolution.self_improvement",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evolution",
        "self_improvement.py",
    ),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["evolution.self_improvement"] = _mod
_spec.loader.exec_module(_mod)

SelfImprovementLoop = _mod.SelfImprovementLoop
ImprovementConfig = _mod.ImprovementConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loop(config=None):
    """Build a SelfImprovementLoop with lightweight mocked forest/consciousness."""
    forest = MagicMock()
    forest.trees = []
    forest.mulch = MagicMock()
    forest.mulch.__len__ = MagicMock(return_value=0)
    forest.mulch.capacity = 100
    forest.anchors = MagicMock()
    forest.anchors.__len__ = MagicMock(return_value=0)
    forest.anchors.capacity = 100
    forest.num_trees = MagicMock(return_value=3)

    consciousness = MagicMock()
    consciousness.reflect = MagicMock(return_value={})

    return SelfImprovementLoop(forest, consciousness, config=config)


def _set_metrics(loop, avg_fitness, memory_usage):
    """Inject specific metrics into the loop's current_metrics."""
    loop.current_metrics = {
        "average_fitness": avg_fitness,
        "memory_usage": memory_usage,
        "num_trees": 3,
        "anchor_util": 0.0,
    }


# ---------------------------------------------------------------------------
# ImprovementConfig rain parameters
# ---------------------------------------------------------------------------

class TestImprovementConfigRainParams:
    def test_default_rain_thresholds(self):
        cfg = ImprovementConfig()
        assert cfg.rain_fitness_threshold == 5.0
        assert cfg.rain_memory_threshold == 0.70
        assert cfg.rain_cooldown_multiplier == 0.6

    def test_custom_rain_thresholds(self):
        cfg = ImprovementConfig(
            rain_fitness_threshold=3.0,
            rain_memory_threshold=0.50,
            rain_cooldown_multiplier=0.8,
        )
        assert cfg.rain_fitness_threshold == 3.0
        assert cfg.rain_memory_threshold == 0.50
        assert cfg.rain_cooldown_multiplier == 0.8

    def test_rain_disabled_via_multiplier_one(self):
        """Setting multiplier to 1.0 effectively disables rain (no reduction)."""
        cfg = ImprovementConfig(rain_cooldown_multiplier=1.0)
        loop = _make_loop(cfg)
        _set_metrics(loop, avg_fitness=10.0, memory_usage=0.1)  # very healthy

        # With multiplier=1.0, effective == base
        for action in cfg.cooldown_cycles:
            base = cfg.cooldown_cycles[action]
            effective = loop._get_effective_cooldown(action)
            assert effective == base


# ---------------------------------------------------------------------------
# _get_effective_cooldown
# ---------------------------------------------------------------------------

class TestGetEffectiveCooldown:
    def test_healthy_conditions_reduce_cooldown(self):
        loop = _make_loop()
        _set_metrics(loop, avg_fitness=8.0, memory_usage=0.50)

        # Actions with base_cd > 1 should be reduced; those with base_cd=1
        # remain at 1 (minimum floor).
        for action, base_cd in loop.config.cooldown_cycles.items():
            effective = loop._get_effective_cooldown(action)
            if base_cd > 1:
                assert effective < base_cd, f"{action}: {effective} should be < {base_cd}"
            else:
                # Already at minimum; floor keeps it at 1
                assert effective == 1, f"{action}: effective should be 1, got {effective}"

    def test_healthy_conditions_minimum_one(self):
        """Effective cooldown must be at least 1 even when healthy."""
        cfg = ImprovementConfig(rain_cooldown_multiplier=0.1)
        loop = _make_loop(cfg)
        _set_metrics(loop, avg_fitness=10.0, memory_usage=0.10)

        for action in loop.config.cooldown_cycles:
            effective = loop._get_effective_cooldown(action)
            assert effective >= 1, f"{action}: effective cooldown must be >= 1"

    def test_stressed_fitness_no_reduction(self):
        """Low fitness → no rain → full cooldown."""
        loop = _make_loop()
        _set_metrics(loop, avg_fitness=2.0, memory_usage=0.40)

        for action, base_cd in loop.config.cooldown_cycles.items():
            effective = loop._get_effective_cooldown(action)
            assert effective == base_cd

    def test_stressed_memory_no_reduction(self):
        """High memory → no rain → full cooldown."""
        loop = _make_loop()
        _set_metrics(loop, avg_fitness=10.0, memory_usage=0.85)

        for action, base_cd in loop.config.cooldown_cycles.items():
            effective = loop._get_effective_cooldown(action)
            assert effective == base_cd

    def test_zero_base_cooldown_returns_zero(self):
        cfg = ImprovementConfig()
        cfg.cooldown_cycles["test_action"] = 0
        loop = _make_loop(cfg)
        _set_metrics(loop, avg_fitness=10.0, memory_usage=0.10)
        assert loop._get_effective_cooldown("test_action") == 0

    def test_unknown_action_returns_zero(self):
        loop = _make_loop()
        _set_metrics(loop, avg_fitness=10.0, memory_usage=0.10)
        assert loop._get_effective_cooldown("nonexistent_action") == 0


# ---------------------------------------------------------------------------
# _is_rain_active
# ---------------------------------------------------------------------------

class TestIsRainActive:
    def test_rain_active_when_healthy(self):
        loop = _make_loop()
        _set_metrics(loop, avg_fitness=8.0, memory_usage=0.50)
        assert loop._is_rain_active() is True

    def test_rain_inactive_low_fitness(self):
        loop = _make_loop()
        _set_metrics(loop, avg_fitness=3.0, memory_usage=0.50)
        assert loop._is_rain_active() is False

    def test_rain_inactive_high_memory(self):
        loop = _make_loop()
        _set_metrics(loop, avg_fitness=8.0, memory_usage=0.80)
        assert loop._is_rain_active() is False

    def test_rain_inactive_both_bad(self):
        loop = _make_loop()
        _set_metrics(loop, avg_fitness=1.0, memory_usage=0.95)
        assert loop._is_rain_active() is False

    def test_rain_boundary_fitness_exact_threshold(self):
        """Fitness exactly at threshold is NOT above threshold → inactive."""
        loop = _make_loop()
        _set_metrics(loop, avg_fitness=5.0, memory_usage=0.50)
        assert loop._is_rain_active() is False

    def test_rain_boundary_memory_exact_threshold(self):
        """Memory exactly at threshold is NOT below threshold → inactive."""
        loop = _make_loop()
        _set_metrics(loop, avg_fitness=8.0, memory_usage=0.70)
        assert loop._is_rain_active() is False


# ---------------------------------------------------------------------------
# _is_on_cooldown with rain effect
# ---------------------------------------------------------------------------

class TestIsOnCooldownWithRain:
    def test_cooldown_expires_sooner_when_raining(self):
        """Under rain, cooldown should expire sooner (action not on cooldown earlier)."""
        action = "snapshot_teacher"  # base_cd = 2
        loop = _make_loop()

        # Simulate: cooldown was applied at cycle 1, expires at cycle 3
        loop.cycle_count = 1
        loop._apply_cooldown(action)
        assert loop._cooldown_until_cycle[action] == 3

        # At cycle 2 (still within normal cooldown), advance to cycle 2
        loop.cycle_count = 2

        # With stressed conditions → still on cooldown
        _set_metrics(loop, avg_fitness=1.0, memory_usage=0.90)
        assert loop._is_on_cooldown(action) is True

        # With healthy conditions (rain reduces by 40%, base=2 → effective=1)
        # rain_reduction = 2 - 1 = 1, effective_until = 3 - 1 = 2
        # cycle_count=2 is NOT < effective_until=2 → NOT on cooldown
        _set_metrics(loop, avg_fitness=8.0, memory_usage=0.50)
        assert loop._is_on_cooldown(action) is False

    def test_no_cooldown_when_not_applied(self):
        loop = _make_loop()
        _set_metrics(loop, avg_fitness=8.0, memory_usage=0.50)
        assert loop._is_on_cooldown("plant_trees") is False

    def test_full_cooldown_without_rain(self):
        action = "prune_trees"  # base_cd = 2
        loop = _make_loop()
        loop.cycle_count = 1
        loop._apply_cooldown(action)

        loop.cycle_count = 2
        _set_metrics(loop, avg_fitness=1.0, memory_usage=0.90)  # stressed
        assert loop._is_on_cooldown(action) is True

        loop.cycle_count = 3
        assert loop._is_on_cooldown(action) is False


# ---------------------------------------------------------------------------
# run_cycle result includes rain_active
# ---------------------------------------------------------------------------

class TestRunCycleRainActive:
    def test_rain_active_key_present_in_result(self):
        loop = _make_loop()

        # Patch _collect_metrics to return controlled values
        loop._collect_metrics = MagicMock(return_value={
            "average_fitness": 8.0,
            "memory_usage": 0.50,
            "num_trees": 3,
            "memory_util": 0.0,
            "anchor_util": 0.0,
        })
        loop._find_opportunities = MagicMock(return_value=[])

        result = loop.run_cycle()
        assert "rain_active" in result

    def test_rain_active_true_when_healthy(self):
        loop = _make_loop()

        healthy_metrics = {
            "average_fitness": 8.0,
            "memory_usage": 0.50,
            "num_trees": 3,
            "memory_util": 0.0,
            "anchor_util": 0.0,
        }
        loop._collect_metrics = MagicMock(return_value=healthy_metrics)
        loop._find_opportunities = MagicMock(return_value=[])

        result = loop.run_cycle()
        assert result["rain_active"] is True

    def test_rain_active_false_when_stressed(self):
        loop = _make_loop()

        stressed_metrics = {
            "average_fitness": 2.0,
            "memory_usage": 0.90,
            "num_trees": 3,
            "memory_util": 0.90,
            "anchor_util": 0.0,
        }
        loop._collect_metrics = MagicMock(return_value=stressed_metrics)
        loop._find_opportunities = MagicMock(return_value=[])

        result = loop.run_cycle()
        assert result["rain_active"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
