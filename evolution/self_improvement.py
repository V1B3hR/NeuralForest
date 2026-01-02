"""
Self-Improvement Loop for NeuralForest (Top-notch Edition)

This module implements a pragmatic, high-signal self-improvement controller
for ForestEcosystem + ForestConsciousness.

Key properties:
- Safe, reversible actions (no deepcopy of torch modules).
- Apply-then-validate per action; rollback only the last action if needed.
- Learns which actions work (bandit-style scoring) and adapts priorities.
- Cooldowns and safeguards prevent thrashing.
- Uses both: consciousness.reflect() + direct metrics.

Forest API assumed (matches NeuralForest.py):
- forest.trees (iterable with .id, .fitness, maybe .age/.bark)
- forest.mulch with: .data (deque-like of tuples) and .capacity, __len__
- forest.anchors with: .capacity, __len__
- forest.num_trees(), forest._plant_tree(), forest._prune_trees(ids, min_keep=?)
- forest.snapshot_teacher(), forest.teacher_snapshot

Note: Search-tool results may be incomplete; this implementation avoids calling
nonexistent methods such as reset_population().
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable
import time
import math
import random
from collections import deque, defaultdict

# Module logger (do not configure global logging here)
import logging
logger = logging.getLogger(__name__)


# ----------------------------
# Configuration + action model
# ----------------------------

@dataclass
class ImprovementConfig:
    # How many actions per cycle max
    max_improvements_default: int = 3

    # Validation tolerances
    # Allow small stochastic dips without rollback
    avg_fitness_drop_tolerance: float = 0.35
    # Hard safety ceiling
    memory_util_ceiling: float = 0.95
    anchor_util_ceiling: float = 0.98

    # Scoring weights for validation
    w_avg_fitness: float = 1.0
    w_memory_util: float = 0.4
    w_anchor_util: float = 0.2
    w_num_trees: float = 0.05  # small weight; size changes are not inherently good

    # Action cooldowns (cycles)
    cooldown_cycles: Dict[str, int] = None  # filled in __post_init__

    # Memory pruning behavior
    memory_keep_ratio: float = 0.70  # keep top 70% by priority

    # Tree pruning thresholds
    prune_min_keep: int = 3
    prune_if_fitness_below: float = 2.0

    # Exploration vs exploitation for action choice (bandit)
    exploration_epsilon: float = 0.15

    # History sizes
    max_history: int = 250
    max_recent_improvements_in_summary: int = 10

    def __post_init__(self):
        if self.cooldown_cycles is None:
            self.cooldown_cycles = {
                "plant_trees": 1,
                "prune_memory": 1,
                "snapshot_teacher": 2,
                "prune_trees": 2,
                "plant_specialist": 1,
            }


@dataclass
class ActionOutcome:
    timestamp: float
    action: str
    opportunity: Dict[str, Any]
    success: bool
    message: str
    duration: float
    # metrics deltas (optional)
    baseline_metrics: Optional[Dict[str, Any]] = None
    new_metrics: Optional[Dict[str, Any]] = None
    score_delta: Optional[float] = None
    # rollback info
    rolled_back: bool = False
    rollback_message: str = ""
    error: str = ""


# ----------------------------
# Self improvement loop
# ----------------------------

class SelfImprovementLoop:
    """
    Autonomous improvement cycle that runs continuously.

    Improvements over the original:
    - Per-action validation + rollback (reversible ops)
    - Learning which actions work (bandit scores)
    - Cooldowns and guardrails
    - More robust metrics + validation scoring
    """

    def __init__(self, forest, consciousness, config: Optional[ImprovementConfig] = None):
        self.forest = forest
        self.consciousness = consciousness
        self.config = config or ImprovementConfig()

        self.improvement_history: deque[Dict[str, Any]] = deque(maxlen=self.config.max_history)
        self.cycle_count = 0

        self.baseline_metrics: Optional[Dict[str, Any]] = None
        self.current_metrics: Optional[Dict[str, Any]] = None

        # Action learning (bandit-ish)
        # track: successes, attempts, mean_reward
        self.action_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"attempts": 0.0, "successes": 0.0, "mean_reward": 0.0})

        # Cooldowns per action
        self._cooldown_until_cycle: Dict[str, int] = {}

        # For reversible action rollback (only last action)
        self._last_undo: Optional[Callable[[], Tuple[bool, str]]] = None

    # ----------------------------
    # Public API
    # ----------------------------

    def run_cycle(self, max_improvements: Optional[int] = None, *, verbose: bool = False) -> Dict[str, Any]:
        """
        Run one self-improvement cycle.

        Applies up to max_improvements actions, validating after each.
        If an action fails validation, it is rolled back (when possible) and the cycle continues.
        """
        cycle_start = time.time()
        self.cycle_count += 1
        max_improvements = max_improvements if max_improvements is not None else self.config.max_improvements_default

        baseline = self._collect_metrics()
        if self.baseline_metrics is None:
            self.baseline_metrics = baseline

        analysis = self.consciousness.reflect()
        opportunities = self._find_opportunities(analysis, baseline)

        # Choose and apply actions iteratively
        applied_outcomes: List[Dict[str, Any]] = []
        rollbacks = 0

        # We keep a moving baseline within the cycle so each action is evaluated incrementally
        action_baseline = baseline

        for _ in range(max_improvements):
            chosen = self._choose_opportunity(opportunities)
            if chosen is None:
                break

            opp = chosen
            action = opp.get("action")

            outcome = self._apply_one_with_validation(opp, action_baseline, verbose=verbose)
            applied_outcomes.append(asdict(outcome))

            # Update learning stats
            self._update_action_stats(action, outcome)

            # Remove this opportunity so we don't repeat it immediately
            opportunities = [o for o in opportunities if o is not opp]

            # If action succeeded (and not rolled back), update baseline for next action
            if outcome.success and not outcome.rolled_back and outcome.new_metrics:
                action_baseline = outcome.new_metrics
            if outcome.rolled_back:
                rollbacks += 1

        self.current_metrics = self._collect_metrics()

        cycle_time = time.time() - cycle_start
        result = {
            "cycle": self.cycle_count,
            "opportunities_found": len(self._find_opportunities(analysis, baseline)),
            "improvements_attempted": len(applied_outcomes),
            "rollbacks": rollbacks,
            "baseline_metrics": baseline,
            "final_metrics": self.current_metrics,
            "cycle_time_seconds": cycle_time,
            "applied": applied_outcomes,
            "action_stats": {k: dict(v) for k, v in self.action_stats.items()},
        }

        # store cycle summary entry
        self.improvement_history.append({
            "timestamp": time.time(),
            "cycle": self.cycle_count,
            "result": result,
        })

        return result

    def get_improvement_summary(self) -> Dict[str, Any]:
        """Summarize performance of the self-improvement loop."""
        if not self.improvement_history:
            return {
                "total_cycles": 0,
                "total_actions": 0,
                "success_rate": 0.0,
                "action_stats": {},
                "recent_cycles": [],
            }

        total_actions = 0
        successes = 0
        for entry in self.improvement_history:
            applied = entry["result"].get("applied", [])
            total_actions += len(applied)
            successes += sum(1 for a in applied if a.get("success") and not a.get("rolled_back"))

        success_rate = (successes / total_actions) if total_actions else 0.0

        recent = list(self.improvement_history)[-self.config.max_recent_improvements_in_summary:]
        return {
            "total_cycles": len(self.improvement_history),
            "total_actions": total_actions,
            "successful_actions": successes,
            "success_rate": success_rate,
            "action_stats": {k: dict(v) for k, v in self.action_stats.items()},
            "recent_cycles": recent,
        }

    def reset(self):
        """Reset history, stats and metrics."""
        self.improvement_history.clear()
        self.cycle_count = 0
        self.baseline_metrics = None
        self.current_metrics = None
        self.action_stats.clear()
        self._cooldown_until_cycle.clear()
        self._last_undo = None

    # ----------------------------
    # Core mechanics
    # ----------------------------

    def _apply_one_with_validation(self, opportunity: Dict[str, Any], baseline: Dict[str, Any], *, verbose: bool) -> ActionOutcome:
        action = opportunity.get("action", "unknown")
        t0 = time.time()
        self._last_undo = None

        try:
            # apply
            apply_ok, msg = self._apply_action(action, opportunity)
            # collect metrics after
            new_metrics = self._collect_metrics()

            # score/validate
            ok, score_delta, reasons = self._validate(baseline, new_metrics)

            rolled_back = False
            rb_msg = ""
            if (not apply_ok) or (not ok):
                # rollback if possible
                if self._last_undo is not None:
                    rb_ok, rb_msg = self._last_undo()
                    rolled_back = True
                    rb_msg = f"{'OK' if rb_ok else 'FAILED'}: {rb_msg}"
                else:
                    rb_msg = "No undo available for this action."

                # mark cooldown even for failures to avoid thrash
                self._apply_cooldown(action)

                if verbose:
                    logger.warning(f"Action {action} failed/invalid. reasons={reasons}; rollback={rb_msg}")

                return ActionOutcome(
                    timestamp=t0,
                    action=action,
                    opportunity=opportunity,
                    success=False,
                    message=msg if apply_ok else f"Apply failed: {msg}",
                    duration=time.time() - t0,
                    baseline_metrics=baseline,
                    new_metrics=new_metrics,
                    score_delta=score_delta,
                    rolled_back=rolled_back,
                    rollback_message=rb_msg,
                    error="; ".join(reasons) if reasons else "",
                )

            # success -> set cooldown
            self._apply_cooldown(action)

            return ActionOutcome(
                timestamp=t0,
                action=action,
                opportunity=opportunity,
                success=True,
                message=msg,
                duration=time.time() - t0,
                baseline_metrics=baseline,
                new_metrics=new_metrics,
                score_delta=score_delta,
                rolled_back=False,
                rollback_message="",
                error="",
            )

        except Exception as e:
            self._apply_cooldown(action)
            return ActionOutcome(
                timestamp=t0,
                action=action,
                opportunity=opportunity,
                success=False,
                message=f"Exception in action '{action}'",
                duration=time.time() - t0,
                baseline_metrics=baseline,
                new_metrics=None,
                score_delta=None,
                rolled_back=False,
                rollback_message="",
                error=str(e),
            )

    def _apply_action(self, action: str, opportunity: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Apply one action and register an undo function when feasible.
        Only uses actions that exist in your repo today.
        """
        # Cooldown guard
        if self._is_on_cooldown(action):
            return False, f"Action '{action}' is on cooldown."

        if action == "plant_trees":
            before = self.forest.num_trees()
            self.forest._plant_tree()
            after = self.forest.num_trees()

            # undo: prune the newest tree (highest id) if we can identify it
            added_ids = self._tree_ids_set() - self._tree_ids_set_from_count(before)
            # fallback: compute by diffing ids before/after using snapshot
            # We'll do safer undo: remove the max-id tree if count increased
            def undo():
                if self.forest.num_trees() <= before:
                    return True, "No tree to undo (count already restored)."
                # remove the highest id tree
                try:
                    newest = max(self.forest.trees, key=lambda t: getattr(t, "id", -1))
                    self.forest._prune_trees([newest.id], min_keep=self.config.prune_min_keep)
                    return True, f"Pruned newly planted tree id={newest.id}"
                except Exception as ex:
                    return False, f"Could not undo plant_trees: {ex}"

            self._last_undo = undo
            ok = after > before
            return ok, f"Planted {after - before} tree(s)"

        if action == "snapshot_teacher":
            # snapshot is "side-effect" but harmless; undo by dropping reference
            self.forest.snapshot_teacher()

            def undo():
                self.forest.teacher_snapshot = None
                return True, "Cleared teacher_snapshot"

            self._last_undo = undo
            ok = getattr(self.forest, "teacher_snapshot", None) is not None
            return ok, "Created teacher snapshot for distillation"

        if action == "prune_memory":
            # reversible by storing removed items
            mulch = self.forest.mulch
            before_data = list(mulch.data)
            before_len = len(mulch)

            if before_len > 0:
                sorted_data = sorted(before_data, key=lambda x: x[2], reverse=True)
                keep_count = max(0, int(len(sorted_data) * self.config.memory_keep_ratio))
                new_data = sorted_data[:keep_count]
                mulch.data = deque(new_data, maxlen=mulch.capacity)

            after_len = len(mulch)
            removed = before_len - after_len

            def undo():
                try:
                    mulch.data = deque(before_data, maxlen=mulch.capacity)
                    return True, f"Restored mulch data to {len(mulch)} items"
                except Exception as ex:
                    return False, f"Could not restore mulch data: {ex}"

            self._last_undo = undo
            return True, f"Pruned {removed} low-priority memories"

        if action == "prune_trees":
            before_ids = [t.id for t in self.forest.trees]
            before_count = self.forest.num_trees()

            if not self.forest.trees:
                return False, "No trees to prune"

            weakest = min(self.forest.trees, key=lambda t: getattr(t, "fitness", 0.0))
            if getattr(weakest, "fitness", 999.0) >= self.config.prune_if_fitness_below:
                return False, "No trees weak enough to prune"

            if before_count <= self.config.prune_min_keep:
                return False, f"Cannot prune below min_keep={self.config.prune_min_keep}"

            self.forest._prune_trees([weakest.id], min_keep=self.config.prune_min_keep)
            after_count = self.forest.num_trees()

            # Undo is not feasible without model serialization; mark as non-reversible.
            self._last_undo = None

            return (after_count < before_count), f"Pruned weakest tree id={weakest.id} (fitness={weakest.fitness:.2f})"

        if action == "plant_specialist":
            # Base ForestEcosystem has no specialization API; we can still plant a tree.
            before = self.forest.num_trees()
            self.forest._plant_tree()
            after = self.forest.num_trees()

            details = opportunity.get("details", {}) or {}
            spec = details.get("task", details.get("reason", "general"))

            def undo():
                if self.forest.num_trees() <= before:
                    return True, "No tree to undo (count already restored)."
                try:
                    newest = max(self.forest.trees, key=lambda t: getattr(t, "id", -1))
                    self.forest._prune_trees([newest.id], min_keep=self.config.prune_min_keep)
                    return True, f"Pruned newly planted (specialist) tree id={newest.id}"
                except Exception as ex:
                    return False, f"Could not undo plant_specialist: {ex}"

            self._last_undo = undo
            ok = after > before
            return ok, f"Planted specialist-like tree for '{spec}'"

        return False, f"Unknown action: {action}"

    # ----------------------------
    # Opportunities / selection
    # ----------------------------

    def _find_opportunities(self, analysis: Dict[str, Any], metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        opportunities: List[Dict[str, Any]] = []

        # From consciousness
        num_trees = analysis.get("forest_size", metrics.get("num_trees", 0))
        overall_fitness = analysis.get("overall_fitness", metrics.get("average_fitness", 0.0))
        mem_util = analysis.get("memory_usage", {}).get("mulch_utilization", metrics.get("memory_usage", 0.0))

        tree_health = analysis.get("tree_health", {})
        weak_trees = tree_health.get("weak_trees", 0)

        # 1) Keep forest from being too small
        if num_trees < 5:
            opportunities.append({
                "type": "increase_capacity",
                "priority": 0.90,
                "action": "plant_trees",
                "reason": f"forest_size_low ({num_trees} trees)",
            })

        # 2) Performance improvements: snapshot teacher for distillation
        if overall_fitness < 5.0:
            opportunities.append({
                "type": "improve_performance",
                "priority": 0.75,
                "action": "snapshot_teacher",
                "reason": f"low_fitness ({overall_fitness:.2f})",
            })

        # 3) Memory pressure -> prune
        if mem_util > 0.90:
            opportunities.append({
                "type": "memory_optimization",
                "priority": 0.70,
                "action": "prune_memory",
                "reason": f"high_memory_usage ({mem_util:.1%})",
            })

        # 4) Weak trees -> prune (non-reversible, lower priority)
        if weak_trees > 0 and num_trees > max(3, self.config.prune_min_keep):
            opportunities.append({
                "type": "prune_weak",
                "priority": 0.45,
                "action": "prune_trees",
                "reason": f"weak_trees_present ({weak_trees})",
            })

        # 5) Knowledge gaps -> plant “specialist-like” tree
        for gap in analysis.get("knowledge_gaps", []) or []:
            opportunities.append({
                "type": "fill_gap",
                "priority": 0.65 * float(gap.get("severity", 1.0)),
                "action": "plant_specialist",
                "details": gap,
                "reason": f"knowledge_gap: {gap.get('reason', 'unknown')}",
            })

        # Sort by base priority (before bandit adjustment)
        opportunities.sort(key=lambda x: x.get("priority", 0.0), reverse=True)
        return opportunities

    def _choose_opportunity(self, opportunities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Choose next opportunity using:
        - filter cooldown
        - bandit-adjusted priority
        - epsilon-greedy exploration
        """
        if not opportunities:
            return None

        candidates = [o for o in opportunities if not self._is_on_cooldown(o.get("action", ""))]
        if not candidates:
            return None

        # Compute bandit-adjusted score
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for opp in candidates:
            action = opp.get("action", "unknown")
            base_p = float(opp.get("priority", 0.0))

            st = self.action_stats[action]
            attempts = st["attempts"]
            mean_reward = st["mean_reward"]

            # UCB-like bonus (lightweight)
            bonus = 0.0
            if attempts > 0:
                bonus = 0.15 * math.sqrt(math.log(max(2.0, self.cycle_count)) / attempts)
            else:
                bonus = 0.25  # encourage trying unknown actions

            adjusted = base_p * (0.6 + 0.4 * (0.5 + mean_reward)) + bonus
            scored.append((adjusted, opp))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Epsilon greedy: sometimes explore among top few
        if random.random() < self.config.exploration_epsilon and len(scored) > 1:
            topk = min(3, len(scored))
            return random.choice([opp for _, opp in scored[:topk]])

        return scored[0][1]

    # ----------------------------
    # Validation / metrics
    # ----------------------------

    def _collect_metrics(self) -> Dict[str, Any]:
        trees = getattr(self.forest, "trees", None)
        if not trees:
            return {
                "num_trees": 0,
                "average_fitness": 0.0,
                "min_fitness": 0.0,
                "max_fitness": 0.0,
                "memory_usage": 0.0,
                "anchor_usage": 0.0,
            }

        fitnesses = [float(getattr(t, "fitness", 0.0)) for t in trees]
        mulch = self.forest.mulch
        anchors = self.forest.anchors

        return {
            "num_trees": len(trees),
            "average_fitness": sum(fitnesses) / max(1, len(fitnesses)),
            "min_fitness": min(fitnesses),
            "max_fitness": max(fitnesses),
            "memory_usage": (len(mulch) / mulch.capacity) if getattr(mulch, "capacity", 0) else 0.0,
            "anchor_usage": (len(anchors) / anchors.capacity) if getattr(anchors, "capacity", 0) else 0.0,
        }

    def _validate(self, baseline: Dict[str, Any], new_metrics: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """
        Returns:
            ok, score_delta, reasons
        """
        reasons: List[str] = []

        # Hard safety checks
        if new_metrics.get("memory_usage", 0.0) > self.config.memory_util_ceiling:
            reasons.append(f"memory_util_above_ceiling ({new_metrics['memory_usage']:.3f} > {self.config.memory_util_ceiling:.3f})")
        if new_metrics.get("anchor_usage", 0.0) > self.config.anchor_util_ceiling:
            reasons.append(f"anchor_util_above_ceiling ({new_metrics['anchor_usage']:.3f} > {self.config.anchor_util_ceiling:.3f})")

        # Fitness tolerance check
        base_fit = float(baseline.get("average_fitness", 0.0))
        new_fit = float(new_metrics.get("average_fitness", 0.0))
        if (base_fit - new_fit) > self.config.avg_fitness_drop_tolerance:
            reasons.append(f"avg_fitness_drop ({base_fit:.3f} -> {new_fit:.3f})")

        # Weighted score delta (higher is better)
        base_score = self._score_metrics(baseline)
        new_score = self._score_metrics(new_metrics)
        score_delta = new_score - base_score

        # If score delta is strongly negative, treat as fail even if tolerances passed
        if score_delta < -0.15:
            reasons.append(f"score_delta_too_negative ({score_delta:.3f})")

        ok = len(reasons) == 0
        return ok, score_delta, reasons

    def _score_metrics(self, m: Dict[str, Any]) -> float:
        """
        Higher score = better.
        We reward fitness, penalize high utilization, lightly reward sufficient size.
        """
        avg_fit = float(m.get("average_fitness", 0.0))
        mem = float(m.get("memory_usage", 0.0))
        anc = float(m.get("anchor_usage", 0.0))
        n = float(m.get("num_trees", 0.0))

        # squash fitness to keep scale sane
        fit_term = math.tanh(avg_fit / 6.0)

        # penalize near-capacity utilization smoothly
        mem_pen = mem ** 2
        anc_pen = anc ** 2

        # small benefit for having some trees, saturating quickly
        size_term = math.tanh(n / 8.0)

        return (
            self.config.w_avg_fitness * fit_term
            - self.config.w_memory_util * mem_pen
            - self.config.w_anchor_util * anc_pen
            + self.config.w_num_trees * size_term
        )

    # ----------------------------
    # Learning + cooldowns
    # ----------------------------

    def _update_action_stats(self, action: str, outcome: ActionOutcome) -> None:
        st = self.action_stats[action]
        st["attempts"] += 1.0
        if outcome.success and not outcome.rolled_back:
            st["successes"] += 1.0

        # reward: use score_delta if present, else +1/-1
        if outcome.score_delta is not None:
            reward = float(outcome.score_delta)
        else:
            reward = 1.0 if (outcome.success and not outcome.rolled_back) else -1.0

        # EMA update
        alpha = 0.15
        st["mean_reward"] = (1 - alpha) * st["mean_reward"] + alpha * reward

    def _apply_cooldown(self, action: str) -> None:
        cd = int(self.config.cooldown_cycles.get(action, 0))
        if cd > 0:
            self._cooldown_until_cycle[action] = self.cycle_count + cd

    def _is_on_cooldown(self, action: str) -> bool:
        until = self._cooldown_until_cycle.get(action, 0)
        return self.cycle_count < until

    # ----------------------------
    # Helpers
    # ----------------------------

    def _tree_ids_set(self) -> set:
        return {getattr(t, "id", None) for t in self.forest.trees}

    def _tree_ids_set_from_count(self, count: int) -> set:
        # Not truly possible without snapshot; keep empty. (used only as a harmless hint)
        return set()
