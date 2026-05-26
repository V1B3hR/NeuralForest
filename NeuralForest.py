# NeuralForest v2.1 — single-cell full script (continual learning + per-tree NAS-ready architecture)
# Adds:
# - Per-tree configurable architectures (each tree stores its own arch)
# - Correct teacher snapshot and checkpointing for heterogeneous trees
# - Safe residual (learnable skip projection) (no torch.eye hacks)
#
# Dependencies: torch, numpy, matplotlib, networkx

import math
import random
import logging
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ----------------------------
# 0) Utilities
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


# ----------------------------
# 1) Memory: Prioritized Replay + Coreset anchors
# ----------------------------
class PrioritizedMulch:
    """
    Stores experiences with priorities and supports weighted sampling.
    Item: (x, y, priority, features)
    """

    def __init__(self, capacity=8000, alpha=0.7, eps=1e-3):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.data = deque(maxlen=capacity)

    @property
    def buffer(self):
        """Backward-compat alias for ``self.data``."""
        return self.data

    def __len__(self):
        return len(self.data)

    def add(self, x, y, priority, features=None):
        p = float(abs(priority) + self.eps)
        feat = features.detach().cpu() if features is not None else None
        self.data.append((x.detach().cpu(), y.detach().cpu(), p, feat))

    def sample(self, batch_size, mix_hard=0.6):
        n = len(self.data)
        if n < batch_size:
            return None, None

        hard_n = int(batch_size * mix_hard)
        rand_n = batch_size - hard_n
        xs, ys = [], []

        if hard_n > 0:
            priorities = torch.tensor(
                [item[2] for item in self.data], dtype=torch.float32
            )
            probs = priorities.pow(self.alpha)
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, num_samples=hard_n, replacement=(hard_n > n))
            for i in idx.tolist():
                x, y, _p, _f = self.data[i]
                xs.append(x)
                ys.append(y)

        if rand_n > 0:
            batch = random.sample(self.data, rand_n)
            for x, y, _p, _f in batch:
                xs.append(x)
                ys.append(y)

        batch_x = torch.stack(xs).to(DEVICE)
        batch_y = torch.stack(ys).to(DEVICE)
        return batch_x, batch_y

    def sample_features(self, batch_size: int) -> Optional[torch.Tensor]:
        """
        Return a feature sample from mulch, weighted by priority.
        """
        available = [item for item in self.data if item[3] is not None]
        if len(available) < batch_size:
            return None

        weights = torch.tensor([item[2] for item in available], dtype=torch.float32)
        weights = weights / weights.sum()
        idx = torch.multinomial(
            weights,
            num_samples=min(batch_size, len(available)),
            replacement=False,
        )
        chosen = [available[i] for i in idx.tolist()]
        return torch.stack([item[3] for item in chosen])


class AnchorCoreset:
    """
    Keeps a small representative set of anchors (x,y).
    """

    def __init__(self, capacity=256):
        self.capacity = capacity
        self.data = []  # list of (x, y)

    def __len__(self):
        return len(self.data)

    def add(self, x, y):
        x = x.detach().cpu()
        y = y.detach().cpu()

        if len(self.data) < self.capacity:
            self.data.append((x, y))
            return

        xs = torch.stack([item[0] for item in self.data])
        dists = (xs - x).view(len(xs), -1).pow(2).mean(dim=1)
        min_dist_new = dists.min().item()

        mean_x = xs.mean(dim=0, keepdim=True)
        redund = (xs - mean_x).view(len(xs), -1).pow(2).mean(dim=1)
        replace_idx = int(torch.argmin(redund).item())

        if min_dist_new > 1e-3:
            self.data[replace_idx] = (x, y)

    def sample(self, batch_size):
        if len(self.data) < batch_size:
            return None, None
        batch = random.sample(self.data, batch_size)
        x = torch.stack([b[0] for b in batch]).to(DEVICE)
        y = torch.stack([b[1] for b in batch]).to(DEVICE)
        return x, y


# ----------------------------
# 2) Routing / gating
# ----------------------------
class GatingRouter(nn.Module):
    def __init__(self, input_dim, max_trees, hidden=32, balance_coeff=0.01):
        super().__init__()
        self.max_trees = max_trees
        self.balance_coeff = balance_coeff
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, max_trees)
        )
        # EMA utilization histogram — one slot per possible tree (no-grad buffer)
        self.register_buffer("utilization_ema", torch.zeros(max_trees))
        self._ema_alpha = 0.01  # slow-moving so monitoring is stable

    def forward(self, x, num_trees):
        scores = self.net(x)[:, :num_trees]  # [B, T]
        return scores

    def balance_loss(self, routing_weights):
        """KL(p_router || uniform) load-balancing auxiliary loss.

        Args:
            routing_weights: ``[B, T]`` soft routing weights (post-softmax).

        Returns:
            Scalar tensor — zero when all trees receive equal traffic.
        """
        mean_w = routing_weights.mean(dim=0)  # [T]
        T = mean_w.shape[0]
        uniform = torch.ones_like(mean_w) / T
        eps = 1e-8
        kl = (mean_w * (torch.log(mean_w + eps) - torch.log(uniform + eps))).sum()
        return kl

    @torch.no_grad()
    def update_utilization(self, routing_weights):
        """Update EMA utilization histogram for expert-utilization monitoring."""
        mean_usage = routing_weights.mean(dim=0)  # [T]
        T = mean_usage.shape[0]
        self.utilization_ema[:T] = (
            (1 - self._ema_alpha) * self.utilization_ema[:T]
            + self._ema_alpha * mean_usage
        )

    def get_utilization_stats(self, num_trees):
        """Return a dict with per-tree EMA utilization and summary stats."""
        utils = self.utilization_ema[:num_trees]
        return {
            "utilization": utils.tolist(),
            "min": float(utils.min()),
            "max": float(utils.max()),
            "std": float(utils.std()),
        }


def topk_softmax(scores, k):
    B, T = scores.shape
    k = min(k, T)
    topv, topi = torch.topk(scores, k=k, dim=1)
    w = torch.softmax(topv, dim=1)
    weights = torch.zeros_like(scores)
    weights.scatter_(1, topi, w)
    return weights


# ----------------------------
# 3) Per-tree architecture (NAS-ready)
# ----------------------------
@dataclass(frozen=True)
class TreeArch:
    # "depth" counts hidden layers (>=1)
    num_layers: int = 1
    hidden_dim: int = 32
    activation: str = "tanh"  # relu|gelu|tanh|swish
    dropout: float = 0.0
    normalization: str = "none"  # none|layer|batch
    residual: bool = False

    def to_dict(self):
        return asdict(self)


def _make_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name in ("swish", "silu"):
        return nn.SiLU()
    return nn.Tanh()


def _make_norm(kind: str, dim: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


class TreeExpert(nn.Module):
    """
    Tree with its own architecture.

    Keeps compatibility with the rest of your ecosystem:
    - id, age, bark, fitness
    - step_age(), update_fitness()
    """

    def __init__(self, input_dim: int, tree_id: int, arch: TreeArch):
        super().__init__()
        self.id = tree_id
        self.arch = arch  # IMPORTANT: per-tree arch stored here

        self.age = 0
        self.bark = 0.0
        self.fitness = 5.0
        self._ema_loss = None  # EMA baseline for improvement-relative fitness

        # Build MLP trunk
        layers = []
        in_dim = input_dim

        # We do residual only as a single skip from input->hidden for stability
        self.use_residual = bool(arch.residual and arch.num_layers >= 2)
        self.skip_proj = None
        if self.use_residual and in_dim != arch.hidden_dim:
            self.skip_proj = nn.Linear(in_dim, arch.hidden_dim, bias=False)

        for _ in range(max(1, arch.num_layers)):
            layers.append(nn.Linear(in_dim, arch.hidden_dim))
            layers.append(_make_norm(arch.normalization, arch.hidden_dim))
            layers.append(_make_activation(arch.activation))
            if arch.dropout and arch.dropout > 0.0:
                layers.append(nn.Dropout(float(arch.dropout)))
            in_dim = arch.hidden_dim

        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(arch.hidden_dim, 1)

    def forward(self, x):
        # x: [B, input_dim]
        h = self.trunk(x)
        if self.use_residual:
            skip = x if self.skip_proj is None else self.skip_proj(x)
            # only add skip if shapes align
            if skip.shape == h.shape:
                h = h + skip
        return self.head(h)

    def step_age(self):
        self.age += 1
        if self.age > 80:
            self.bark = min(0.985, self.bark + 0.01)

    def update_fitness(self, loss_value):
        loss_f = float(loss_value)

        # ── EMA baseline: tracks long-run average loss for this tree ──────────
        if self._ema_loss is None:
            self._ema_loss = loss_f
        else:
            self._ema_loss = 0.95 * self._ema_loss + 0.05 * loss_f

        # ── Bounded, scaled reward (same as before) ────────────────────────────
        reward = 10.0 / (loss_f + 0.1)
        reward = min(reward, 10.0)

        # ── Improvement bonus: reward improvement over EMA baseline ───────────
        # Positive when current loss < EMA baseline (tree is getting better).
        improvement = (self._ema_loss - loss_f) / (self._ema_loss + 1e-6)
        improvement_bonus = max(0.0, improvement) * 3.0
        combined = min(reward + improvement_bonus, 10.0)

        self.fitness = 0.97 * self.fitness + 0.03 * combined


# ----------------------------
# 4) Forest ecosystem (per-tree arch + correct snapshot + checkpoints)
# ----------------------------
class ForestEcosystem(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, max_trees=24, enable_graveyard=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_trees = max_trees

        self.graph = nx.Graph()
        self.trees = nn.ModuleList()
        self.router = GatingRouter(input_dim, max_trees=max_trees, hidden=32)

        self.mulch = PrioritizedMulch(capacity=10000, alpha=0.7)
        self.anchors = AnchorCoreset(capacity=256)

        self.tree_counter = 0

        # Topology version — incremented on every plant or prune so that
        # external task heads can detect structural changes and rebuild.
        self.topology_version = 0

        # Tree Graveyard for Phase 3b: Legacy & Memory Management
        self.enable_graveyard = enable_graveyard
        if enable_graveyard:
            try:
                from evolution.tree_graveyard import TreeGraveyard
                self.graveyard = TreeGraveyard(
                    max_records=10000,
                    save_weights=False,  # Can enable for full archival
                    auto_save=True,
                )
            except ImportError:
                self.graveyard = None
        else:
            self.graveyard = None
        
        # Track current generation for graveyard
        self.current_generation = 0

        # Optional per-forest distribution / defaults for new trees
        self.default_arch = TreeArch(
            num_layers=1,
            hidden_dim=hidden_dim,
            activation="tanh",
            dropout=0.0,
            normalization="none",
            residual=False,
        )

        self._plant_tree()  # start with one tree

        self.teacher_snapshot = None

    def num_trees(self):
        return len(self.trees)

    def _plant_tree(self, arch: Optional[TreeArch] = None):
        if self.num_trees() >= self.max_trees:
            return

        if arch is None:
            arch = self.default_arch

        t = TreeExpert(self.input_dim, self.tree_counter, arch).to(DEVICE)
        self.trees.append(t)
        self.graph.add_node(t.id)

        # connect to most similar existing tree (param-distance heuristic)
        if self.num_trees() > 1:
            new_tree = t
            best = None
            best_dist = float("inf")
            for other in self.trees[:-1]:
                dist = 0.0
                try:
                    for p1, p2 in zip(new_tree.parameters(), other.parameters()):
                        # Only compare if shapes match
                        if p1.shape == p2.shape:
                            dist += (p1.detach() - p2.detach()).norm().item()
                        else:
                            # Use a shape mismatch penalty
                            dist += 100.0
                except Exception:
                    dist = float("inf")
                
                if dist < best_dist:
                    best_dist = dist
                    best = other
            if best is not None:
                self.graph.add_edge(new_tree.id, best.id, weight=2.0)

        self.tree_counter += 1
        self.topology_version += 1

    def _prune_trees(self, ids_to_remove, min_keep=2, reason="low_fitness", resource_history=None):
        """
        Prune trees from the forest, archiving them in the graveyard before removal.
        
        Args:
            ids_to_remove: List of tree IDs to remove
            min_keep: Minimum number of trees to keep in the forest
            reason: Reason for elimination (for graveyard records)
            resource_history: Optional resource allocation history for eliminated trees
        """
        if self.num_trees() <= min_keep:
            return

        keep = [t for t in self.trees if t.id not in set(ids_to_remove)]
        if len(keep) < min_keep:
            sorted_by_fit = sorted(
                list(self.trees), key=lambda t: t.fitness, reverse=True
            )
            keep = sorted_by_fit[:min_keep]

        removed_ids = {t.id for t in self.trees} - {t.id for t in keep}
        
        # Archive eliminated trees to graveyard before removal
        if self.graveyard is not None and removed_ids:
            trees_to_archive = [t for t in self.trees if t.id in removed_ids]
            for tree in trees_to_archive:
                # Extract parent IDs from graph edges
                parent_ids = []
                if self.graph.has_node(tree.id):
                    parent_ids = list(self.graph.neighbors(tree.id))
                
                # Archive the tree
                self.graveyard.archive_tree(
                    tree=tree,
                    elimination_reason=reason,
                    generation=self.current_generation,
                    recent_disruptions=[],  # Could be tracked separately
                    resource_history=resource_history,
                    parent_ids=parent_ids,
                    children_ids=[],  # Could be tracked if we maintain genealogy
                )
        
        self.trees = nn.ModuleList(keep).to(DEVICE)

        for rid in removed_ids:
            if self.graph.has_node(rid):
                self.graph.remove_node(rid)

        if removed_ids:
            self.topology_version += 1
    
    def resurrect_tree(self, tree_id: Optional[int] = None, min_fitness: float = 3.0):
        """
        Resurrect a tree from the graveyard and plant it in the forest.
        
        Args:
            tree_id: Specific tree ID to resurrect (if None, picks best candidate)
            min_fitness: Minimum fitness threshold for auto-selection
        
        Returns:
            The resurrected tree if successful, None otherwise
        """
        if self.graveyard is None:
            return None
        
        if self.num_trees() >= self.max_trees:
            return None
        
        # Get resurrection candidate
        if tree_id is not None:
            record = self.graveyard.get_record(tree_id)
            if record is None:
                return None
        else:
            # Get best candidate automatically
            candidates = self.graveyard.get_resurrection_candidates(
                min_fitness=min_fitness,
                limit=1
            )
            if not candidates:
                return None
            record = candidates[0]
        
        # Resurrect the tree
        resurrected = self.graveyard.resurrect_tree(
            record=record,
            tree_class=TreeExpert,
            input_dim=self.input_dim,
            new_tree_id=self.tree_counter,
        )
        
        # Add to forest
        self.trees.append(resurrected.to(DEVICE))
        self.graph.add_node(resurrected.id)
        
        # Connect to existing trees (same as _plant_tree)
        if self.num_trees() > 1:
            best = None
            best_dist = float("inf")
            for other in self.trees[:-1]:
                dist = 0.0
                try:
                    for p1, p2 in zip(resurrected.parameters(), other.parameters()):
                        # Only compare if shapes match
                        if p1.shape == p2.shape:
                            dist += (p1.detach() - p2.detach()).norm().item()
                        else:
                            # Use a shape mismatch penalty
                            dist += 100.0
                except Exception:
                    dist = float("inf")
                
                if dist < best_dist:
                    best_dist = dist
                    best = other
            if best is not None:
                self.graph.add_edge(resurrected.id, best.id, weight=2.0)
        
        self.tree_counter += 1
        
        return resurrected

    @torch.no_grad()
    def snapshot_teacher(self):
        teacher = ForestTeacher(self).to(DEVICE)
        teacher.eval()
        self.teacher_snapshot = teacher

    def forward_forest(self, x, top_k=3):
        T = self.num_trees()
        scores = self.router(x, num_trees=T)
        weights = topk_softmax(scores, k=top_k)

        outs = [t(x) for t in self.trees]  # each [B,1]
        out_stack = torch.stack(outs, dim=1)  # [B,T,1]
        y = (out_stack * weights.unsqueeze(-1)).sum(dim=1)
        return y, weights, outs

    def apply_bark_gradient_mask(self):
        for t in self.trees:
            if t.bark > 0:
                for p in t.parameters():
                    if p.grad is not None:
                        p.grad.mul_(1.0 - t.bark)

    @torch.no_grad()
    def update_ages(self):
        for t in self.trees:
            t.step_age()

    # --------- checkpoints (now store per-tree arch) ---------
    def save_checkpoint(self, path, metadata: Optional[dict] = None):
        import os

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        tree_states = []
        for t in self.trees:
            tree_states.append(
                {
                    "state_dict": t.state_dict(),
                    "id": t.id,
                    "age": t.age,
                    "bark": t.bark,
                    "fitness": t.fitness,
                    "arch": t.arch.to_dict(),  # IMPORTANT
                }
            )

        mulch_data = [(x, y, p, feat) for x, y, p, feat in self.mulch.data]
        anchor_data = [(x, y) for x, y in self.anchors.data]
        graph_edges = list(self.graph.edges(data=True))

        checkpoint = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "max_trees": self.max_trees,
            "tree_counter": self.tree_counter,
            "default_arch": self.default_arch.to_dict(),
            "tree_states": tree_states,
            "router_state_dict": self.router.state_dict(),
            "mulch_data": mulch_data,
            "mulch_capacity": self.mulch.capacity,
            "mulch_alpha": self.mulch.alpha,
            "anchor_data": anchor_data,
            "anchor_capacity": self.anchors.capacity,
            "graph_edges": graph_edges,
        }

        if metadata:
            checkpoint["metadata"] = dict(metadata)

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)

    @staticmethod
    def _load_checkpoint_dict(path, device):
        """Load a trusted NeuralForest checkpoint artifact."""
        return torch.load(path, map_location=device, weights_only=True)

    @classmethod
    def load_checkpoint(cls, path, device=None):
        """Load a checkpoint produced by this project.

        Trusted artifacts only: this loader is intended for checkpoints saved by
        NeuralForest itself and should not be used on untrusted files.
        """
        if device is None:
            device = DEVICE

        checkpoint = cls._load_checkpoint_dict(path, device)

        required_keys = {
            "input_dim",
            "hidden_dim",
            "max_trees",
            "tree_counter",
            "tree_states",
            "router_state_dict",
            "mulch_data",
            "mulch_capacity",
            "mulch_alpha",
            "anchor_data",
            "anchor_capacity",
            "graph_edges",
        }
        missing_keys = sorted(required_keys - set(checkpoint))
        if missing_keys:
            raise ValueError(
                f"Checkpoint missing required keys: {', '.join(missing_keys)}"
            )

        forest = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            max_trees=checkpoint["max_trees"],
        ).to(device)

        forest.trees = nn.ModuleList()
        forest.graph.clear()
        forest.tree_counter = checkpoint["tree_counter"]
        forest.default_arch = TreeArch(
            **checkpoint.get(
                "default_arch",
                {
                    "num_layers": 1,
                    "hidden_dim": forest.hidden_dim,
                    "activation": "tanh",
                    "dropout": 0.0,
                    "normalization": "none",
                    "residual": False,
                },
            )
        )

        for tree_state in checkpoint["tree_states"]:
            arch_dict = tree_state.get("arch", forest.default_arch.to_dict())
            arch = TreeArch(**arch_dict)

            t = TreeExpert(forest.input_dim, tree_state["id"], arch).to(device)
            t.load_state_dict(tree_state["state_dict"])
            t.age = tree_state["age"]
            t.bark = tree_state["bark"]
            t.fitness = tree_state["fitness"]

            forest.trees.append(t)
            forest.graph.add_node(t.id)

        forest.router.load_state_dict(checkpoint["router_state_dict"])

        forest.mulch = PrioritizedMulch(
            capacity=checkpoint["mulch_capacity"], alpha=checkpoint["mulch_alpha"]
        )
        for item in checkpoint["mulch_data"]:
            if len(item) == 4:
                x, y, p, feat = item
                feat = feat.to(device) if feat is not None else None
            else:
                x, y, p = item
                feat = None
            forest.mulch.add(x.to(device), y.to(device), p, features=feat)

        forest.anchors = AnchorCoreset(capacity=checkpoint["anchor_capacity"])
        for x, y in checkpoint["anchor_data"]:
            forest.anchors.data.append((x.to(device), y.to(device)))

        for u, v, data in checkpoint["graph_edges"]:
            forest.graph.add_edge(u, v, **data)

        logger.info("Checkpoint loaded from %s", path)
        logger.info(
            "Trees: %s, Memory: %s, Anchors: %s",
            forest.num_trees(),
            len(forest.mulch),
            len(forest.anchors),
        )
        return forest


class ForestTeacher(nn.Module):
    """
    Snapshot teacher with correct per-tree architectures.
    """

    def __init__(self, forest: ForestEcosystem):
        super().__init__()
        self.input_dim = forest.input_dim
        self.max_trees = forest.max_trees
        first_param = next(forest.parameters(), None)
        device = first_param.device if first_param is not None else DEVICE

        self.router = GatingRouter(
            self.input_dim, max_trees=self.max_trees, hidden=32
        ).to(device)
        self.router.load_state_dict(
            {k: v.detach().clone() for k, v in forest.router.state_dict().items()}
        )

        self.trees = nn.ModuleList()
        for t in forest.trees:
            nt = TreeExpert(self.input_dim, t.id, t.arch).to(device)
            nt.load_state_dict(
                {k: v.detach().clone() for k, v in t.state_dict().items()}
            )
            nt.eval()
            self.trees.append(nt)

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x, top_k=3):
        T = len(self.trees)
        scores = self.router(x, num_trees=T)
        weights = topk_softmax(scores, k=min(top_k, T))
        outs = [t(x) for t in self.trees]
        out_stack = torch.stack(outs, dim=1)
        y = (out_stack * weights.unsqueeze(-1)).sum(dim=1)
        return y


# ----------------------------
# 4b) Topology-aware task head
# ----------------------------
class AdaptiveTaskHead(nn.Module):
    """
    Topology-aware task head that auto-rebuilds when the forest tree count changes.

    On every forward pass the head checks whether ``forest.topology_version``
    has changed since it was last built; if so it silently rebuilds its linear
    layers to match the new input dimension before running the forward pass.
    This prevents dimension mismatches after prune / plant operations.

    Args:
        forest: The ``ForestEcosystem`` instance this head is attached to.
        num_classes: Number of output classes / regression outputs.
        hidden_dim: Width of the single hidden layer (default 128).
        output_dim_per_tree: Feature dimension contributed by each tree
            (must match however ``forward_forest`` produces features, default 1).
        dropout: Dropout rate (default 0.0).
    """

    def __init__(
        self,
        forest: "ForestEcosystem",
        num_classes: int,
        hidden_dim: int = 128,
        output_dim_per_tree: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._forest = forest
        self._num_classes = num_classes
        self._hidden_dim = hidden_dim
        self._output_dim_per_tree = output_dim_per_tree
        self._dropout_rate = dropout
        self._last_topology = forest.topology_version
        self._build()

    # ------------------------------------------------------------------
    def _get_device(self):
        first_param = next(self._forest.parameters(), None)
        return first_param.device if first_param is not None else DEVICE

    def _build(self):
        in_dim = self._forest.num_trees() * self._output_dim_per_tree
        device = self._get_device()
        self.net = nn.Sequential(
            nn.Linear(in_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Dropout(self._dropout_rate),
            nn.Linear(self._hidden_dim, self._num_classes),
        ).to(device)
        self._last_topology = self._forest.topology_version

    def _maybe_rebuild(self):
        """Rebuild layers if the forest topology has changed."""
        if self._forest.topology_version != self._last_topology:
            self._build()

    # ------------------------------------------------------------------
    def forward(self, x):
        self._maybe_rebuild()
        return self.net(x)


# ----------------------------
# 5) Steward (meta-controller)
# ----------------------------
class Steward:
    def __init__(self, forest: ForestEcosystem):
        self.forest = forest
        self.loss_hist = deque(maxlen=40)
        self.drift_hist = deque(maxlen=40)
        self.last_teacher_snapshot_step = 0

        # Simple per-tree architecture proposal distribution (optional)
        self.arch_pool = [
            TreeArch(
                num_layers=1,
                hidden_dim=forest.hidden_dim,
                activation="tanh",
                dropout=0.0,
                normalization="none",
                residual=False,
            ),
            TreeArch(
                num_layers=2,
                hidden_dim=forest.hidden_dim,
                activation="tanh",
                dropout=0.1,
                normalization="layer",
                residual=False,
            ),
            TreeArch(
                num_layers=3,
                hidden_dim=64,
                activation="gelu",
                dropout=0.1,
                normalization="layer",
                residual=True,
            ),
            TreeArch(
                num_layers=4,
                hidden_dim=128,
                activation="swish",
                dropout=0.2,
                normalization="layer",
                residual=True,
            ),
        ]

    def compute_drift(self, x_batch):
        x = x_batch.detach()
        mu = x.mean().item()
        var = x.var(unbiased=False).item()
        return abs(mu) + 0.5 * abs(var - 1.0)

    def propose_arch(self) -> TreeArch:
        # For now: random from pool. Later: replace with NAS output / bandit learning.
        return random.choice(self.arch_pool)

    def step(self, step_idx, loss_value, x_batch):
        self.loss_hist.append(float(loss_value))
        drift = self.compute_drift(x_batch)
        self.drift_hist.append(float(drift))

        loss_avg = sum(self.loss_hist) / max(1, len(self.loss_hist))
        drift_avg = sum(self.drift_hist) / max(1, len(self.drift_hist))

        # 1) Plant new tree if struggling or drift is high
        if (
            loss_avg > 0.06 or drift_avg > 1.2
        ) and self.forest.num_trees() < self.forest.max_trees:
            if random.random() < 0.25:
                self.forest._plant_tree(arch=self.propose_arch())

        # 2) Prune weak old trees if the forest is big enough
        if self.forest.num_trees() > 4 and random.random() < 0.15:
            weak = []
            for t in self.forest.trees:
                if t.age > 60 and t.fitness < 2.0:
                    weak.append(t.id)
            if weak:
                self.forest._prune_trees(weak, min_keep=2)

        # 3) Periodic teacher snapshot ("sleep") for distillation
        if step_idx - self.last_teacher_snapshot_step > 50:
            if loss_avg < 0.08 or random.random() < 0.2:
                self.forest.snapshot_teacher()
                self.last_teacher_snapshot_step = step_idx


# ----------------------------
# 6) Training step
# ----------------------------
def train_step(
    forest: ForestEcosystem,
    steward: Steward,
    optimizer,
    x_batch,
    y_batch,
    step_idx,
    top_k=3,
    replay_ratio=1.0,
    anchor_ratio=0.4,
    distill_weight=0.25,
):
    forest.train()
    x_batch = x_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)

    optimizer.zero_grad(set_to_none=True)
    y_pred, weights, per_tree = forest.forward_forest(x_batch, top_k=top_k)
    loss_current = mse(y_pred, y_batch)

    # per-tree fitness update
    with torch.no_grad():
        for t, out in zip(forest.trees, per_tree):
            local = mse(out, y_batch).item()
            t.update_fitness(local)

    # store experiences
    with torch.no_grad():
        per_ex = (y_pred - y_batch).pow(2).view(len(x_batch), -1).mean(dim=1)
        best_trees = sorted(forest.trees, key=lambda t: t.fitness, reverse=True)[:3]
        for i in range(len(x_batch)):
            feats = [tree.trunk(x_batch[i : i + 1]).squeeze(0) for tree in best_trees]
            deposited_feat = torch.stack(feats).mean(dim=0) if feats else None
            forest.mulch.add(
                x_batch[i],
                y_batch[i],
                priority=per_ex[i].item(),
                features=deposited_feat,
            )
            forest.anchors.add(x_batch[i], y_batch[i])

    # replay
    loss_replay = torch.tensor(0.0, device=DEVICE)
    if replay_ratio > 0:
        rx, ry = forest.mulch.sample(
            batch_size=int(len(x_batch) * replay_ratio), mix_hard=0.65
        )
        if rx is not None:
            rpred, _, _ = forest.forward_forest(rx, top_k=top_k)
            loss_replay = mse(rpred, ry)

    # anchor
    loss_anchor = torch.tensor(0.0, device=DEVICE)
    if anchor_ratio > 0:
        ax, ay = forest.anchors.sample(
            batch_size=max(8, int(len(x_batch) * anchor_ratio))
        )
        if ax is not None:
            apred, _, _ = forest.forward_forest(ax, top_k=top_k)
            loss_anchor = mse(apred, ay)

    # distillation
    loss_distill = torch.tensor(0.0, device=DEVICE)
    if forest.teacher_snapshot is not None and distill_weight > 0:
        with torch.no_grad():
            teacher_y = forest.teacher_snapshot(x_batch, top_k=top_k)
        loss_distill = mse(y_pred, teacher_y)

    # Litter absorption — young trees absorb feature litter
    loss_litter = torch.tensor(0.0, device=DEVICE)
    young_trees = [t for t in forest.trees if t.age < 20]
    if young_trees:
        litter_features = forest.mulch.sample_features(batch_size=len(x_batch))
        if litter_features is not None:
            litter_features = litter_features.to(DEVICE)
            count = 0
            for young_tree in young_trees:
                young_feat = young_tree.trunk(x_batch[: len(litter_features)])
                min_batch = min(young_feat.shape[0], litter_features.shape[0])
                yf = young_feat[:min_batch]
                lf = litter_features[:min_batch]
                if yf.shape[1] == lf.shape[1]:
                    loss_litter = loss_litter + F.mse_loss(
                        yf, lf.detach()
                    )
                    count += 1
            if count > 0:
                loss_litter = loss_litter / count

    # router balance loss — penalises unequal expert utilisation
    loss_balance = forest.router.balance_coeff * forest.router.balance_loss(weights)
    forest.router.update_utilization(weights.detach())

    total_loss = (
        loss_current
        + 0.7 * loss_replay
        + 0.6 * loss_anchor
        + distill_weight * loss_distill
        + 0.3 * loss_litter
        + loss_balance
    )
    total_loss.backward()

    forest.apply_bark_gradient_mask()
    optimizer.step()
    forest.update_ages()

    steward.step(step_idx, float(loss_current.item()), x_batch)

    return {
        "loss_current": float(loss_current.item()),
        "loss_replay": float(loss_replay.item()),
        "loss_anchor": float(loss_anchor.item()),
        "loss_distill": float(loss_distill.item()),
        "loss_litter": float(loss_litter.item()),
        "loss_balance": float(loss_balance.item()),
        "loss_total": float(total_loss.item()),
        "trees": forest.num_trees(),
        "fitness_mean": float(
            sum(t.fitness for t in forest.trees) / forest.num_trees()
        ),
    }


# ----------------------------
# 7) Optimizer rebuild with best-effort state transfer
# ----------------------------
def rebuild_optimizer_preserve_state(old_opt, new_params, lr=0.03):
    new_opt = optim.Adam(new_params, lr=lr)
    if old_opt is None:
        return new_opt
    old_state = old_opt.state
    for group in new_opt.param_groups:
        for p in group["params"]:
            if p in old_state:
                new_opt.state[p] = old_state[p]
    return new_opt


# ----------------------------
# 8) Visualization
# ----------------------------
@torch.no_grad()
def visualize(forest: ForestEcosystem, X, Y_true, step, stats):
    forest.eval()
    plt.clf()

    plt.subplot(1, 2, 1)
    G = forest.graph
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, seed=42)
        colors, sizes, labels = [], [], {}
        for t in forest.trees:
            dark = min(1.0, t.bark)
            colors.append((0, 1.0 - 0.85 * dark, 0))
            sizes.append(120 + 25 * min(20.0, t.fitness))
            labels[t.id] = f"{t.id}"
        nx.draw(
            G,
            pos,
            node_color=colors,
            node_size=sizes,
            with_labels=True,
            labels=labels,
            font_color="white",
        )
    plt.title(f"Root network | trees={forest.num_trees()} | step={step}")

    plt.subplot(1, 2, 2)
    Xp = X.to(DEVICE)
    yp, _, _ = forest.forward_forest(Xp, top_k=3)
    plt.plot(X.cpu().numpy(), Y_true.cpu().numpy(), "k--", alpha=0.45, label="True")
    plt.plot(X.cpu().numpy(), yp.cpu().numpy(), "g-", linewidth=2.0, label="Forest")
    plt.grid(True, alpha=0.3)
    plt.legend()

    title = (
        f"loss={stats['loss_current']:.4f}  total={stats['loss_total']:.4f}\n"
        f"replay={stats['loss_replay']:.4f}  anchor={stats['loss_anchor']:.4f}  distill={stats['loss_distill']:.4f}\n"
        f"fitness_mean={stats['fitness_mean']:.2f}"
    )
    plt.title(title)
    plt.tight_layout()
    plt.pause(0.01)


# ----------------------------
# 9) Demo loop (nonstationary stream)
# ----------------------------
def target_function(x, t):
    amp = 1.0 + 0.4 * math.sin(t * 0.03)
    phase = 0.5 * math.sin(t * 0.015)
    growth = torch.exp(0.08 * x) * (0.9 + 0.2 * math.sin(t * 0.02))
    return amp * torch.sin(x + phase) / growth


if __name__ == "__main__":
    set_seed(7)
    N = 240
    X = torch.linspace(0, 10, N).reshape(-1, 1)
    X_plot = torch.linspace(0, 10, 250).reshape(-1, 1)

    forest = ForestEcosystem(input_dim=1, hidden_dim=32, max_trees=24).to(DEVICE)
    steward = Steward(forest)

    optimizer = optim.Adam(list(forest.parameters()), lr=0.03)
    forest.snapshot_teacher()

    plt.figure(figsize=(12, 6))

    steps = 260
    batch_size = 48

    for step in range(steps):
        start = (step * 3) % (N - batch_size)
        xb = X[start : start + batch_size]
        yb = target_function(xb, step)

        prev_param_ids = {id(p) for p in forest.parameters()}

        stats = train_step(
            forest,
            steward,
            optimizer,
            xb,
            yb,
            step_idx=step,
            top_k=3,
            replay_ratio=1.0,
            anchor_ratio=0.5,
            distill_weight=0.25,
        )

        new_param_ids = {id(p) for p in forest.parameters()}
        if new_param_ids != prev_param_ids:
            optimizer = rebuild_optimizer_preserve_state(
                optimizer, list(forest.parameters()), lr=0.03
            )

        if step % 10 == 0:
            Y_plot = target_function(X_plot, step).cpu()
            visualize(forest, X_plot.cpu(), Y_plot, step, stats)

    plt.show()
    print("Done.")
