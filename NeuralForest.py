# NeuralForest v2.1 — single-cell full script (continual learning + per-tree NAS-ready architecture)
# Adds:
# - Per-tree configurable architectures (each tree stores its own arch)
# - Correct teacher snapshot and checkpointing for heterogeneous trees
# - Safe residual (learnable skip projection) (no torch.eye hacks)
#
# Dependencies: torch, numpy, matplotlib, networkx

import math
import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------
# 0) Utilities
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(7)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


# ----------------------------
# 1) Memory: Prioritized Replay + Coreset anchors
# ----------------------------
class PrioritizedMulch:
    """
    Stores experiences with priorities and supports weighted sampling.
    Item: (x, y, priority)
    """

    def __init__(self, capacity=8000, alpha=0.7, eps=1e-3):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.data = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def add(self, x, y, priority):
        p = float(abs(priority) + self.eps)
        self.data.append((x.detach().cpu(), y.detach().cpu(), p))

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
                x, y, _p = self.data[i]
                xs.append(x)
                ys.append(y)

        if rand_n > 0:
            batch = random.sample(self.data, rand_n)
            for x, y, _p in batch:
                xs.append(x)
                ys.append(y)

        batch_x = torch.stack(xs).to(DEVICE)
        batch_y = torch.stack(ys).to(DEVICE)
        return batch_x, batch_y


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
    def __init__(self, input_dim, max_trees, hidden=32):
        super().__init__()
        self.max_trees = max_trees
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, max_trees)
        )

    def forward(self, x, num_trees):
        scores = self.net(x)[:, :num_trees]  # [B, T]
        return scores


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
        reward = 1.0 / (float(loss_value) + 1e-4)
        self.fitness = 0.97 * self.fitness + 0.03 * reward


# ----------------------------
# 4) Forest ecosystem (per-tree arch + correct snapshot + checkpoints)
# ----------------------------
class ForestEcosystem(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, max_trees=24):
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
                for p1, p2 in zip(new_tree.parameters(), other.parameters()):
                    dist += (p1.detach() - p2.detach()).norm().item()
                if dist < best_dist:
                    best_dist = dist
                    best = other
            if best is not None:
                self.graph.add_edge(new_tree.id, best.id, weight=2.0)

        self.tree_counter += 1

    def _prune_trees(self, ids_to_remove, min_keep=2):
        if self.num_trees() <= min_keep:
            return

        keep = [t for t in self.trees if t.id not in set(ids_to_remove)]
        if len(keep) < min_keep:
            sorted_by_fit = sorted(
                list(self.trees), key=lambda t: t.fitness, reverse=True
            )
            keep = sorted_by_fit[:min_keep]

        removed_ids = {t.id for t in self.trees} - {t.id for t in keep}
        self.trees = nn.ModuleList(keep).to(DEVICE)

        for rid in removed_ids:
            if self.graph.has_node(rid):
                self.graph.remove_node(rid)

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
    def save_checkpoint(self, path):
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

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

        mulch_data = [(x, y, p) for x, y, p in self.mulch.data]
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

        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path, device=None):
        if device is None:
            device = DEVICE

        checkpoint = torch.load(path, map_location=device)

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
        for x, y, p in checkpoint["mulch_data"]:
            forest.mulch.add(x.to(device), y.to(device), p)

        forest.anchors = AnchorCoreset(capacity=checkpoint["anchor_capacity"])
        for x, y in checkpoint["anchor_data"]:
            forest.anchors.data.append((x.to(device), y.to(device)))

        for u, v, data in checkpoint["graph_edges"]:
            forest.graph.add_edge(u, v, **data)

        print(f"✅ Checkpoint loaded from {path}")
        print(
            f"   Trees: {forest.num_trees()}, Memory: {len(forest.mulch)}, Anchors: {len(forest.anchors)}"
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

        self.router = GatingRouter(
            self.input_dim, max_trees=self.max_trees, hidden=32
        ).to(DEVICE)
        self.router.load_state_dict(
            {k: v.detach().clone() for k, v in forest.router.state_dict().items()}
        )

        self.trees = nn.ModuleList()
        for t in forest.trees:
            nt = TreeExpert(self.input_dim, t.id, t.arch).to(DEVICE)
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
        for i in range(len(x_batch)):
            forest.mulch.add(x_batch[i], y_batch[i], priority=per_ex[i].item())
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

    total_loss = (
        loss_current
        + 0.7 * loss_replay
        + 0.6 * loss_anchor
        + distill_weight * loss_distill
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
N = 240
X = torch.linspace(0, 10, N).reshape(-1, 1)
X_plot = torch.linspace(0, 10, 250).reshape(-1, 1)


def target_function(x, t):
    amp = 1.0 + 0.4 * math.sin(t * 0.03)
    phase = 0.5 * math.sin(t * 0.015)
    growth = torch.exp(0.08 * x) * (0.9 + 0.2 * math.sin(t * 0.02))
    return amp * torch.sin(x + phase) / growth


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
