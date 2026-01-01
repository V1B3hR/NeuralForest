# NeuralForest v2 — single-cell full script (continual learning + self-improvement ecosystem)
# Includes:
# - Prioritized Replay (weighted sampling, no full sort)
# - Coreset "anchor" memory (representative old skills)
# - Gating / routing (top-k trees per input)
# - Distillation (LwF-style) to reduce forgetting
# - Drift detection (simple) to trigger growth/plasticity
# - Fitness on mixed eval set (stabilized)
# - Pruning with safety (keep minimum trees)
# - Optimizer state preservation across grow/prune (best-effort)
#
# Dependencies: torch, numpy, matplotlib, networkx
# Works best in a notebook; visualization uses matplotlib (interactive).

import math
import random
from collections import deque, defaultdict

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


# ----------------------------
# 1) Memory: Prioritized Replay + Coreset anchors
# ----------------------------
class PrioritizedMulch:
    """
    Stores experiences with priorities and supports weighted sampling.
    Simple PER (not SumTree), but avoids sorting entire memory each step.
    We maintain a deque and sample via torch.multinomial on priorities.

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
        # priority should be >=0; store as float
        p = float(abs(priority) + self.eps)
        self.data.append((x.detach().cpu(), y.detach().cpu(), p))

    def sample(self, batch_size, mix_hard=0.6):
        """
        Returns a batch composed of:
        - hard: weighted by priority
        - random: uniformly random
        """
        n = len(self.data)
        if n < batch_size:
            return None, None

        hard_n = int(batch_size * mix_hard)
        rand_n = batch_size - hard_n

        xs, ys = [], []

        # Hard sampling
        if hard_n > 0:
            priorities = torch.tensor([item[2] for item in self.data], dtype=torch.float32)
            probs = priorities.pow(self.alpha)
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, num_samples=hard_n, replacement=(hard_n > n))
            for i in idx.tolist():
                x, y, _p = self.data[i]
                xs.append(x)
                ys.append(y)

        # Random sampling
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
    Keeps a small representative set of "anchors" (core skills).
    We do a cheap online farthest-point heuristic in feature space (x only).
    For 1D signals it's enough; for higher dims you can extend to embeddings.
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

        # Replace the anchor that is "most redundant" w.r.t. this x:
        # find closest anchor; if this x is far from its closest anchor, it is valuable.
        xs = torch.stack([item[0] for item in self.data])
        # distance to closest anchor
        dists = (xs - x).view(len(xs), -1).pow(2).mean(dim=1)
        min_dist_new = dists.min().item()

        # Compute redundancy score for each existing anchor (average distance to others)
        # To keep O(N), approximate: use distance to mean anchor.
        mean_x = xs.mean(dim=0, keepdim=True)
        redund = (xs - mean_x).view(len(xs), -1).pow(2).mean(dim=1)  # larger means less redundant

        # Replace a very redundant point if new is more novel than it
        # pick the most redundant (smallest redund)
        replace_idx = int(torch.argmin(redund).item())

        # only replace if new point is sufficiently novel
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
# 2) Trees (experts) + Gating (router)
# ----------------------------
class TreeExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, tree_id):
        super().__init__()
        self.id = tree_id
        self.age = 0
        self.bark = 0.0  # plasticity shield
        self.fitness = 5.0

        self.trunk = nn.Linear(input_dim, hidden_dim)
        self.crown = nn.Linear(hidden_dim, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.crown(self.act(self.trunk(x)))

    def step_age(self):
        self.age += 1
        if self.age > 80:
            self.bark = min(0.985, self.bark + 0.01)

    def update_fitness(self, loss_value):
        # stable EMA reward: lower loss -> higher reward
        reward = 1.0 / (float(loss_value) + 1e-4)
        self.fitness = 0.97 * self.fitness + 0.03 * reward


class GatingRouter(nn.Module):
    """
    Produces scores per tree given input x.
    We then pick top-k trees per sample and aggregate their outputs.
    """
    def __init__(self, input_dim, max_trees, hidden=32):
        super().__init__()
        self.max_trees = max_trees
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, max_trees)
        )

    def forward(self, x, num_trees):
        # x: [B, input_dim]
        scores = self.net(x)[:, :num_trees]  # [B, T]
        return scores


def topk_softmax(scores, k):
    """
    scores: [B, T]
    returns weights: [B, T] with non-topk set to 0
    """
    B, T = scores.shape
    k = min(k, T)
    topv, topi = torch.topk(scores, k=k, dim=1)
    w = torch.softmax(topv, dim=1)  # [B, k]
    weights = torch.zeros_like(scores)
    weights.scatter_(1, topi, w)
    return weights


# ----------------------------
# 3) The Ecosystem Manager (Forest) + Steward (meta-controller)
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
        self._plant_tree()  # start with one tree

        # For distillation: snapshot ("old forest") will be copied periodically
        self.teacher_snapshot = None

    def num_trees(self):
        return len(self.trees)

    def _plant_tree(self):
        if self.num_trees() >= self.max_trees:
            return
        t = TreeExpert(self.input_dim, self.hidden_dim, self.tree_counter).to(DEVICE)
        self.trees.append(t)
        self.graph.add_node(t.id)

        # Connect to most similar existing tree (param-distance heuristic)
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
        # Keep at least min_keep
        if self.num_trees() <= min_keep:
            return

        keep = [t for t in self.trees if t.id not in set(ids_to_remove)]
        if len(keep) < min_keep:
            # keep the best by fitness
            sorted_by_fit = sorted(list(self.trees), key=lambda t: t.fitness, reverse=True)
            keep = sorted_by_fit[:min_keep]

        removed_ids = {t.id for t in self.trees} - {t.id for t in keep}
        self.trees = nn.ModuleList(keep).to(DEVICE)

        for rid in removed_ids:
            if self.graph.has_node(rid):
                self.graph.remove_node(rid)

    @torch.no_grad()
    def snapshot_teacher(self):
        # Deep-copy a "teacher" model for LwF distillation
        teacher = ForestTeacher(self).to(DEVICE)
        teacher.eval()
        self.teacher_snapshot = teacher

    def forward_forest(self, x, top_k=3):
        """
        Returns:
        - y_pred: [B, 1]
        - weights: [B, T]
        - per_tree_out: list of [B,1]
        """
        T = self.num_trees()
        scores = self.router(x, num_trees=T)           # [B, T]
        weights = topk_softmax(scores, k=top_k)        # [B, T]

        outs = []
        for t in self.trees:
            outs.append(t(x))  # [B,1]

        # Stack: [B, T, 1]
        out_stack = torch.stack(outs, dim=1)
        # weights: [B, T] -> [B, T, 1]
        w = weights.unsqueeze(-1)
        y = (out_stack * w).sum(dim=1)  # [B,1]
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


class ForestTeacher(nn.Module):
    """
    Snapshot teacher: identical forward behavior but frozen.
    Uses the same routing network and tree weights copied at snapshot time.
    """
    def __init__(self, forest: ForestEcosystem):
        super().__init__()
        self.input_dim = forest.input_dim
        self.hidden_dim = forest.hidden_dim
        self.max_trees = forest.max_trees

        # Copy router
        self.router = GatingRouter(self.input_dim, max_trees=self.max_trees, hidden=32).to(DEVICE)
        self.router.load_state_dict({k: v.detach().clone() for k, v in forest.router.state_dict().items()})

        # Copy trees
        self.trees = nn.ModuleList()
        for t in forest.trees:
            nt = TreeExpert(self.input_dim, self.hidden_dim, t.id).to(DEVICE)
            nt.load_state_dict({k: v.detach().clone() for k, v in t.state_dict().items()})
            nt.eval()
            self.trees.append(nt)

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x, top_k=3):
        T = len(self.trees)
        scores = self.router(x)[:, :T]
        weights = topk_softmax(scores, k=min(top_k, T))
        outs = [t(x) for t in self.trees]  # [B,1]
        out_stack = torch.stack(outs, dim=1)  # [B,T,1]
        y = (out_stack * weights.unsqueeze(-1)).sum(dim=1)
        return y


class Steward:
    """
    Meta-controller:
    - monitors loss trend + drift
    - decides on planting/pruning
    - triggers teacher snapshots (sleep/consolidation)
    """
    def __init__(self, forest: ForestEcosystem):
        self.forest = forest
        self.loss_hist = deque(maxlen=40)
        self.drift_hist = deque(maxlen=40)

        self.last_teacher_snapshot_step = 0

    def compute_drift(self, x_batch):
        # simple drift proxy: change in mean/var of inputs over time
        x = x_batch.detach()
        mu = x.mean().item()
        var = x.var(unbiased=False).item()
        score = abs(mu) + 0.5 * abs(var - 1.0)
        return score

    def step(self, step_idx, loss_value, x_batch):
        self.loss_hist.append(float(loss_value))
        drift = self.compute_drift(x_batch)
        self.drift_hist.append(float(drift))

        loss_avg = sum(self.loss_hist) / max(1, len(self.loss_hist))
        drift_avg = sum(self.drift_hist) / max(1, len(self.drift_hist))

        # 1) Plant new tree if struggling or drift is high
        if (loss_avg > 0.06 or drift_avg > 1.2) and self.forest.num_trees() < self.forest.max_trees:
            if random.random() < 0.25:
                self.forest._plant_tree()

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
            # snapshot more often when stable
            if loss_avg < 0.08 or random.random() < 0.2:
                self.forest.snapshot_teacher()
                self.last_teacher_snapshot_step = step_idx


# ----------------------------
# 4) Training step with losses: current + replay + anchors + distillation
# ----------------------------
def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

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

    # -------- current forward --------
    optimizer.zero_grad(set_to_none=True)
    y_pred, weights, per_tree = forest.forward_forest(x_batch, top_k=top_k)

    loss_current = mse(y_pred, y_batch)

    # -------- per-tree fitness update (on current) --------
    with torch.no_grad():
        for t, out in zip(forest.trees, per_tree):
            local = mse(out, y_batch).item()
            t.update_fitness(local)

    # -------- store experiences in mulch + anchors --------
    with torch.no_grad():
        # per-example priority
        per_ex = (y_pred - y_batch).pow(2).view(len(x_batch), -1).mean(dim=1)  # [B]
        for i in range(len(x_batch)):
            forest.mulch.add(x_batch[i], y_batch[i], priority=per_ex[i].item())
            forest.anchors.add(x_batch[i], y_batch[i])

    # -------- replay batch --------
    loss_replay = torch.tensor(0.0, device=DEVICE)
    if replay_ratio > 0:
        rx, ry = forest.mulch.sample(batch_size=int(len(x_batch) * replay_ratio), mix_hard=0.65)
        if rx is not None:
            rpred, _, _ = forest.forward_forest(rx, top_k=top_k)
            loss_replay = mse(rpred, ry)

    # -------- anchor batch (core skills) --------
    loss_anchor = torch.tensor(0.0, device=DEVICE)
    if anchor_ratio > 0:
        ax, ay = forest.anchors.sample(batch_size=max(8, int(len(x_batch) * anchor_ratio)))
        if ax is not None:
            apred, _, _ = forest.forward_forest(ax, top_k=top_k)
            loss_anchor = mse(apred, ay)

    # -------- distillation to reduce forgetting --------
    loss_distill = torch.tensor(0.0, device=DEVICE)
    if forest.teacher_snapshot is not None and distill_weight > 0:
        with torch.no_grad():
            teacher_y = forest.teacher_snapshot(x_batch, top_k=top_k)
        # LwF-style: keep outputs close to teacher
        loss_distill = mse(y_pred, teacher_y)

    total_loss = loss_current + 0.7 * loss_replay + 0.6 * loss_anchor + distill_weight * loss_distill
    total_loss.backward()

    # Bark mask (reduced plasticity for old trees)
    forest.apply_bark_gradient_mask()

    optimizer.step()
    forest.update_ages()

    # Meta-controller decisions
    steward.step(step_idx, float(loss_current.item()), x_batch)

    return {
        "loss_current": float(loss_current.item()),
        "loss_replay": float(loss_replay.item()),
        "loss_anchor": float(loss_anchor.item()),
        "loss_distill": float(loss_distill.item()),
        "loss_total": float(total_loss.item()),
        "trees": forest.num_trees(),
        "fitness_mean": float(sum(t.fitness for t in forest.trees) / forest.num_trees()),
    }


# ----------------------------
# 5) Optimizer rebuild with best-effort state transfer
# ----------------------------
def rebuild_optimizer_preserve_state(old_opt, new_params, lr=0.03):
    """
    Best-effort: keeps state for parameters that still exist (same object identity).
    New params start fresh.
    """
    new_opt = optim.Adam(new_params, lr=lr)

    if old_opt is None:
        return new_opt

    # map old state by param object
    old_state = old_opt.state
    for group in new_opt.param_groups:
        for p in group["params"]:
            if p in old_state:
                new_opt.state[p] = old_state[p]
    return new_opt


# ----------------------------
# 6) Visualization
# ----------------------------
@torch.no_grad()
def visualize(forest: ForestEcosystem, X, Y_true, step, stats):
    forest.eval()
    plt.clf()

    # (A) Graph
    plt.subplot(1, 2, 1)
    G = forest.graph
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, seed=42)
        colors = []
        sizes = []
        labels = {}
        for t in forest.trees:
            dark = min(1.0, t.bark)
            colors.append((0, 1.0 - 0.85 * dark, 0))
            sizes.append(120 + 25 * min(20.0, t.fitness))
            labels[t.id] = str(t.id)
        nx.draw(G, pos, node_color=colors, node_size=sizes, with_labels=True, labels=labels, font_color="white")
    plt.title(f"Root network | trees={forest.num_trees()} | step={step}")

    # (B) Function approximation
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
# 7) Demo: Continual stream (nonstationary target) + training loop
# ----------------------------
# Stream: x in [0, 10], target slowly changes over time (concept drift)
N = 240
X = torch.linspace(0, 10, N).reshape(-1, 1)
X_plot = torch.linspace(0, 10, 250).reshape(-1, 1)

def target_function(x, t):
    # nonstationary: amplitude and phase drift
    amp = 1.0 + 0.4 * math.sin(t * 0.03)
    phase = 0.5 * math.sin(t * 0.015)
    growth = torch.exp(0.08 * x) * (0.9 + 0.2 * math.sin(t * 0.02))
    return amp * torch.sin(x + phase) / growth  # stays mostly bounded

# Initialize forest
forest = ForestEcosystem(input_dim=1, hidden_dim=32, max_trees=24).to(DEVICE)
steward = Steward(forest)

optimizer = optim.Adam(list(forest.parameters()), lr=0.03)
forest.snapshot_teacher()  # initial teacher

plt.figure(figsize=(12, 6))

steps = 260
batch_size = 48

for step in range(steps):
    # streaming window moves
    start = (step * 3) % (N - batch_size)
    xb = X[start:start+batch_size]
    yb = target_function(xb, step)

    # Train
    prev_param_ids = {id(p) for p in forest.parameters()}
    stats = train_step(
        forest, steward, optimizer,
        xb, yb,
        step_idx=step,
        top_k=3,
        replay_ratio=1.0,
        anchor_ratio=0.5,
        distill_weight=0.25
    )

    # If forest structure changed (grow/prune), rebuild optimizer preserving state
    new_param_ids = {id(p) for p in forest.parameters()}
    if new_param_ids != prev_param_ids:
        optimizer = rebuild_optimizer_preserve_state(optimizer, list(forest.parameters()), lr=0.03)

    # Visualize
    if step % 10 == 0:
        Y_plot = target_function(X_plot, step).cpu()
        visualize(forest, X_plot.cpu(), Y_plot, step, stats)

plt.show()

print("Done.")
