"""
Real Neural Architecture Search (NAS) for NeuralForest Trees — Per-Tree (B-mode)

NAS = Neural Architecture Search:
Automatically discovers good neural network architectures by SEARCHING a space
of designs (layers/width/activation/norm/dropout/residual) and EVALUATING them
by training on real data, rather than using a hand-written heuristic.

This version is designed for your NeuralForest v2.1 where:
- Each TreeExpert has its own architecture (TreeArch stored per tree).
- ForestEcosystem._plant_tree(arch=TreeArch(...)) is supported.
- forest.mulch.sample(...) and forest.anchors.sample(...) exist.

What this module provides:
- TreeArchitectureSearch: evolutionary NAS with real training + validation.
- It evaluates candidate architectures on real batches sampled from anchors/mulch.
- It can PLANT a new tree with the best discovered architecture (B-mode integration).
- Optional: feed discovered architectures into a steward pool (if you have one).

Key differences from your previous NAS script:
- FIXED residual (no torch.eye / runtime allocations).
- Uses real forest samplers (anchors.sample / mulch.sample) instead of iterating mulch.data.
- Removes global forest.tree_architecture setter (that was A-mode). We plant per-tree arch.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Repro / device helpers
# ----------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device_from_forest(forest) -> torch.device:
    try:
        for p in forest.parameters():
            return p.device
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


# ----------------------------
# Import canonical TreeArch / TreeExpert if available
# ----------------------------
# Your NeuralForest v2.1 defines TreeArch and TreeExpert in NeuralForest.py.
# We import them to guarantee schema compatibility with your forest.
try:
    from NeuralForest import TreeArch as CanonicalTreeArch  # type: ignore
    from NeuralForest import TreeExpert as CanonicalTreeExpert  # type: ignore
except Exception:  # pragma: no cover
    CanonicalTreeArch = None
    CanonicalTreeExpert = None


# ----------------------------
# Fallback TreeArch / TreeExpert (only used if imports fail)
# ----------------------------
@dataclass(frozen=True)
class TreeArch:
    num_layers: int
    hidden_dim: int
    activation: str
    dropout: float
    normalization: str  # "layer" | "batch" | "none"
    residual: bool

    def to_dict(self) -> Dict[str, Any]:
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
    Fallback configurable TreeExpert (used only if NeuralForest.TreeExpert import fails).
    Matches the v2.1 behavior: per-tree arch stored, residual uses learnable skip_proj.
    """

    def __init__(self, input_dim: int, tree_id: int, arch: TreeArch):
        super().__init__()
        self.id = tree_id
        self.arch = arch

        self.age = 0
        self.bark = 0.0
        self.fitness = 5.0

        in_dim = input_dim
        layers: List[nn.Module] = []

        self.use_residual = bool(arch.residual and arch.num_layers >= 2)
        self.skip_proj = None
        if self.use_residual and in_dim != arch.hidden_dim:
            self.skip_proj = nn.Linear(in_dim, arch.hidden_dim, bias=False)

        for _ in range(max(1, int(arch.num_layers))):
            layers.append(nn.Linear(in_dim, arch.hidden_dim))
            layers.append(_make_norm(arch.normalization, arch.hidden_dim))
            layers.append(_make_activation(arch.activation))
            if arch.dropout and float(arch.dropout) > 0.0:
                layers.append(nn.Dropout(float(arch.dropout)))
            in_dim = arch.hidden_dim

        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(arch.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_2d(x)
        h = self.trunk(x)
        if self.use_residual:
            skip = x if self.skip_proj is None else self.skip_proj(x)
            if skip.shape == h.shape:
                h = h + skip
        return self.head(h)


def _arch_type() -> type:
    return CanonicalTreeArch if CanonicalTreeArch is not None else TreeArch


def _tree_type() -> type:
    return CanonicalTreeExpert if CanonicalTreeExpert is not None else TreeExpert


# ----------------------------
# NAS Config
# ----------------------------
@dataclass
class NASConfig:
    generations: int = 15
    population_size: int = 14
    elite_fraction: float = 0.25
    tournament_k: int = 3
    mutation_rate: float = 0.25
    crossover_rate: float = 0.8

    # Real evaluation
    train_steps: int = 80
    val_steps: int = 25
    batch_size: int = 48
    lr: float = 2e-3
    weight_decay: float = 1e-4

    # Sampling preferences
    use_anchors_prob: float = 0.65
    mulch_mix_hard: float = 0.65

    # Early stop if no improvement
    early_stop_patience: int = 5

    # Penalize over-large models a bit
    complexity_penalty: float = 0.08

    # Make eval fairer: use fixed validation batches per search run
    fixed_val_batches: int = 6

    # Repro / cache
    seed: int = 1337
    enable_cache: bool = True

    # Device override
    device: Optional[str] = None  # "cpu" / "cuda" / None=auto


# ----------------------------
# Architecture Search
# ----------------------------
class TreeArchitectureSearch:
    """
    Evolutionary NAS for per-tree architectures.

    Returns the best arch dict and provides helpers:
    - plant_tree_with_best_arch(): plants a new tree in the forest using best arch (B-mode)
    - add_best_arch_to_pool(steward): pushes best arch into steward.arch_pool if present
    """

    def __init__(
        self,
        forest,
        *,
        search_space: Optional[Dict[str, List[Any]]] = None,
        config: Optional[NASConfig] = None,
    ):
        self.forest = forest
        self.config = config or NASConfig()

        _set_seed(self.config.seed)

        if self.config.device is None:
            self.device = _device_from_forest(forest)
        else:
            self.device = torch.device(self.config.device)

        # Defaults tailored to your v2.1 script
        self.search_space = search_space or {
            "num_layers": [1, 2, 3, 4, 5],
            "hidden_dim": [16, 32, 64, 128, 256],
            "activation": ["tanh", "relu", "gelu", "swish"],
            "dropout": [0.0, 0.1, 0.2, 0.3],
            "normalization": ["none", "layer", "batch"],
            "residual": [False, True],
        }

        self.population: List[Tuple[Any, float]] = []  # (TreeArch, fitness)
        self.hall_of_fame: List[Tuple[Any, float]] = []
        self.generation: int = 0

        self._eval_cache: Dict[str, float] = {}
        self._fixed_val: List[Tuple[torch.Tensor, torch.Tensor]] = []

    # ---------- arch ops ----------

    def random_architecture(self):
        Arch = _arch_type()
        return Arch(
            num_layers=random.choice(self.search_space["num_layers"]),
            hidden_dim=random.choice(self.search_space["hidden_dim"]),
            activation=random.choice(self.search_space["activation"]),
            dropout=random.choice(self.search_space["dropout"]),
            normalization=random.choice(self.search_space["normalization"]),
            residual=random.choice(self.search_space["residual"]),
        )

    def mutate(self, arch, mutation_rate: Optional[float] = None):
        mr = self.config.mutation_rate if mutation_rate is None else mutation_rate
        d = dict(arch.to_dict() if hasattr(arch, "to_dict") else asdict(arch))
        for k in list(d.keys()):
            if random.random() < mr:
                d[k] = random.choice(self.search_space[k])
        Arch = _arch_type()
        return Arch(**d)

    def crossover(self, a, b):
        da = dict(a.to_dict() if hasattr(a, "to_dict") else asdict(a))
        db = dict(b.to_dict() if hasattr(b, "to_dict") else asdict(b))
        child = {k: random.choice([da[k], db[k]]) for k in da.keys()}
        Arch = _arch_type()
        return Arch(**child)

    def _arch_key(self, arch) -> str:
        d = dict(arch.to_dict() if hasattr(arch, "to_dict") else asdict(arch))
        return json.dumps(d, sort_keys=True)

    # ---------- data sampling (uses your real samplers) ----------

    def _sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prefer anchors if available
        use_anchors = (
            hasattr(self.forest, "anchors")
            and len(self.forest.anchors) > 0
            and (random.random() < self.config.use_anchors_prob)
        )

        if use_anchors:
            bx, by = self.forest.anchors.sample(batch_size=batch_size)
            if bx is not None and by is not None:
                bx = _ensure_2d(bx.to(self.device))
                by = _ensure_2d(by.to(self.device))
                return bx, by

        # Fallback to prioritized mulch sampler
        if not hasattr(self.forest, "mulch") or len(self.forest.mulch) == 0:
            raise RuntimeError("No data: forest.anchors empty and forest.mulch empty.")

        bx, by = self.forest.mulch.sample(
            batch_size=batch_size, mix_hard=self.config.mulch_mix_hard
        )
        if bx is None or by is None:
            raise RuntimeError("Not enough mulch data to sample a batch.")
        bx = _ensure_2d(bx.to(self.device))
        by = _ensure_2d(by.to(self.device))
        return bx, by

    def _build_fixed_validation(self) -> None:
        self._fixed_val = []
        for _ in range(int(self.config.fixed_val_batches)):
            x, y = self._sample_batch(self.config.batch_size)
            self._fixed_val.append((x.detach(), y.detach()))

    # ---------- evaluation (real training + validation) ----------

    def evaluate_architecture(self, arch) -> float:
        key = self._arch_key(arch)
        if self.config.enable_cache and key in self._eval_cache:
            return self._eval_cache[key]

        Tree = _tree_type()
        # v2.1 forest has input_dim attribute
        input_dim = int(getattr(self.forest, "input_dim", 1))

        model = Tree(input_dim, tree_id=-1, arch=arch).to(
            self.device
        )  # expects (input_dim, tree_id, arch)
        model.train()

        opt = torch.optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        # Train
        for _ in range(int(self.config.train_steps)):
            x, y = self._sample_batch(self.config.batch_size)
            pred = model(x)
            pred = _ensure_2d(pred)
            y = _ensure_2d(y)
            if y.shape[-1] != pred.shape[-1] and pred.shape[-1] == 1:
                y = y[:, :1]
            loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Validate on fixed validation batches (fairer)
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in self._fixed_val:
                pred = model(x)
                pred = _ensure_2d(pred)
                yy = _ensure_2d(y)
                if yy.shape[-1] != pred.shape[-1] and pred.shape[-1] == 1:
                    yy = yy[:, :1]
                losses.append(float(F.mse_loss(pred, yy).item()))

        val_loss = float(sum(losses) / max(1, len(losses)))

        # complexity penalty
        params = sum(p.numel() for p in model.parameters())
        complexity = math.log10(max(10.0, float(params)))
        fitness = -val_loss - float(self.config.complexity_penalty) * complexity

        if self.config.enable_cache:
            self._eval_cache[key] = fitness
        return fitness

    # ---------- selection ----------

    def _tournament_select(self, scored: List[Tuple[Any, float]]):
        k = min(int(self.config.tournament_k), len(scored))
        contestants = random.sample(scored, k)
        contestants.sort(key=lambda x: x[1], reverse=True)
        return contestants[0][0]

    # ---------- main search ----------

    def search(self) -> Dict[str, Any]:
        print(
            f"🧬 REAL NAS (per-tree) | generations={self.config.generations} pop={self.config.population_size} "
            f"device={self.device}"
        )

        self._eval_cache.clear()
        self.hall_of_fame.clear()
        self.generation = 0
        self._build_fixed_validation()

        # init
        self.population = [
            (self.random_architecture(), float("-inf"))
            for _ in range(int(self.config.population_size))
        ]

        best_seen = float("-inf")
        no_improve = 0

        for gen in range(int(self.config.generations)):
            t0 = time.time()

            # evaluate
            scored: List[Tuple[Any, float]] = []
            for arch, _ in self.population:
                fit = self.evaluate_architecture(arch)
                scored.append((arch, fit))

            scored.sort(key=lambda x: x[1], reverse=True)

            # track best
            best_arch, best_fit = scored[0]
            self.hall_of_fame.append((best_arch, best_fit))
            self.hall_of_fame.sort(key=lambda x: x[1], reverse=True)
            self.hall_of_fame = self.hall_of_fame[
                : max(8, int(self.config.population_size * 0.5))
            ]

            if best_fit > best_seen + 1e-6:
                best_seen = best_fit
                no_improve = 0
            else:
                no_improve += 1

            dt = time.time() - t0
            print(
                f"  Gen {gen+1:02d}/{self.config.generations} | best_fitness={best_fit:.6f} | time={dt:.1f}s"
            )
            print(
                f"    best_arch={best_arch.to_dict() if hasattr(best_arch, 'to_dict') else asdict(best_arch)}"
            )

            if no_improve >= int(self.config.early_stop_patience):
                print(f"⏹️ Early stop: no improvement for {no_improve} generations.")
                self.generation = gen + 1
                break

            # next gen (elitism + tournament + crossover + mutate)
            elite_n = max(
                1,
                int(
                    int(self.config.population_size) * float(self.config.elite_fraction)
                ),
            )
            elites = scored[:elite_n]

            children: List[Tuple[Any, float]] = []
            while len(children) < (int(self.config.population_size) - elite_n):
                p1 = self._tournament_select(scored)
                p2 = self._tournament_select(scored)

                if random.random() < float(self.config.crossover_rate):
                    child = self.crossover(p1, p2)
                else:
                    child = p1

                child = self.mutate(
                    child, mutation_rate=float(self.config.mutation_rate)
                )
                children.append((child, float("-inf")))

            self.population = elites + children
            self.generation = gen + 1

        best = max(self.hall_of_fame, key=lambda x: x[1])
        best_arch, best_fit = best[0], best[1]
        best_dict = (
            best_arch.to_dict() if hasattr(best_arch, "to_dict") else asdict(best_arch)
        )

        print("✅ NAS complete.")
        print(f"   Best fitness: {best_fit:.6f}")
        print(f"   Best architecture: {best_dict}")
        return best_dict

    # ---------- utilities ----------

    def get_best_architecture(self) -> Tuple[Dict[str, Any], float]:
        if not self.hall_of_fame:
            a = self.random_architecture()
            d = a.to_dict() if hasattr(a, "to_dict") else asdict(a)
            return d, float("-inf")
        best = max(self.hall_of_fame, key=lambda x: x[1])
        arch = best[0]
        d = arch.to_dict() if hasattr(arch, "to_dict") else asdict(arch)
        return d, float(best[1])

    def save_search_results(self, path: str) -> None:
        best_arch, best_fit = self.get_best_architecture()
        results = {
            "generations_completed": self.generation,
            "config": asdict(self.config),
            "search_space": self.search_space,
            "hall_of_fame": [
                {
                    "architecture": (
                        a.to_dict() if hasattr(a, "to_dict") else asdict(a)
                    ),
                    "fitness": float(f),
                }
                for a, f in self.hall_of_fame
            ],
            "best_architecture": best_arch,
            "best_fitness": best_fit,
        }
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ NAS results saved to {path}")

    # ---------- B-mode integration helpers ----------

    def plant_tree_with_best_arch(self) -> Dict[str, Any]:
        """
        B-mode: Plant a new tree with the best discovered architecture.
        Requires your ForestEcosystem._plant_tree(arch=TreeArch(...)) to exist.
        """
        best_arch_dict, best_fit = self.get_best_architecture()
        Arch = _arch_type()
        arch_obj = Arch(**best_arch_dict)
        if not hasattr(self.forest, "_plant_tree"):
            raise AttributeError("forest has no _plant_tree method")
        # plant new tree with this architecture
        self.forest._plant_tree(arch=arch_obj)
        return {"planted_architecture": best_arch_dict, "best_fitness": best_fit}

    def add_best_arch_to_pool(self, steward) -> Dict[str, Any]:
        """
        If you have a Steward with steward.arch_pool list, push best arch into it.
        """
        best_arch_dict, best_fit = self.get_best_architecture()
        Arch = _arch_type()
        arch_obj = Arch(**best_arch_dict)

        if not hasattr(steward, "arch_pool") or steward.arch_pool is None:
            raise AttributeError("steward has no arch_pool list")
        steward.arch_pool.append(arch_obj)
        return {"added_architecture": best_arch_dict, "best_fitness": best_fit}
