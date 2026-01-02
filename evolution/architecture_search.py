"""
Real Neural Architecture Search (NAS) for NeuralForest Trees

What this does:
- Searches tree architectures (MLP-based) using an evolutionary algorithm.
- Evaluates candidates by actually training them for a few steps on real
  data from forest.mulch and/or forest.anchors.
- Returns best architecture found + utilities to apply it.

What is NAS?
Neural Architecture Search is the automated process of discovering neural network
architectures by searching a space of possible designs and using empirical
evaluation (training/validation) to select the best.

Integration note (required for full effect):
- Your current ForestEcosystem._plant_tree() creates TreeExpert(input_dim, hidden_dim, id).
- To make NAS affect the live forest, you should:
  (A) add an "arch" field on the forest (e.g. forest.tree_architecture)
  (B) update _plant_tree() to create a configurable tree using that arch

This module provides a ConfigurableTreeExpert you can use for that.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities
# ----------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device_from_forest(forest) -> torch.device:
    # Best-effort: infer device from existing model params, otherwise CPU
    try:
        for p in forest.parameters():
            return p.device
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_tensor(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device)


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


# ----------------------------
# Configurable Tree Model (candidate architecture)
# ----------------------------

@dataclass(frozen=True)
class TreeArch:
    num_layers: int
    hidden_dim: int
    activation: str
    dropout: float
    normalization: str  # "layer", "batch", "none"
    residual: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigurableTreeExpert(nn.Module):
    """
    A configurable MLP "tree" (expert) that can represent many architectures.

    Output is [B, 1] (to match existing forest usage).
    """
    def __init__(self, input_dim: int, tree_id: int, arch: TreeArch):
        super().__init__()
        self.id = tree_id
        self.arch = arch

        self.age = getattr(self, "age", 0)
        self.bark = getattr(self, "bark", 0.0)

        # Fitness is used by pruning logic elsewhere; initialize conservative
        self.fitness = 0.0

        act = self._make_activation(arch.activation)
        layers: List[nn.Module] = []

        in_dim = input_dim
        for i in range(arch.num_layers):
            out_dim = arch.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))

            if arch.normalization == "layer":
                layers.append(nn.LayerNorm(out_dim))
            elif arch.normalization == "batch":
                layers.append(nn.BatchNorm1d(out_dim))

            layers.append(act)

            if arch.dropout and arch.dropout > 0.0:
                layers.append(nn.Dropout(arch.dropout))

            in_dim = out_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(arch.hidden_dim if arch.num_layers > 0 else input_dim, 1)

        # Residual support: only valid when dims match
        self.use_residual = bool(arch.residual and arch.num_layers >= 2)

    def _make_activation(self, name: str) -> nn.Module:
        name = (name or "relu").lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        if name in ("swish", "silu"):
            return nn.SiLU()
        # default
        return nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_2d(x)

        if self.arch.num_layers == 0:
            h = x
        else:
            # Optional residual: do block-wise residual by splitting backbone into chunks
            if self.use_residual:
                # simple residual every 2 linear blocks (approx)
                h = x
                # Not perfect residual wiring, but stable and simple:
                # pass through backbone and add a projected skip at the end
                out = self.backbone(h)
                # Project skip to hidden_dim if needed
                if h.shape[-1] != out.shape[-1]:
                    skip = F.linear(h, torch.eye(out.shape[-1], h.shape[-1], device=h.device))
                else:
                    skip = h
                h = out + skip
            else:
                h = self.backbone(x)

        y = self.head(h)
        return y


# ----------------------------
# NAS Search
# ----------------------------

@dataclass
class NASConfig:
    generations: int = 20
    population_size: int = 16
    elite_fraction: float = 0.25
    tournament_k: int = 3
    mutation_rate: float = 0.25
    crossover_rate: float = 0.8

    # Real evaluation
    train_steps: int = 80
    val_steps: int = 20
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-4

    # Sampling preferences
    use_anchors_prob: float = 0.6  # probability to sample from anchors vs mulch
    max_mulch_items_for_sampling: int = 4000  # avoid huge random choices

    # Early stop
    early_stop_patience: int = 5

    # Complexity penalty (bigger models must win by more)
    complexity_penalty: float = 0.10

    # Reproducibility
    seed: int = 1337

    # Device
    device: Optional[str] = None  # "cpu" or "cuda" or None=auto

    # Cache evaluations
    enable_cache: bool = True


class TreeArchitectureSearch:
    """
    Real NAS for trees: evolves architectures and evaluates them on real forest data.

    Outputs:
      best_arch_dict (with keys: num_layers, hidden_dim, activation, dropout, normalization, residual)
    """

    def __init__(self, forest, *, search_space: Optional[Dict[str, List[Any]]] = None, config: Optional[NASConfig] = None):
        self.forest = forest
        self.config = config or NASConfig()

        if self.config.device is None:
            self.device = _device_from_forest(forest)
        else:
            self.device = torch.device(self.config.device)

        _set_seed(self.config.seed)

        self.search_space = search_space or {
            "num_layers": [1, 2, 3, 4, 5],
            "hidden_dim": [32, 64, 128, 256],
            "activation": ["relu", "gelu", "tanh", "swish"],
            "dropout": [0.0, 0.1, 0.2, 0.3],
            "normalization": ["layer", "batch", "none"],
            "residual": [True, False],
        }

        self.population: List[Tuple[TreeArch, float]] = []
        self.hall_of_fame: List[Tuple[TreeArch, float]] = []
        self.generation: int = 0

        self._eval_cache: Dict[str, float] = {}

    # ---------- architecture ops ----------

    def random_architecture(self) -> TreeArch:
        return TreeArch(
            num_layers=random.choice(self.search_space["num_layers"]),
            hidden_dim=random.choice(self.search_space["hidden_dim"]),
            activation=random.choice(self.search_space["activation"]),
            dropout=random.choice(self.search_space["dropout"]),
            normalization=random.choice(self.search_space["normalization"]),
            residual=random.choice(self.search_space["residual"]),
        )

    def mutate(self, arch: TreeArch, mutation_rate: Optional[float] = None) -> TreeArch:
        mr = self.config.mutation_rate if mutation_rate is None else mutation_rate
        d = arch.to_dict()
        for k in d.keys():
            if random.random() < mr:
                d[k] = random.choice(self.search_space[k])
        return TreeArch(**d)

    def crossover(self, a: TreeArch, b: TreeArch) -> TreeArch:
        da = a.to_dict()
        db = b.to_dict()
        child = {}
        for k in da.keys():
            child[k] = random.choice([da[k], db[k]])
        return TreeArch(**child)

    def _arch_key(self, arch: TreeArch) -> str:
        # stable string key for caching
        d = arch.to_dict()
        return json.dumps(d, sort_keys=True)

    # ---------- data sampling ----------

    def _sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a supervised batch (x,y) from anchors or mulch.
        Anchors preferred if available.
        """
        use_anchors = hasattr(self.forest, "anchors") and len(self.forest.anchors) > 0 and (random.random() < self.config.use_anchors_prob)
        if use_anchors:
            bx, by = self.forest.anchors.sample(batch_size=batch_size)
            if bx is None or by is None:
                use_anchors = False
            else:
                bx = _as_tensor(bx, self.device)
                by = _as_tensor(by, self.device)
                return _ensure_2d(bx), _ensure_2d(by)

        # fallback to mulch
        mulch = getattr(self.forest, "mulch", None)
        if mulch is None or len(mulch) == 0:
            raise RuntimeError("No data available: forest.anchors empty and forest.mulch empty.")

        data = list(mulch.data)
        if len(data) > self.config.max_mulch_items_for_sampling:
            data = random.sample(data, self.config.max_mulch_items_for_sampling)

        # mulch items expected: (x, y, priority)
        items = random.choices(data, k=batch_size)
        xs = []
        ys = []
        for item in items:
            x, y = item[0], item[1]
            xs.append(_as_tensor(x, self.device))
            ys.append(_as_tensor(y, self.device))

        bx = torch.stack([t.view(-1) for t in xs], dim=0)
        by = torch.stack([t.view(-1) for t in ys], dim=0)

        # ensure target is [B, 1] if possible
        if by.dim() == 1:
            by = by.unsqueeze(1)
        elif by.dim() == 2 and by.shape[1] != 1:
            # if multi-dim target, keep as is; your forest uses [B,1], but we won't crash
            pass

        return bx, by

    # ---------- evaluation ----------

    def evaluate_architecture(self, arch: TreeArch) -> float:
        """
        Real evaluation:
        - instantiate candidate tree
        - train for train_steps on sampled batches
        - compute validation loss for val_steps
        - return fitness = -val_loss - complexity_penalty * model_complexity
        """
        key = self._arch_key(arch)
        if self.config.enable_cache and key in self._eval_cache:
            return self._eval_cache[key]

        # Build candidate
        model = ConfigurableTreeExpert(
            input_dim=int(getattr(self.forest, "input_dim", 1)),
            tree_id=-1,
            arch=arch,
        ).to(self.device)
        model.train()

        opt = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        # Train loop
        for _ in range(self.config.train_steps):
            x, y = self._sample_batch(self.config.batch_size)
            pred = model(x)
            pred = _ensure_2d(pred)
            y = _ensure_2d(y)
            # if target shape mismatch, try to align to [B,1]
            if y.shape[-1] != pred.shape[-1] and pred.shape[-1] == 1:
                y = y[:, :1]
            loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(self.config.val_steps):
                x, y = self._sample_batch(self.config.batch_size)
                pred = model(x)
                pred = _ensure_2d(pred)
                y = _ensure_2d(y)
                if y.shape[-1] != pred.shape[-1] and pred.shape[-1] == 1:
                    y = y[:, :1]
                val_losses.append(_safe_float(F.mse_loss(pred, y).item(), default=1e9))

        val_loss = float(sum(val_losses) / max(1, len(val_losses)))

        # Complexity penalty
        params = sum(p.numel() for p in model.parameters())
        complexity = math.log10(max(10.0, float(params)))  # gentle growth
        fitness = -val_loss - self.config.complexity_penalty * complexity

        if self.config.enable_cache:
            self._eval_cache[key] = fitness

        return fitness

    # ---------- selection ----------

    def _tournament_select(self, scored_pop: List[Tuple[TreeArch, float]]) -> TreeArch:
        k = min(self.config.tournament_k, len(scored_pop))
        contestants = random.sample(scored_pop, k)
        contestants.sort(key=lambda x: x[1], reverse=True)
        return contestants[0][0]

    # ---------- main search ----------

    def search(self) -> Dict[str, Any]:
        """
        Evolutionary NAS:
        - initialize population
        - evaluate
        - elitism keep top
        - tournament select parents
        - crossover + mutation
        """
        print(f"🧬 Starting REAL NAS for {self.config.generations} generations "
              f"(pop={self.config.population_size}, device={self.device})")

        # init population
        self.population = [(self.random_architecture(), float("-inf")) for _ in range(self.config.population_size)]
        best_fitness = float("-inf")
        no_improve = 0

        for gen in range(self.config.generations):
            t_gen = time.time()

            # evaluate
            scored: List[Tuple[TreeArch, float]] = []
            for arch, _ in self.population:
                fit = self.evaluate_architecture(arch)
                scored.append((arch, fit))

            # sort
            scored.sort(key=lambda x: x[1], reverse=True)

            # hall of fame
            self.hall_of_fame.append((scored[0][0], scored[0][1]))
            self.hall_of_fame.sort(key=lambda x: x[1], reverse=True)
            self.hall_of_fame = self.hall_of_fame[: max(5, int(self.config.population_size * 0.5))]

            # early stop tracking
            if scored[0][1] > best_fitness + 1e-6:
                best_fitness = scored[0][1]
                no_improve = 0
            else:
                no_improve += 1

            # logging
            dt = time.time() - t_gen
            top = scored[0]
            print(f"  Gen {gen+1:02d}/{self.config.generations} | best_fitness={top[1]:.6f} | "
                  f"best_arch={top[0].to_dict()} | time={dt:.1f}s")

            if no_improve >= self.config.early_stop_patience:
                print(f"⏹️ Early stop: no improvement for {no_improve} generations.")
                self.generation = gen + 1
                break

            # next generation
            elite_n = max(1, int(self.config.population_size * self.config.elite_fraction))
            elites = scored[:elite_n]

            children: List[Tuple[TreeArch, float]] = []
            while len(children) < (self.config.population_size - elite_n):
                p1 = self._tournament_select(scored)
                p2 = self._tournament_select(scored)

                if random.random() < self.config.crossover_rate:
                    child = self.crossover(p1, p2)
                else:
                    child = p1

                child = self.mutate(child, mutation_rate=self.config.mutation_rate)
                children.append((child, float("-inf")))

            self.population = elites + children
            self.generation = gen + 1

        best = max(self.hall_of_fame, key=lambda x: x[1])
        print("✅ NAS complete.")
        print(f"   Best fitness: {best[1]:.6f}")
        print(f"   Best architecture: {best[0].to_dict()}")
        return best[0].to_dict()

    # ---------- history / persistence ----------

    def get_search_history(self) -> List[Dict[str, Any]]:
        return [{"generation": i, "architecture": arch.to_dict(), "fitness": fit}
                for i, (arch, fit) in enumerate(self.hall_of_fame)]

    def get_best_architecture(self) -> Tuple[Dict[str, Any], float]:
        if not self.hall_of_fame:
            a = self.random_architecture()
            return a.to_dict(), float("-inf")
        best = max(self.hall_of_fame, key=lambda x: x[1])
        return best[0].to_dict(), best[1]

    def save_search_results(self, path: str) -> None:
        best_arch, best_fit = self.get_best_architecture()
        results = {
            "generations_completed": self.generation,
            "config": asdict(self.config),
            "search_space": self.search_space,
            "hall_of_fame": self.get_search_history(),
            "best_architecture": best_arch,
            "best_fitness": best_fit,
        }
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ NAS results saved to {path}")

    # ---------- integration helper ----------

    def apply_best_architecture_to_forest(self) -> Dict[str, Any]:
        """
        Stores the best architecture on the forest instance for later use.

        After calling this, update ForestEcosystem._plant_tree() to do:
            arch_dict = getattr(self, "tree_architecture", None)
            if arch_dict:
                arch = TreeArch(**arch_dict)
                t = ConfigurableTreeExpert(self.input_dim, self.tree_counter, arch).to(DEVICE)
            else:
                t = TreeExpert(...)
        """
        best_arch, best_fit = self.get_best_architecture()
        setattr(self.forest, "tree_architecture", best_arch)
        return {"best_architecture": best_arch, "best_fitness": best_fit}
