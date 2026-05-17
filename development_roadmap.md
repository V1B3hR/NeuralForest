# 🌲 NeuralForest — Development Roadmap

> "A healthy forest is not measured by the tallest tree, but by the resilience of the whole."

A technical, developer-facing roadmap that captures known weaknesses, planned fixes, and the true growth path forward.

---

## 🪲 Known Bugs (Fix First)

| Bug ID | File | Severity | Description | Status |
|---|---|---|---|---|
| BUG 1 | `training_demos/cifar10_full_training.py` | Critical | `TreeNet.__init__` loop body indentation made file unparseable. | ✅ Fixed |
| BUG 2 | `training_demos/cifar10_full_training.py` | Critical | `task_head_input_dim` was static while tree count changed via pruning/planting. | ✅ Fixed |
| BUG 3 | `training_demos/cifar10_full_training.py` | Medium | `PrioritizedMulch.add` used O(n) `pop(0)` on list. | ✅ Fixed |
| BUG 4 | `NeuralForest.py` | High | Module-level side effects (`set_seed`, global tensors) ran on import. | ✅ Fixed |
| BUG 5 | `NeuralForest.py` | High | `save_checkpoint` failed for paths without directory component. | ✅ Fixed |
| BUG 6 | `NeuralForest.py` | Medium | `torch.load` lacked explicit `weights_only` argument. | ✅ Fixed |
| BUG 7 | `NeuralForest.py` | High | `ForestTeacher` hardcoded global `DEVICE` instead of forest device. | ✅ Fixed |
| BUG 8 | `NeuralForest.py` | Critical | Fitness update formula decayed fitness early and broke pruning signal. | ✅ Fixed |
| BUG 9 | `NeuralForest.py` | High | `litter_loss` shape gate was too strict and silently skipped often. | ✅ Fixed |
| BUG 10 | `ecosystem_simulation.py` | High | Optimizer was not rebuilt after pruning removed parameters. | ✅ Fixed |
| BUG 11 | `evolution/tree_graveyard.py` | Medium | Hardcoded `/tmp` paths were not cross-platform. | ✅ Fixed |

---

## 🪨 Structural Weaknesses (Architectural Debt)

### 1) 🔇 The Silent Router
- **Description:** `GatingRouter` routes without explicit load-balancing pressure.
- **Impact:** dominant trees monopolize traffic while specialists starve (expert collapse).
- **Proposed solution:** add router balance objective (`KL(p_router || uniform)` or entropy floor) and monitor expert utilization histograms.

### 2) 🌱 The Rootless Sapling
- **Description:** new trees are born from random `TreeArch` + random weights.
- **Impact:** high variance starts, delayed contribution, fragile early fitness.
- **Proposed solution:** seed bank from top graveyard trees (architecture + optional weight templates).

### 3) 🍃 The Isolated Canopy
- **Description:** tree communication is mostly conceptual; no robust hidden-state passing graph.
- **Impact:** weak collaborative specialization; forest behaves like loosely grouped independent trees.
- **Proposed solution:** implement true message passing in `mycelium/knowledge_transfer.py` over `ForestEcosystem.graph`.

### 4) �� The Duplicate Forest
- **Description:** training demos duplicate core classes (`ForestEcosystem`, `TreeArch`, `PrioritizedMulch`).
- **Impact:** bug fixes diverge and behavior drifts between demo and core engine.
- **Proposed solution:** collapse demo duplicates into imports from canonical core modules.

### 5) ❄️ The Frozen Fitness
- **Description:** historical fitness shaping under-rewarded normal early losses.
- **Impact:** weak pruning/select signals and premature culling.
- **Proposed solution:** keep bounded, scaled reward now; then upgrade to baseline-relative (EMA-improvement) fitness.

### 6) 🪦 The Forgetful Graveyard
- **Description:** `save_weights=False` by default means many resurrected trees lose learned weights.
- **Impact:** resurrection restores form, not memory.
- **Proposed solution:** default weight persistence (configurable) with robust storage cleanup policy.

### 7) 🍂 The Disconnected Seasons
- **Description:** `SeasonalCycle` exists but is not wired into core training control.
- **Impact:** season logic remains decorative instead of operational.
- **Proposed solution:** route season state into `Steward.step()` and training hyperparameters.

### 8) 📏 The Dim Task Head
- **Description:** task-head dimensionality risk appears when forest topology changes dynamically.
- **Impact:** potential feature/interface drift and gradient corruption.
- **Proposed solution:** always rebuild or adapt task head when structural tree count changes.

---

## 🌱 Growth Directions (True North)

```text
        🌳 Canopy Intelligence
             /        \
      🍄 Mycelium    🧠 Consciousness
           |             |
      🌡️ Seasons ---- 🧬 Genetics
             \        /
              🌍 Biome Transfer
```

### 🍄 Phase A: Mycelium Awakening (Tree Communication)
Implement hidden-state message passing across `ForestEcosystem.graph` edges. Neighbors exchange features; receivers apply learned weighted aggregation.

### 🧬 Phase B: Sexual Reproduction (Genetic Crossover)
Generate offspring from two strong parent trees with mixed `TreeArch` traits and warm-started blended weights.

### 🌡️ Phase C: Seasons Activation
Wire `SeasonalCycle` into `Steward` and train-step controls:
- **spring:** high growth, low prune pressure
- **summer:** peak learning rate and balanced replay
- **autumn:** stronger pruning + distillation
- **winter:** no planting, strong replay + anchors

### 🌍 Phase D: Biome Transfer (Cross-domain Forest)
Serialize full forests as reusable biomes; transplant into new domains and let fitness+bark select survivors.

### 🔭 Phase E: Canopy Attention (Global Forest Self-Attention)
Replace simple weighted sum in `forward_forest` with compact self-attention across tree outputs.

### 🧠 Phase F: Consciousness Integration
Invoke `ForestConsciousness.reflect_and_plan()` every N epochs in steward-driven training.

---

## 📊 Technical Debt Map

| Component | Debt Level | Notes | Priority |
|---|---|---|---|
| Routing (`GatingRouter`) | High | No expert balance loss, collapse risk | P0 |
| Demo/Core parity | High | Duplicated ecosystem classes | P0 |
| Fitness shaping | High | Needs baseline-relative stability | P0 |
| Mycelium communication | High | Placeholder-level coupling | P1 |
| Seasonal integration | Medium | Season engine not driving training | P1 |
| Graveyard persistence | Medium | Resurrection often weightless | P1 |
| Task-head structural coupling | Medium | Requires strict topology-sync guardrails | P1 |
| Consciousness orchestration | Medium | Exists but not automated in core loop | P2 |
| Biome transfer workflows | Low | Not yet productized | P3 |

---

## ✅ Acceptance Criteria

### For Bug Fixes
- [ ] All existing test suites used by repository CI pass.
- [ ] `training_demos/cifar10_full_training.py` imports and runs without syntax errors.
- [ ] CIFAR-10 full demo survives prune/plant events without task-head shape mismatch.
- [ ] `ForestEcosystem.save_checkpoint("forest.pt")` succeeds.
- [ ] `load_checkpoint` roundtrip succeeds with explicit `weights_only` arg.
- [ ] `ForestTeacher` tensors are created on forest device.
- [ ] Pruning in ecosystem simulation rebuilds optimizer and continues training.
- [ ] Graveyard defaults work on Linux/macOS/Windows temp directories.

### For Growth Phases
- [ ] **Phase A:** measurable inter-tree message usage and improved adaptation under drift.
- [ ] **Phase B:** offspring survival/fitness beats random planting baseline.
- [ ] **Phase C:** season transitions change growth/prune/replay behavior in logs.
- [ ] **Phase D:** biome transplant reduces warmup epochs on target domain.
- [ ] **Phase E:** canopy attention improves validation loss vs weighted-sum aggregator.
- [ ] **Phase F:** consciousness planner triggers periodically and emits actionable policies.

---

## 🗓️ Suggested Sprint Order

- **Sprint 1 (Bugs):** Fix all 11 bugs. All tests green.
- **Sprint 2 (Debt):** Fix 4 structural weaknesses (Silent Router, Duplicate Forest, Frozen Fitness, Dim Task Head).
- **Sprint 3 (Mycelium):** Implement Phase A.
- **Sprint 4 (Seasons):** Implement Phase C.
- **Sprint 5 (Genetics):** Implement Phase B.
- **Sprint 6 (Consciousness):** Implement Phase F.
- **Sprint 7 (Canopy):** Implement Phase E.
- **Sprint 8 (Biomes):** Implement Phase D.
