# 🌲 NeuralForest — The Tropical Evolution: Phase 0

This map outlines the transformation of NeuralForest from a mechanical plantation into a conscious, symbiotic tropical ecosystem while keeping the repository grounded in practical engineering steps.

## 🗺️ Tropical Forest Map (Overview)

### 🌿 Phase 1: Soil Purification & Rebirth ✅
- [x] **Task:** Fix the 11 critical bugs identified in `development_roadmap.md` and reflected by the current regression coverage in `tests/test_bugfix_regressions.py` and `tests/test_rain_cooldown.py`.
- [x] **Identity Shift:** Rename `evolution/tree_graveyard.py` to `evolution/humus_nursery.py` (`HumusNursery` class). Backward-compat shim kept at `evolution/tree_graveyard.py`. Updated references to "pruning" as "natural selection" in ecosystem_simulation.py.
- [x] **Vibe:** Clean soil is the foundation of a healthy forest.

### 🌸 Phase 2: The Flowering Flora (Symbiosis) ✅
- [x] **Flowers (Attraction Interfaces):** Added `is_blooming` property and `bloom_signal()` method to `SpecialistTree` (groves/base_grove.py). Trees bloom when `fitness ≥ 7.0` and `expertise_score ≥ 0.5`. Added `get_blooming_trees()` and `form_symbiotic_clusters()` to `Grove` to group blooming trees into cooperative symbiotic clusters. Tests in `tests/test_phase2.py`.
- [x] **Litter Layer (Multimodal Memory):** `mycelium/knowledge_transfer.py` provides `litter_absorption_loss()` — young trees passively absorb feature litter dropped into `PrioritizedMulch` and `GroveMulch` by mature trees. Short-term/long-term memory separation is handled by priority-weighted replay buffers.

### 🦋 Phase 3: The Moving Fauna (Dynamic Agents)
- [ ] **Bees (Pollinators):** Genetic crossover agents that transfer weight "nectar" between trees to prevent stagnation.
- [ ] **Monkeys (Chaos Resilience):** Agile agents that jump between "branches" (sub-networks) to test stability without harming healthy structures.
- [ ] **Complexity Predators:** Optimization processes that "consume" inefficient trees to maintain forest elegance.

### 🌧️ Phase 4: The Hydrological Cycle (Natural Forces)
- [ ] **Rain:** Represent data streams and gradient flows as life-giving rain.
- [ ] **Evaporation:** Model weight decay and parameter cooling as evaporation, returning "moisture" to the atmosphere for the next cycle.

### 🧠 Phase 5: Canopy Consciousness (Global Unity)
- [ ] **Reflection:** The forest becomes aware of its state (entropy, stability) and adjusts its "seasons" based on collective health, not just a clock.

## 🛠️ Prioritized Job (Immediate Actions)
1. Create `tropical_forest_map.md` with these phases.
2. Prepare the soil by batching the 11 bug fixes in the next PR by subsystem:
   - **Batch A — Core vitality:** module imports, layer construction, mulch eviction, task-head sizing, checkpoint save/load, forest-device consistency, and early fitness updates.
   - **Batch B — Ecosystem integrity:** optimizer rebuild after natural selection plus the `tree_graveyard` → `humus_nursery` identity migration.
   - **Batch C — Hydrological behavior:** rain cooldown logic, thresholds, and reporting in the self-improvement loop.
3. Keep the next wave migration-friendly for environments like Antigravity or PyCharm IDE by favoring stable entry points, explicit paths, centralized configuration, and IDE-neutral setup docs.

## 🌱 Phase 0 Execution Note

The best next PR is a single soil-preparation pass that:
- locks the 11 bug fixes to regression tests first,
- performs the naming migration immediately after the tests are green,
- and only then expands into flowers, litter, fauna, and hydrological intelligence.

**VIBE CODING MODE: BOTANIST / CONSERVATIONIST ACTIVE.**
