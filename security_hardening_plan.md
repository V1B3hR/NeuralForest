Phase 1 — Critical security and correctness
1. api/forest_api.py
Risk level: Critical

Vulnerability / issue checklist

 Uses torch.load(...) directly in checkpoint paths.
 Checkpoint validation path also deserializes content.
 get_info() deserializes full checkpoint without trust boundary.
 health_check() assumes nested status keys always exist.
 predict() does not fully guard empty-forest or invalid top_k scenarios.
 Error responses are structured, but observability is limited.
 ForestCheckpoint.save() re-loads saved artifact just to inject metadata.
Why it matters

Main deserialization attack surface.
Main external-facing API path.
Empty-state bug can break monitoring.
Required fixes

 Centralize checkpoint loading in safe helper methods.
 Add trusted-only checkpoint warning in code comments/docstrings.
 Stabilize get_forest_status() schema for empty forests.
 Make health_check() robust for zero-tree state.
 Validate/clamp top_k.
 Improve malformed input handling.
 Avoid unnecessary read-back load in save() if possible.
Recommended tests

 malformed checkpoint validation
 empty forest health check
 invalid input shape
 top_k > number of trees
 valid round-trip checkpoint load
2. NeuralForest.py
Risk level: Critical

Vulnerability / issue checklist

 load_checkpoint() uses torch.load(..., weights_only=False).
 Trust boundary for checkpoint files is undocumented in loading logic.
 Load path is core and likely reused across repo.
 Print-based checkpoint messages instead of structured logging.
Why it matters

Core deserialization entrypoint.
If this stays unsafe, wrappers won’t fully solve the problem.
Required fixes

 Add explicit “trusted artifacts only” docstring and comments.
 Route loading through a common validation/helper path.
 Add integrity hook support if practical.
 Prefer logging over print for load/save status in non-demo code.
Recommended tests

 trusted valid checkpoint loads successfully
 malformed checkpoint fails cleanly
 missing required keys handled predictably
3. canopy/hierarchical_router.py
Risk level: High

Vulnerability / issue checklist

 Batch routing bug: uses topk_indices[0, i], so sample 0 controls whole batch.
 Grove failures are swallowed with print(...).
 Degraded routing is not surfaced in returned metadata.
 Production path lacks structured logging.
Why it matters

Major inference correctness bug.
Silent degradation undermines reliability and benchmarking.
Required fixes

 Refactor routing to operate per sample.
 Replace print(...) with logger.warning(...).
 Add failed_groves / failure_count to routing_info.
 Add tests for mixed-routing batches.
Recommended tests

 two-sample batch picks different groves
 one grove failure is logged and surfaced
 output shape preserved after refactor
Phase 2 — High-value hardening and doc alignment
4. tropical_forest_map.md
Risk level: Low, but important for expectations

Vulnerability / issue checklist

 Not executable, so low direct security risk.
 Overstates production maturity with “deployment-ready”.
 Does not mention remaining hardening work.
Why it matters

Misleading maturity claims can cause unsafe adoption.
Required fixes

 Change “deployment-ready” wording to “deployment-oriented” or similar.
 Add a hardening follow-up bullet:
safe checkpoint handling
observability
production validation
security review
Recommended edits

 Keep the phase structure intact.
 Add a concise Phase 7 follow-up note rather than rewriting the file.
5. readme.md
Risk level: Medium

Vulnerability / issue checklist

 Describes production API without security caveats.
 Lacks trusted-checkpoint warning.
 May imply stronger production readiness than implementation supports.
 Still references tree_graveyard.py in project structure while roadmap/map mentions humus_nursery.py.
Why it matters

README is the primary trust surface for users.
Required fixes

 Add “Security Notes” section.
 State checkpoints are trusted-only artifacts.
 Clarify production API is production-oriented, not fully hardened.
 Reconcile stale structure references if needed.
Recommended tests

No code tests, but doc review checklist:
 API example includes warning
 deployment language is moderated
 structure references are consistent
Phase 3 — Broader repo hygiene and operational hardening
6. Demo / script files using sys.path.insert(...)
Examples from search:

phase1_demo.py
phase2_demo.py
phase2_ecosystem_demo.py
phase3_demo.py
phase3_evolution_demo.py
phase6_demo.py
phase7_demo.py
training_demos/layer_wise_optimizer.py
training_demos/few_shot_demo.py
Risk level: Low to Medium

Vulnerability / issue checklist

 Runtime path mutation can create ambiguous import resolution.
 Mostly demo-only, but some patterns appear in non-demo code too.
 Weak packaging discipline for production deployment.
Why it matters

More of a maintainability and environment-hardening issue than a direct exploit.
Required fixes

 Do not block main hardening PR on all of these.
 Track as follow-up packaging cleanup.
 Remove from library code first, demos later.
7. requirements.txt
Risk level: Medium

Vulnerability / issue checklist

 Loosely pinned dependencies.
 No lockfile.
 Reproducibility and supply-chain confidence are limited.
Required fixes

 Tighten tested version ranges.
 Add pinned dev/test environment in follow-up PR.
 Consider audit tooling in CI later.
Phase 4 — Follow-up correctness and resilience review
8. evolution/self_improvement.py
Risk level: Medium

Vulnerability / issue checklist

 Mostly not a security file, but action execution/rollback logic is sensitive.
 Some actions are non-reversible.
 Could benefit from stronger invariant tests.
 Cooldown logic is complex and should be regression-protected.
Required fixes

 Not in the first PR unless touching related tests.
 Add follow-up tests around rollback/cooldown behavior.
9. evolution/humus_nursery.py
Risk level: Medium

Vulnerability / issue checklist

 Loads resurrected weights via torch.load(...).
 Another deserialization path.
 Likely lower exposure than API path, but still relevant.
Required fixes

 Include in deserialization audit if feasible in same PR.
 If not, add explicit follow-up issue.
 Apply same trusted-artifact rule here.
Recommended implementation phases
Phase A — Must do first
Goal: remove biggest risk and biggest correctness issue.

Files

api/forest_api.py
NeuralForest.py
canopy/hierarchical_router.py
Deliverables

checkpoint handling hardening path
per-sample canopy routing
stable empty-forest health/status
logging instead of print in canopy
Phase B — Align documentation with reality
Files

tropical_forest_map.md
readme.md
Deliverables

trusted-only checkpoint guidance
moderated Phase 7 claims
mention hardening still pending
Phase C — Regression coverage
Files

new tests under tests/
Recommended test files

tests/test_checkpoint_api.py
tests/test_api_health.py
tests/test_canopy_batch_routing.py
Deliverables

test coverage for all fixes from A and B where applicable
Phase D — Follow-up hardening backlog
Files

demos/scripts with sys.path.insert(...)
requirements.txt
evolution/humus_nursery.py
evolution/self_improvement.py
Deliverables

packaging cleanup
dependency hardening
full deserialization audit
additional resilience tests
Recommended PR description
Title
Harden checkpoint handling, fix canopy batch routing, and align production docs

Summary
hardens checkpoint handling paths and documents trusted-only checkpoint usage
fixes canopy batch routing so each sample selects groves independently
stabilizes API health/status behavior for empty forests
replaces print-based routing warnings with structured logging
updates docs to better reflect current production maturity
Why
This PR addresses the highest-priority risks in the current production-oriented surface:

unsafe checkpoint deserialization is the main security concern
canopy routing currently applies sample 0’s grove selection to the full batch
health/status endpoints can fail on empty forests
routing failures are not surfaced clearly
docs overstate deployment readiness relative to current hardening
Scope
Code

centralize and harden checkpoint loading behavior
make canopy routing per-sample for batched inference
stabilize empty-forest API schema
replace print warnings with logger-based observability
Docs

add trusted-checkpoint usage guidance
soften “deployment-ready” wording
document remaining hardening work
Tests

checkpoint validation and malformed input handling
empty forest health/status behavior
canopy mixed-batch routing and grove-failure handling
Out of scope
full packaging cleanup for all demo scripts
dependency lockfile / CI security workflow
serialization format replacement
broader architecture refactors
Short file-by-file checklist version
Must-fix now
 api/forest_api.py
 NeuralForest.py
 canopy/hierarchical_router.py
Must-update docs now
 tropical_forest_map.md
 readme.md
Must-add tests now
 tests/test_checkpoint_api.py
 tests/test_api_health.py
 tests/test_canopy_batch_routing.py
Follow-up later
 demo scripts with sys.path.insert(...)
 requirements.txt
 evolution/humus_nursery.py
 evolution/self_improvement.py
