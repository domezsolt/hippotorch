hippotorch changelog

Date: 2026-01-03

Summary of changes
- Added `linear_mixture_schedule` utility for semantic/uniform ratio ramp.
- Exported schedule at package level (hippotorch.memory and hippotorch).
- Implemented minimal `DQNAgent` and CartPole wake/sleep example script.
- Added consolidation clustering benchmark script and docs.
- Wrote quickstart, hybrid sampling, diagnostics, wake/sleep, CartPole, and consolidation benchmark docs.
- Updated roadmap progress in `hippotorch_framework.md` and mapped in `docs/roadmap_progress.md`.
- Added CartPole sensitivity CLI sweeping mixture ratio, temperature, momentum, and consolidation budgets; updated sensitivity and benchmarks docs.
- Added requirements.txt (core dependency: torch), and dependencies doc with optional example/test packages.
- Added notebook_probes.md with copy/paste cells for quick sanity checks (no Gym required).
- Added configurable flags:
  - cartpole_wakesleep: --mixture-ratio, --schedule-start/--schedule-end, --no-schedule
  - cartpole_multiseed_eval: --hybrid-mixture, --schedule-start/--schedule-end, --no-schedule
  - compare_cartpole.sh: CLI flags for seeds/steps/batch/warmup/schedule/mixture/etc.
- CartPole milestone reached: hybrid matches/exceeds uniform replay on aggregate (see `results/` logs).
 - Added POMDP-friendly sequence sampling: `HippocampalReplayBuffer.sample_windows` for temporal windows.
 - Implemented lazy key refresh during retrieval to avoid O(N) stalls.
 - Switched reward-weighted InfoNCE to rank-based weighting for stability across reward scales/signs.
 - Added smart eviction policy in `MemoryStore` (importance score) to protect valuable episodes.
 - Added Amnesiac's Corridor benchmark script (`examples/amnesiac_corridor.py`) illustrating credit assignment under interference and long horizons.

Notes
- Regression harness covers a “needle-in-haystack” scenario; Phase 1 milestone marked complete.
- CartPole example added; the “matches standard replay” milestone requires running the example to confirm.
- Consolidation clustering benchmark script added; improvement should be observable on synthetic data.
