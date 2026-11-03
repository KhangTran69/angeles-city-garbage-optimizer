# Optimizer Sensitivity Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the fixed optimizer benchmark with reproducible ±10% demand sensitivity scenarios and a self-describing JSON report.

**Architecture:** Keep the benchmark in `src/benchmark.py`; create scenario-specific waste mappings from immutable defaults, share one deterministic Haversine matrix, and use the same seeds and algorithm settings for all scenarios. Hash canonical scenario inputs, including early-stopping patience and minimum improvement, and report source/runtime metadata, per-run metrics, and cooperative timeout status. The timeout deadline begins at invocation, but setup before the first generation check and an active generation are uninterruptible. Update README usage and interpretation without changing the Streamlit app.

**Tech Stack:** Python 3.9+, NumPy, pytest, argparse, hashlib, json, subprocess, platform.

**Spec:** `docs/superpowers/specs/2026-11-03-optimizer-sensitivity-report-design.md`

## Global Constraints

- Preserve the existing base input: six named barangays, seeds 42–44, three vehicles, 10,000 kg per vehicle, population 20, and 30 generations.
- Change only waste demand between the base, -10%, and +10% scenarios.
- Use the built-in default coordinates and Haversine distance matrix; do not use saved edits or live APIs.
- Do not claim field accuracy or a hard wall-clock guarantee.
- Keep timeout handling cooperative: the deadline begins at invocation, setup before the first generation check and an active generation cannot be interrupted, and the best-so-far result is retained with its timeout status.

## Review Focus

- Scenario feasibility must use the scenario waste mapping rather than the global default mapping; test that capacity outcomes change when a scenario crosses a route limit.
- Scenario fingerprints must change when demand/configuration changes and remain stable for identical canonical inputs.
- Same seed and same scenario must reproduce route and feasibility fields; do not include runtime in deterministic equality assertions.
- Timeout status must be paired with the returned route’s feasibility and distance rather than discarding best-so-far output.
- Source revision detection must be safe outside a Git checkout and must not make the CLI depend on external network access.

---

### Task 1: Define and fingerprint fixed sensitivity scenarios

**Files:**
- Modify: `src/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Preserve `run_fixed_benchmark()` as the public callable, adding optional `timeout_seconds: float | None = None`.
- Add a deterministic helper that returns scenario descriptors for demand multipliers 0.9, 1.0, and 1.1.
- Add a canonical fingerprint helper for scenario inputs.

- [ ] Write tests asserting exactly three named scenarios, the unchanged six-stop/seeds/base config, scaled exact demand values (including a synthetic half-kilogram tie rounded up), an unchanged base mapping, and different stable hashes for each demand scenario.
- [ ] Run `python -m pytest tests/test_benchmark.py -q` and confirm a behavior assertion fails before implementation.
- [ ] Implement scenario construction using a copied base waste mapping and deterministic nearest-kilogram, ties-up rounding; hash canonical JSON of scenario name, multiplier, rounding rule, scenario inputs, and all explicit fleet/GA configuration, including mutation/crossover rates, early-stopping patience/minimum improvement, and optional per-run timeout budget.
- [ ] Run the focused tests and confirm the scenarios/fingerprints pass.

### Task 2: Report comparable per-scenario and per-seed outcomes

**Files:**
- Modify: `src/benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] Write tests asserting each scenario compares its greedy baseline and GA using the same waste mapping and shared distance model, and reports per-seed route distance, feasibility, runtime, and timeout plus distance/feasibility/runtime/timeout summaries.
- [ ] Run the focused tests and confirm they fail because the report lacks scenario-level metrics and timeout fields.
- [ ] Update feasibility calculation to accept scenario waste data; pass the optional cooperative timeout into `run_ga`; retain its best-so-far result and timeout marker.
- [ ] Add Git revision and dirty-state detection plus a deterministic source-tree hash over tracked and nonignored untracked project files, excluding caches/build output, local secrets, and the exact JSON report output path; ensure the output path does not affect dirty state. Use a safe unavailable-metadata fallback outside Git; record Python/NumPy versions.
- [ ] Run the focused tests; repeat the CLI and compare stable fields/fingerprints while excluding runtime fields.

### Task 3: Expose timeout option and document interpretation

**Files:**
- Modify: `src/benchmark.py`
- Modify: `tests/test_benchmark.py`
- Modify: `README.md`

- [ ] Add CLI parsing for optional `--timeout-seconds`; reject zero, negative, NaN, and infinity with a clear argument error.
- [ ] Add CLI tests for valid JSON output and invalid timeout values.
- [ ] Document the exact command, scenario construction, report fields/fingerprint, independent per-run timeout-budget scope, invocation-time deadline, uninterruptible setup/generation intervals, and the limits of illustrative demand, Haversine distances, cooperative timeouts, and runtime comparisons.
- [ ] Run focused benchmark tests and inspect the CLI output for all documented fields.

### Task 4: Verify the complete repository

**Files:** no additional changes expected.

- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m ruff check src tests --select E9,F`.
- [ ] Run `python -m basedpyright src`.
- [ ] Run `python -m compileall -q src tests` and `git diff --check`.
- [ ] Inspect the final diff and a saved JSON report; confirm no unrelated files or generated artifacts are included.
