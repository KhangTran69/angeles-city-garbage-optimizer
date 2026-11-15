# Optimizer Input Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate GA inputs before solver work and report infeasible demand explicitly to direct and Streamlit callers.

**Architecture:** Add a shared inexpensive validator and a structured `InfeasibleRoutingProblem` exception in `src/optimization.py`. Keep the four-value `run_ga` return unchanged; direct calls remain strict by default, while the main app can opt into a feasibility-labeled diagnostic result below the existing workload ceiling.

**Tech Stack:** Python, Streamlit, pytest, NumPy.

**Spec:** `docs/superpowers/specs/2026-11-04-optimizer-input-validation-design.md`

## Global Constraints

- Preserve the existing four-value `run_ga` return tuple for feasible and empty selections.
- Keep the existing 130% UI workload ceiling and do not alter GA search behavior.
- Match API maxima to the UI: 25 vehicles, population 500, generations 1,000, and capacity 20,000 kg.
- Add no dependencies.
- Treat bundled waste values as illustrative.

## Review Focus

- Unknown or repeated stops: direct calls fail before name-to-index lookup; pin with `test_run_ga_rejects_unknown_and_duplicate_stops`.
- Missing, negative, or non-finite selected waste: direct calls fail before load calculation; pin with parametrized selected-waste validation tests.
- Non-positive or above-maximum fleet/capacity and invalid or above-maximum GA bounds: direct calls fail before seeding; pin with parameterized numeric-boundary tests.
- One oversized stop or total demand above capacity: receive stable reason codes/details before GA iteration; pin with `test_run_ga_reports_oversized_stop_and_over_capacity_demand`.
- Streamlit research benchmark failure: show a clear error and no success metrics; pin with `test_sensitivity_analysis_displays_infeasible_error`.

---

### Task 1: Validate solver inputs and classify infeasible demand

**Files:**
- Modify: `src/optimization.py`
- Modify: `src/config.py`
- Test: `tests/test_optimization.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `InfeasibleRoutingProblem(ValueError)` with `reason_codes: list[str]` and `details: dict[str, Any]`.
- Produces: `validate_ga_inputs(...) -> Dict[str, int]`, returning the effective waste mapping after validating selected inputs.
- Constraint: an empty mapping keeps the existing bundled-data fallback; a non-empty custom mapping must cover every selected barangay.

- [x] Add tests for unknown/duplicate stops, absent/negative/non-finite selected waste, invalid and above-maximum fleet/capacity/GA settings, exact feasible capacity boundaries, single-stop overload, and total fleet overload.
- [x] Run `python -m pytest tests/test_optimization.py -q`; confirm failures name missing validation/exception behavior.
- [x] Implement a shared validator and structured infeasibility exception; call validation at the start of `run_ga` and `run_benchmark`, before seed construction, indexing, or distance setup.
- [x] Document input bounds, waste requirements, and infeasible-demand behavior in the README.
- [x] Re-run `python -m pytest tests/test_optimization.py -q` and confirm all existing feasible/empty behavior remains green.

### Task 2: Surface infeasibility at Streamlit entry points

**Files:**
- Modify: `app.py`
- Modify: `src/research_validation.py`
- Test: `tests/test_optimization.py`

**Interfaces:**
- Consumes: `InfeasibleRoutingProblem` and `validate_ga_inputs(...)` from Task 1.
- Produces: a visible, explicitly infeasible diagnostic candidate for under-ceiling main dispatches; research benchmark requests still reject infeasible input.

- [x] Add a focused Streamlit-path regression that forces `run_benchmark` to raise `InfeasibleRoutingProblem` and asserts the UI displays the error without success output.
- [x] Run the focused UI regression and confirm it fails because the error is not yet handled.
- [x] Catch invalid solver inputs at the main dispatch and research benchmark entry points; keep infeasible dispatch candidates diagnostic-only.
- [x] Run focused tests, then the repository verification commands from `.github/workflows/ci.yml`, `python -m compileall -q src tests`, and `git diff --check`.

### Task 3: Preserve workload policy and make feasibility visible

**Files:**
- Modify: `src/optimization.py`, `src/reporting.py`, `app.py`, `README.md`
- Test: `tests/test_optimization.py`, `tests/test_reporting.py`

- [x] Add an opt-in solver diagnostic mode and a route feasibility assessment while preserving strict defaults.
- [x] Keep the 130% workload ceiling, label infeasible results, preserve status in JSON/CSV, and disable Dispatch ZIP for infeasible or unverified results.
- [x] Show per-seed feasibility and timeout values in benchmark results.
- [x] Add a Streamlit AppTest for an under-ceiling infeasible candidate, persistent explanation, and disabled Dispatch ZIP.
- [x] Re-run full verification after the review correction: pytest, Ruff E9/F, Basedpyright, compileall, and `git diff --check`.
