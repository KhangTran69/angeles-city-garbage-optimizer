# Optimizer Sensitivity Report Design

## Intent

Extend the existing fixed-input CLI benchmark so a reviewer can reproduce the current comparison and see how its results respond to small, controlled changes in illustrative waste estimates. Preserve the existing three seeds, six barangays, Haversine matrix, capacity-aware greedy baseline, and genetic algorithm settings.

## Goals

- Report the current demand scenario plus uniform -10% and +10% demand scenarios, using the same seed list and all other benchmark inputs.
- For each scenario, report exact input values and a canonical SHA-256 input fingerprint.
- For each seed, report baseline/optimizer route distance and feasibility, optimizer runtime, and cooperative timeout status.
- Record the source revision when the benchmark runs from a Git checkout and record Python/NumPy versions.
- Allow an optional cooperative `--timeout-seconds` budget and label it as cooperative in output and documentation.
- Keep conclusions bounded to the stated illustrative inputs and straight-line distance approximation.

## Non-goals

- Changing route optimization behavior, the Streamlit interface, or existing default seed/configuration values.
- Adding live road-service calls, municipal datasets, or claims of field effectiveness.
- Enforcing a hard wall-clock deadline. The deadline starts at solver invocation, but setup before the first generation check and an active generation cannot be interrupted, so runtime can exceed the cooperative budget in either case.
- Replacing or duplicating the existing benchmark and sensitivity UI.

## Design

Keep the benchmark implementation in `src/benchmark.py` and expose its existing CLI. Produce a schema-versioned JSON document with source identity, runtime environment, fixed benchmark settings, and three named demand scenarios. Build each scenario from the same selected stops and base waste estimates, scaling every selected estimate by 0.9, 1.0, or 1.1 and rounding positive kilogram values to the nearest integer with ties rounded up. Each scenario uses the same fixed Haversine distance matrix, fleet, capacity, algorithm settings, and seeds.

Compute an input fingerprint from canonical JSON containing the scenario name and multiplier, exact waste values, explicit rounding rule, ordered selected stops, built-in coordinates/depot, Haversine model and radius, seeds, and all fleet/algorithm settings (including mutation and crossover rates and early-stopping patience/minimum improvement). Include a configured timeout budget in the fingerprint when present because it changes run conditions. Keep source identity separate from this scenario fingerprint: report the Git revision, whether the worktree is dirty, and a deterministic source-tree SHA-256 so a dirty checkout cannot be mistaken for the code at `HEAD`. The source-tree hash covers Git-tracked files and nonignored untracked project files in stable path order, excluding documented caches/build artifacts, local environment secrets, and the exact report output file. The dirty flag ignores that output file too. Include baseline feasibility and distance once per scenario, plus optimizer runs and summaries. Preserve feasibility as a separate outcome from distance so infeasible results cannot be described as wins. The optional cooperative timeout budget applies independently to each scenario/seed GA run. Its deadline starts at invocation, while setup before the first generation check and any active generation are uninterruptible; runtime may exceed the budget during either period. Report the budget and timeout status per run, and retain the best-so-far route.

The report is a CLI artifact, so no UI change is needed unless implementation review finds the existing display is the only place the result must be communicated. The README will document the command, scenario construction, fingerprint, fields, and limitations.

## Acceptance

- The existing fixed scenario remains represented with seeds 42–44 and its current baseline/optimizer configuration.
- The report contains three explicit demand scenarios with stable, distinct input fingerprints and exact per-scenario values.
- The fingerprint changes when any demand or explicit fleet/GA setting changes, including mutation rate, crossover rate, early-stopping patience/minimum improvement, or a configured per-run timeout budget.
- The fingerprint includes the scenario multiplier and the named nearest-kilogram, ties-up rounding rule.
- Demand scaling rounds positive fractional kilograms to the nearest integer, with exact half-kilogram ties rounded up; scenario construction does not mutate bundled source data.
- Each scenario reports baseline feasibility/distance and per-seed optimizer distance, feasibility, runtime, and timeout state, with summary feasibility, distance spread, runtime, and timeout rates.
- The report identifies Git revision, dirty/clean worktree state, and a source-tree SHA-256; it can run outside Git with a safe unavailable-metadata fallback.
- Writing the report to an in-repository path does not change the hash or dirty flag on the next run; the hash still changes when tracked or nonignored untracked project files change.
- Repeated runs produce identical non-runtime results and fingerprints for the same revision and inputs.
- `--timeout-seconds` rejects non-positive/non-finite values and reports that its budget applies separately to each scenario/seed run. Metadata and documentation state that the deadline begins at invocation, setup before the first check and an active generation cannot be interrupted, runtime may exceed the budget in either case, and the best-so-far route is retained.
- Documentation identifies waste estimates as illustrative, distances as straight-line Haversine, timeouts as cooperative, and the results as algorithm sensitivity evidence rather than operational validation.
- The repository test, lint, type-check, and compile checks pass.

## Calls made for this implementation

- Use a uniform ±10% waste-demand perturbation as a transparent, small sensitivity band.
- Keep the same six-stop scenario, seeds, capacity, distance model, and algorithm settings across all three cases so demand is the only changed input.
- Treat runtime as environment-dependent; compare and report it without making a performance guarantee.
