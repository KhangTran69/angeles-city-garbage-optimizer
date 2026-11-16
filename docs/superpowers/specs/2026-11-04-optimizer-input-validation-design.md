# Optimizer Input Validation Design

## Intent

Reject malformed genetic-algorithm inputs before route indexing or population work, and identify mathematically infeasible demand instead of returning routes that appear usable. The first user-facing target is the main dispatch flow and its research benchmark callers.

## Goals

- Validate selected barangays, selected waste values, vehicle count and capacity, GA population/generation settings, mutation/crossover rates, early-stopping settings, and timeout before solver setup.
- Bound direct API inputs by the current UI maxima: 25 vehicles, population 500, 1,000 generations, and 20,000 kg per vehicle.
- Preserve the existing four-value `run_ga` return tuple for feasible and empty selections.
- Raise a specific `InfeasibleRoutingProblem` exception with stable reason codes and machine-readable details when a stop exceeds one vehicle's capacity or total demand exceeds the fleet's capacity.
- Keep direct solver calls strict by default; permit the main app to request an explicit diagnostic candidate for an infeasible input under the existing 130% workload ceiling.
- Show a clear feasibility result for dispatch and benchmark outputs; never label or package an infeasible route candidate as a dispatch.

## Non-goals

- Changing the GA search, overload scoring, route decoding, or the existing 130% UI guard.
- Returning a new result type for successful solver calls; an optional status dictionary reports route feasibility without changing the four-value return tuple.
- Claiming operational feasibility from the illustrative bundled waste estimates.

## Approaches

- **Specific exception at the solver boundary (chosen):** keep successful callers compatible and make infeasible requests explicit. UI callers catch and display the exception. This is the narrowest coherent change across the solver and its current UI entry points.
- **New result wrapper for every call:** would represent infeasible status directly, but would require migrating every `run_ga` consumer and compatibility surface.
- **Opt-in diagnostic mode:** preserve the strict default and allow only callers that opt in to receive infeasible diagnostic routes; the main UI labels them and disables dispatch packaging.

## Constraints and acceptance

- Feasible routes and valid empty selections keep the existing tuple shape and behavior.
- Invalid input fails before matrix construction, seeding, or GA iteration with a stable `ValueError` message; the configured maxima match the UI controls.
- Infeasible demand raises `InfeasibleRoutingProblem` by default; `reason_codes` identifies `stop_exceeds_vehicle_capacity` and/or `total_demand_exceeds_fleet_capacity`, and `details` records loads and capacities.
- The main app's diagnostic mode is opt-in and limited by the existing 130% workload guard. It labels infeasible candidates, preserves the result status in JSON/CSV, and disables Dispatch ZIP.
- Benchmark UI exposes per-seed feasibility and timeout flags.
- No dependencies are added. The project's declared test, Ruff, Basedpyright, compile, and diff checks remain the verification targets.

## Calls made for this implementation

- A single-stop load above per-vehicle capacity is infeasible even when total fleet capacity is sufficient.
- A selected-stop list must contain distinct names from the bundled barangay set; a supplied custom waste mapping must include each selected stop.
- Preserve the existing empty-mapping behavior: `None` or `{}` selects the bundled estimates, while a non-empty custom mapping must include each selected stop.
- The UI's existing 130% precheck is the workload ceiling. The main app can show an under-ceiling infeasible candidate only as a diagnostic; all direct solver calls and benchmark inputs remain strict by default.
