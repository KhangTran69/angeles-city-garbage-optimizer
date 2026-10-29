"""Fixed-input, fixed-seed comparison for the route optimizer and a greedy baseline."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections import Counter
from typing import Any

import numpy as np

from src.data import BARANGAYS, BARANGAY_WASTE_KG, DEFAULT_BARANGAY_COORDS, DEPOT, DEPOT_COORD
from src.optimization import greedy_baseline, run_ga
from src.utils import get_route_load, haversine

BENCHMARK_STOPS = (
    "Amsic",
    "Anunas",
    "Balibago",
    "Capaya",
    "Claro M. Recto",
    "Cuayan",
)
BENCHMARK_VEHICLES = 3
BENCHMARK_CAPACITY_KG = 10_000
BENCHMARK_SEEDS = (42, 43, 44)
BENCHMARK_POPULATION = 20
BENCHMARK_GENERATIONS = 30


def _route_feasibility(routes: list[list[str]]) -> dict[str, Any]:
    counts = Counter(stop for route in routes for stop in route)
    return {
        "all_selected_stops_once": counts == Counter(BENCHMARK_STOPS),
        "all_routes_within_capacity": all(
            get_route_load(route, BARANGAY_WASTE_KG) <= BENCHMARK_CAPACITY_KG for route in routes
        ),
        "route_count": len(routes),
    }


def _benchmark_distances() -> tuple[np.ndarray, dict[str, int]]:
    """Build a deterministic matrix from code defaults, ignoring editable saved coordinates."""
    points = [DEPOT, *BARANGAYS]
    coordinates = {DEPOT: DEPOT_COORD, **DEFAULT_BARANGAY_COORDS}
    point_to_index = {point: index for index, point in enumerate(points)}
    matrix = np.array(
        [[haversine(coordinates[left], coordinates[right]) for right in points] for left in points],
        dtype=float,
    )
    return matrix, point_to_index


def run_fixed_benchmark() -> dict[str, Any]:
    """Return deterministic GA runs and greedy baseline metrics on one distance model."""
    distance_matrix, point_to_index = _benchmark_distances()
    baseline_distance, baseline_routes = greedy_baseline(
        list(BENCHMARK_STOPS),
        BENCHMARK_VEHICLES,
        waste_dict=BARANGAY_WASTE_KG,
        road_graph=None,
        dist_matrix=distance_matrix,
        pt_to_idx=point_to_index,
        vehicle_capacity_kg=BENCHMARK_CAPACITY_KG,
    )
    baseline_feasibility = _route_feasibility(baseline_routes)

    runs: list[dict[str, Any]] = []
    for seed in BENCHMARK_SEEDS:
        started = time.perf_counter()
        routes, _, total_distance, _ = run_ga(
            list(BENCHMARK_STOPS),
            n_vehicles=BENCHMARK_VEHICLES,
            pop_size=BENCHMARK_POPULATION,
            generations=BENCHMARK_GENERATIONS,
            mutation_rate=0.05,
            crossover_rate=0.85,
            vehicle_capacity_kg=BENCHMARK_CAPACITY_KG,
            waste_dict=BARANGAY_WASTE_KG,
            road_graph=None,
            distance_matrix=distance_matrix,
            point_to_index=point_to_index,
            random_seed=seed,
        )
        feasibility = _route_feasibility(routes)
        runtime_seconds = time.perf_counter() - started
        runs.append(
            {
                "seed": seed,
                "route_distance_km": round(float(total_distance), 2),
                "runtime_seconds": round(runtime_seconds, 6),
                "feasible": (
                    feasibility["all_selected_stops_once"]
                    and feasibility["all_routes_within_capacity"]
                    and feasibility["route_count"] <= BENCHMARK_VEHICLES
                ),
                "feasibility": feasibility,
            }
        )

    distances = [run["route_distance_km"] for run in runs]
    runtimes = [run["runtime_seconds"] for run in runs]
    baseline_feasible = (
        baseline_feasibility["all_selected_stops_once"]
        and baseline_feasibility["all_routes_within_capacity"]
        and baseline_feasibility["route_count"] <= BENCHMARK_VEHICLES
    )
    return {
        "schema_version": 1,
        "runtime_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "benchmark": "fixed-seed GA vs. greedy baseline",
        "distance_model": (
            "Haversine from built-in default coordinates; saved coordinate edits and live routing are ignored"
        ),
        "data_note": "Bundled waste values are illustrative estimates, not validated operational measurements.",
        "configuration": {
            "selected_barangays": list(BENCHMARK_STOPS),
            "waste_kg_by_barangay": {name: BARANGAY_WASTE_KG[name] for name in BENCHMARK_STOPS},
            "vehicles": BENCHMARK_VEHICLES,
            "vehicle_capacity_kg": BENCHMARK_CAPACITY_KG,
            "population": BENCHMARK_POPULATION,
            "generations": BENCHMARK_GENERATIONS,
            "seeds": list(BENCHMARK_SEEDS),
        },
        "baseline": {
            "name": "capacity-aware greedy nearest-stop heuristic",
            "route_distance_km": round(float(baseline_distance), 2),
            "feasible": baseline_feasible,
            "feasibility": baseline_feasibility,
        },
        "optimizer": {
            "name": "genetic algorithm with capacity-aware decoding and 2-opt",
            "runs": runs,
            "mean_route_distance_km": round(sum(distances) / len(distances), 2),
            "distance_stddev_km": round(statistics.pstdev(distances), 2),
            "distance_range_km": [min(distances), max(distances)],
            "mean_runtime_seconds": round(statistics.mean(runtimes), 6),
            "best_route_distance_km": min(distances),
            "all_runs_feasible": all(run["feasible"] for run in runs),
        },
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fixed-input route benchmark.")
    parser.add_argument("--json-out", help="Write the report to a JSON file; stdout is the default.")
    args = parser.parse_args(argv)
    rendered = json.dumps(run_fixed_benchmark(), indent=2, sort_keys=True) + "\n"
    if args.json_out:
        from pathlib import Path

        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
