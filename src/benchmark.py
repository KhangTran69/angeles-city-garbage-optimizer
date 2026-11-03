"""Fixed-input sensitivity comparison for the route optimizer and greedy baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.data import BARANGAYS, BARANGAY_WASTE_KG, DEFAULT_BARANGAY_COORDS, DEPOT, DEPOT_COORD
from src.optimization import EARLY_STOP_MIN_DELTA, EARLY_STOP_PATIENCE, greedy_baseline, run_ga
from src.utils import get_route_load, haversine

BENCHMARK_STOPS = ("Amsic", "Anunas", "Balibago", "Capaya", "Claro M. Recto", "Cuayan")
BENCHMARK_VEHICLES = 3
BENCHMARK_CAPACITY_KG = 10_000
BENCHMARK_SEEDS = (42, 43, 44)
BENCHMARK_POPULATION = 20
BENCHMARK_GENERATIONS = 30
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.85
SCENARIO_MULTIPLIERS = (("minus_10_percent", 0.9), ("base", 1.0), ("plus_10_percent", 1.1))
DEMAND_ROUNDING_RULE = "nearest_integer_ties_up"


def _scenario_waste(name: str, multiplier: float, base_waste: dict[str, int] | None = None) -> dict[str, int]:
    """Return selected scenario values; positive half kilograms round up."""
    values = BARANGAY_WASTE_KG if base_waste is None else base_waste
    return {stop: math.floor(values[stop] * multiplier + 0.5) for stop in (BENCHMARK_STOPS if base_waste is None else values)}


def _scenario_fingerprint(
    scenario: dict[str, Any], *, selected_stops: list[str], coordinates: dict[str, Any], depot: Any,
    seeds: list[int], vehicles: int, capacity_kg: int, population: int, generations: int,
    mutation_rate: float, crossover_rate: float, early_stopping_patience: int,
    early_stopping_min_delta: float, timeout_seconds: float | None,
) -> str:
    payload = {
        "scenario": scenario["name"], "multiplier": scenario["multiplier"],
        "rounding_rule": scenario["rounding_rule"],
        "waste_kg_by_barangay": scenario["waste_kg_by_barangay"],
        "selected_stops": selected_stops, "coordinates": coordinates, "depot": depot,
        "distance_model": {"name": "Haversine", "earth_radius_km": 6371.0}, "seeds": seeds,
        "vehicles": vehicles, "capacity_kg": capacity_kg, "population": population,
        "generations": generations, "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate, "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "timeout_seconds_per_run": timeout_seconds,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _route_feasibility(routes: list[list[str]], waste_dict: dict[str, int]) -> dict[str, Any]:
    counts = Counter(stop for route in routes for stop in route)
    return {
        "all_selected_stops_once": counts == Counter(BENCHMARK_STOPS),
        "all_routes_within_capacity": all(get_route_load(route, waste_dict) <= BENCHMARK_CAPACITY_KG for route in routes),
        "route_count": len(routes),
    }


def _is_feasible(feasibility: dict[str, Any]) -> bool:
    return bool(feasibility["all_selected_stops_once"] and feasibility["all_routes_within_capacity"] and feasibility["route_count"] <= BENCHMARK_VEHICLES)


def _benchmark_distances() -> tuple[np.ndarray, dict[str, int]]:
    """Build a deterministic matrix from code defaults, ignoring editable saved coordinates."""
    points = [DEPOT, *BARANGAYS]
    coordinates = {DEPOT: DEPOT_COORD, **DEFAULT_BARANGAY_COORDS}
    point_to_index = {point: index for index, point in enumerate(points)}
    matrix = np.array([[haversine(coordinates[left], coordinates[right]) for right in points] for left in points], dtype=float)
    return matrix, point_to_index


def _source_identity(output_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    excluded_dirs = {
        ".git", ".pytest_cache", ".ruff_cache", "__pycache__", ".mypy_cache",
        ".basedpyright", ".venv", "venv", "build", "dist", "target", "htmlcov",
        "cache", "logs", "exports",
    }
    output = Path(output_path).resolve() if output_path is not None else None
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=root, capture_output=True, check=True,
        ).stdout
        output_relative = None
        if output is not None:
            try:
                output_relative = output.relative_to(root).as_posix()
            except ValueError:
                pass
        entries = status.split(b"\0")
        dirty_entries = []
        index = 0
        while index < len(entries) and entries[index]:
            entry = entries[index]
            index += 1
            state, raw_path = entry[:2], entry[3:]
            paths_in_entry = [raw_path.decode("utf-8", errors="surrogateescape")]
            if b"R" in state or b"C" in state:
                if index < len(entries):
                    paths_in_entry.append(entries[index].decode("utf-8", errors="surrogateescape"))
                    index += 1
            if output_relative is None or output_relative not in paths_in_entry:
                dirty_entries.append(entry)
        dirty = bool(dirty_entries)
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None

    try:
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root, capture_output=True, check=True,
        ).stdout
        paths = (root / item.decode("utf-8", errors="surrogateescape") for item in listed.split(b"\0") if item)
    except (OSError, subprocess.CalledProcessError):
        # Outside a usable Git worktree, hash the project files while still skipping
        # common generated directories and local environment files.
        paths = (path for path in root.rglob("*") if path.is_file())

    digest = hashlib.sha256()
    included = []
    for path in paths:
        try:
            relative_path = path.relative_to(root)
            relative = relative_path.as_posix()
        except ValueError:
            continue
        if not path.is_file() or excluded_dirs.intersection(relative_path.parts):
            continue
        if relative_path.name.startswith(".env") and relative_path.name != ".env.example":
            continue
        if output is not None and path.resolve() == output:
            continue
        included.append((relative, path))
    for relative, path in sorted(included):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"revision": revision, "dirty": dirty, "tree_sha256": digest.hexdigest()}


def run_fixed_benchmark(timeout_seconds: float | None = None, *, source_output_path: str | Path | None = None) -> dict[str, Any]:
    """Return scenario runs; an optional GA budget applies independently to each seed/scenario."""
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be positive and finite")
    distance_matrix, point_to_index = _benchmark_distances()
    source = _source_identity(source_output_path)
    scenarios: dict[str, Any] = {}
    for scenario_name, multiplier in SCENARIO_MULTIPLIERS:
        waste = _scenario_waste(scenario_name, multiplier)
        baseline_distance, baseline_routes = greedy_baseline(
            list(BENCHMARK_STOPS), BENCHMARK_VEHICLES, waste_dict=waste, road_graph=None,
            dist_matrix=distance_matrix, pt_to_idx=point_to_index, vehicle_capacity_kg=BENCHMARK_CAPACITY_KG,
        )
        baseline_feasibility = _route_feasibility(baseline_routes, waste)
        runs: list[dict[str, Any]] = []
        for seed in BENCHMARK_SEEDS:
            status: dict[str, bool] = {}
            started = time.perf_counter()
            routes, _, total_distance, _ = run_ga(
                list(BENCHMARK_STOPS), n_vehicles=BENCHMARK_VEHICLES, pop_size=BENCHMARK_POPULATION,
                generations=BENCHMARK_GENERATIONS, mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                vehicle_capacity_kg=BENCHMARK_CAPACITY_KG, waste_dict=waste, road_graph=None,
                distance_matrix=distance_matrix, point_to_index=point_to_index, random_seed=seed,
                timeout_seconds=timeout_seconds, run_status=status,
                early_stopping_patience=EARLY_STOP_PATIENCE,
                early_stopping_min_delta=EARLY_STOP_MIN_DELTA,
            )
            runtime_seconds = time.perf_counter() - started
            feasibility = _route_feasibility(routes, waste)
            runs.append({
                "seed": seed, "route_distance_km": round(float(total_distance), 2),
                "runtime_seconds": round(runtime_seconds, 6), "feasible": _is_feasible(feasibility),
                "feasibility": feasibility, "timeout_seconds": timeout_seconds,
                "timed_out": status.get("timed_out", False),
            })
        distances = [run["route_distance_km"] for run in runs]
        runtimes = [run["runtime_seconds"] for run in runs]
        optimizer = {
            "name": "genetic algorithm with capacity-aware decoding and 2-opt", "runs": runs,
            "mean_route_distance_km": round(statistics.mean(distances), 2),
            "distance_stddev_km": round(statistics.pstdev(distances), 2),
            "distance_range_km": [min(distances), max(distances)],
            "mean_runtime_seconds": round(statistics.mean(runtimes), 6),
            "best_route_distance_km": min(distances), "all_runs_feasible": all(run["feasible"] for run in runs),
            "summary": {
                "feasibility_rate": sum(run["feasible"] for run in runs) / len(runs),
                "mean_distance_km": round(statistics.mean(distances), 2),
                "distance_range_km": [min(distances), max(distances)],
                "mean_runtime_seconds": round(statistics.mean(runtimes), 6),
                "timeout_rate": sum(run["timed_out"] for run in runs) / len(runs),
            },
        }
        descriptor = {
            "name": scenario_name, "multiplier": multiplier,
            "rounding_rule": DEMAND_ROUNDING_RULE, "waste_kg_by_barangay": waste,
        }
        coordinates = {name: list(DEFAULT_BARANGAY_COORDS[name]) for name in BENCHMARK_STOPS}
        scenarios[scenario_name] = {
            **descriptor,
            "input_fingerprint": _scenario_fingerprint(
                descriptor, selected_stops=list(BENCHMARK_STOPS), coordinates=coordinates,
                depot={"name": DEPOT, "coordinates": list(DEPOT_COORD)}, seeds=list(BENCHMARK_SEEDS),
                vehicles=BENCHMARK_VEHICLES, capacity_kg=BENCHMARK_CAPACITY_KG,
                population=BENCHMARK_POPULATION, generations=BENCHMARK_GENERATIONS,
                mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                early_stopping_patience=EARLY_STOP_PATIENCE,
                early_stopping_min_delta=EARLY_STOP_MIN_DELTA,
                timeout_seconds=timeout_seconds,
            ),
            "baseline": {"name": "capacity-aware greedy nearest-stop heuristic", "route_distance_km": round(float(baseline_distance), 2), "feasible": _is_feasible(baseline_feasibility), "feasibility": baseline_feasibility},
            "optimizer": optimizer,
        }
    base = scenarios["base"]
    return {
        "schema_version": 2, "source": source,
        "runtime_environment": {"python": platform.python_version(), "numpy": np.__version__},
        "benchmark": "fixed-seed GA vs. greedy baseline demand sensitivity",
        "distance_model": "Haversine from built-in default coordinates; saved coordinate edits and live routing are ignored",
        "data_note": "Bundled waste values are illustrative estimates, not validated operational measurements.",
        "configuration": {
            "selected_barangays": list(BENCHMARK_STOPS), "waste_kg_by_barangay": base["waste_kg_by_barangay"],
            "vehicles": BENCHMARK_VEHICLES, "vehicle_capacity_kg": BENCHMARK_CAPACITY_KG,
            "population": BENCHMARK_POPULATION, "generations": BENCHMARK_GENERATIONS,
            "mutation_rate": MUTATION_RATE, "crossover_rate": CROSSOVER_RATE, "seeds": list(BENCHMARK_SEEDS),
            "early_stopping_patience": EARLY_STOP_PATIENCE,
            "early_stopping_min_delta": EARLY_STOP_MIN_DELTA,
            "timeout_seconds_per_ga_run": timeout_seconds,
            "timeout_semantics": "The per-run deadline starts at invocation; setup before the first generation check and an active generation cannot be interrupted, so runtime may exceed the budget in either case. The best-so-far route is retained.",
        },
        "baseline": base["baseline"], "optimizer": base["optimizer"], "scenarios": scenarios,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fixed-input route sensitivity benchmark.")
    parser.add_argument("--json-out", help="Write the report to a JSON file; stdout is the default.")
    parser.add_argument("--timeout-seconds", type=float, help="Optional positive finite cooperative budget per scenario/seed GA run.")
    args = parser.parse_args(argv)
    if args.timeout_seconds is not None and (not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0):
        parser.error("--timeout-seconds must be positive and finite")
    rendered = json.dumps(run_fixed_benchmark(args.timeout_seconds, source_output_path=args.json_out), indent=2, sort_keys=True) + "\n"
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
