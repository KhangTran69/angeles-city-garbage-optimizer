"""Fixed-input sensitivity comparison for the route optimizer and greedy baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.data import BARANGAY_WASTE_KG, BARANGAYS, DEFAULT_BARANGAY_COORDS, DEPOT, DEPOT_COORD
from src.optimization import (
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    InfeasibleRoutingProblem,
    greedy_baseline,
    run_ga,
    validate_ga_inputs,
)
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
    distance_matrix: dict[str, Any],
) -> str:
    payload = {
        "scenario": scenario["name"], "multiplier": scenario["multiplier"],
        "rounding_rule": scenario["rounding_rule"],
        "waste_kg_by_barangay": scenario["waste_kg_by_barangay"],
        "selected_stops": selected_stops, "coordinates": coordinates, "depot": depot,
        "distance_matrix": distance_matrix, "seeds": seeds,
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


def _cpu_family() -> str | None:
    processor = platform.processor().strip()
    generic_processors = {
        "arm", "arm64", "aarch64", "amd64", "i386", "i686", "x86_64", "unknown",
    }
    if processor and processor.casefold() not in generic_processors | {platform.machine().casefold()}:
        return processor

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True, timeout=1,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return processor or None
        return result.stdout.strip() or processor or None
    if platform.system() == "Linux":
        try:
            cpu_info = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return processor or None
        entries: dict[str, str] = {}
        for line in cpu_info.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().casefold() in {"model name", "hardware", "processor"}:
                entries.setdefault(key.strip().casefold(), value.strip())
        for key in ("model name", "hardware", "processor"):
            value = entries.get(key)
            if value and not value.isdigit():
                return value
    if platform.system() == "Windows":
        return os.environ.get("PROCESSOR_IDENTIFIER") or processor or None
    return processor or None


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "operating_system": {
            "name": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "architecture": platform.machine(),
        "cpu_family": _cpu_family(),
        "logical_cpu_count": os.cpu_count(),
    }


def _evaluation_case(
    name: str,
    selected_stops: list[str],
    waste_kg_by_barangay: dict[str, int],
    vehicles: int,
    capacity_kg: int,
    distance_matrix: np.ndarray,
    point_to_index: dict[str, int],
    *,
    allow_infeasible: bool = False,
) -> dict[str, Any]:
    seed = 42
    solver_configuration = {
        "seed": seed,
        "population": 4,
        "generations": 2,
        "mutation_rate": MUTATION_RATE,
        "crossover_rate": CROSSOVER_RATE,
        "early_stopping_patience": EARLY_STOP_PATIENCE,
        "early_stopping_min_delta": EARLY_STOP_MIN_DELTA,
        "timeout_seconds": None,
        "allow_infeasible": allow_infeasible,
    }
    inputs = {
        "selected_barangays": selected_stops,
        "waste_kg_by_barangay": waste_kg_by_barangay,
        "vehicles": vehicles,
        "vehicle_capacity_kg": capacity_kg,
        "distance_model": "Haversine from built-in default coordinates",
        "coordinates": {
            stop: list(DEFAULT_BARANGAY_COORDS[stop])
            for stop in selected_stops
            if stop in DEFAULT_BARANGAY_COORDS
        },
        "depot": {"name": DEPOT, "coordinates": list(DEPOT_COORD)},
    }
    fingerprint_payload = {
        "name": name,
        "input": inputs,
        "solver": solver_configuration,
    }
    canonical = json.dumps(
        fingerprint_payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    input_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    validation_started = time.perf_counter()
    try:
        validate_ga_inputs(
            selected_stops,
            n_vehicles=vehicles,
            pop_size=solver_configuration["population"],
            generations=solver_configuration["generations"],
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
            vehicle_capacity_kg=capacity_kg,
            waste_dict=waste_kg_by_barangay,
            timeout_seconds=solver_configuration["timeout_seconds"],
            early_stopping_patience=EARLY_STOP_PATIENCE,
            early_stopping_min_delta=EARLY_STOP_MIN_DELTA,
            allow_infeasible=allow_infeasible,
        )
    except InfeasibleRoutingProblem as error:
        elapsed = time.perf_counter() - validation_started
        return {
            "input": inputs,
            "configuration": solver_configuration,
            "input_fingerprint": input_fingerprint,
            "outcome": "INFEASIBLE_INPUT",
            "input_validation": {
                "valid": True,
                "feasible": False,
                "reason_codes": error.reason_codes,
                "details": error.details,
            },
            "solver_result": None,
            "validation_runtime_seconds": round(elapsed, 6),
        }
    except ValueError as error:
        elapsed = time.perf_counter() - validation_started
        return {
            "input": inputs,
            "configuration": solver_configuration,
            "input_fingerprint": input_fingerprint,
            "outcome": "REJECTED_INVALID_INPUT",
            "input_validation": {
                "valid": False,
                "feasible": None,
                "reason_codes": [],
                "details": {},
                "error": {"type": type(error).__name__, "message": str(error)},
            },
            "solver_result": None,
            "validation_runtime_seconds": round(elapsed, 6),
        }

    status: dict[str, Any] = {}
    started = time.perf_counter()
    routes, _, total_distance, _ = run_ga(
        selected_stops,
        n_vehicles=vehicles,
        pop_size=solver_configuration["population"],
        generations=solver_configuration["generations"],
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        vehicle_capacity_kg=capacity_kg,
        waste_dict=waste_kg_by_barangay,
        road_graph=None,
        distance_matrix=distance_matrix,
        point_to_index=point_to_index,
        random_seed=seed,
        run_status=status,
        early_stopping_patience=EARLY_STOP_PATIENCE,
        early_stopping_min_delta=EARLY_STOP_MIN_DELTA,
        allow_infeasible=allow_infeasible,
    )
    elapsed = time.perf_counter() - started
    input_validation = {
        "valid": True,
        "feasible": status["input_feasible"],
        "reason_codes": status["input_reason_codes"],
        "details": status["input_details"],
    }
    solver_result = {
        "feasible": status["feasible"],
        "reason_codes": status["reason_codes"],
        "overloaded_routes": status["overloaded_routes"],
        "route_count": len(routes),
        "route_loads_kg": [float(get_route_load(route, waste_kg_by_barangay)) for route in routes],
        "route_distance_km": round(float(total_distance), 2),
        "runtime_seconds": round(elapsed, 6),
    }
    return {
        "input": inputs,
        "configuration": solver_configuration,
        "input_fingerprint": input_fingerprint,
        "outcome": "FEASIBLE_RESULT" if status["feasible"] else "INFEASIBLE_DIAGNOSTIC",
        "input_validation": input_validation,
        "solver_result": solver_result,
    }


def _fixed_evaluation_cases() -> dict[str, dict[str, Any]]:
    distance_matrix, point_to_index = _benchmark_distances()
    specifications = (
        ("capacity_constrained", ["Amsic", "Anunas"], {"Amsic": 1000, "Anunas": 1000}, 2, 1000, False),
        ("infeasible_demand", ["Amsic", "Anunas", "Balibago"], {"Amsic": 700, "Anunas": 700, "Balibago": 700}, 2, 1000, True),
        ("invalid_input", ["Unknown Zone"], {"Unknown Zone": 100}, 1, 1000, False),
        ("oversized_stop", ["Amsic"], {"Amsic": 1001}, 1, 1000, True),
    )
    return {
        name: _evaluation_case(
            name, selected_stops, waste, vehicles, capacity,
            distance_matrix, point_to_index, allow_infeasible=allow_infeasible,
        )
        for name, selected_stops, waste, vehicles, capacity, allow_infeasible in specifications
    }


def _benchmark_distances() -> tuple[np.ndarray, dict[str, int]]:
    """Build a deterministic matrix from code defaults, ignoring editable saved coordinates."""
    points = [DEPOT, *BARANGAYS]
    coordinates = {DEPOT: DEPOT_COORD, **DEFAULT_BARANGAY_COORDS}
    point_to_index = {point: index for index, point in enumerate(points)}
    matrix = np.array([[haversine(coordinates[left], coordinates[right]) for right in points] for left in points], dtype=float)
    return matrix, point_to_index


def _distance_matrix_identity(matrix: np.ndarray, labels: list[str]) -> str:
    canonical_matrix = np.asarray(matrix, dtype=np.dtype("<f8"), order="C")
    header = json.dumps(
        {"labels": labels, "shape": list(canonical_matrix.shape), "units": "km"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header + b"\0")
    digest.update(canonical_matrix.tobytes(order="C"))
    return digest.hexdigest()


def _resolve_distance_matrix(
    matrix: np.ndarray | None, labels: list[str] | tuple[str, ...] | None,
    source: str | None, distance_type: str | None,
    attribution: str | None = None, license_name: str | None = None,
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    if matrix is None:
        if any(value is not None for value in (labels, source, distance_type, attribution, license_name)):
            raise ValueError("distance metadata requires a distance matrix")
        matrix, point_to_index = _benchmark_distances()
        matrix_labels = [DEPOT, *BARANGAYS]
        metadata: dict[str, Any] = {
            "source": "built-in default coordinates", "type": "haversine",
            "units": "km", "labels": matrix_labels,
        }
    else:
        if labels is None or not source or not distance_type:
            raise ValueError("an external matrix requires labels, distance_source, and distance_type")
        matrix_labels = list(labels)
        if any(not isinstance(label, str) or not label.strip() for label in matrix_labels):
            raise ValueError("distance matrix labels must be non-empty strings")
        if len(set(matrix_labels)) != len(matrix_labels):
            raise ValueError("distance matrix labels must be unique")
        missing = {DEPOT, *BARANGAYS}.difference(matrix_labels)
        if missing:
            raise ValueError(f"distance matrix is missing modeled points: {', '.join(sorted(missing))}")
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (len(matrix_labels), len(matrix_labels)):
            raise ValueError("distance matrix must be square and match the number of labels")
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("distance matrix values must be finite and non-negative")
        if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-9):
            raise ValueError("distance matrix diagonal must be zero")
        point_to_index = {label: index for index, label in enumerate(matrix_labels)}
        metadata = {
            "source": source.strip(), "type": distance_type.strip(),
            "units": "km", "labels": matrix_labels,
        }
        if attribution:
            metadata["attribution"] = attribution.strip()
        if license_name:
            metadata["license"] = license_name.strip()
        if not metadata["source"] or not metadata["type"]:
            raise ValueError("distance_source and distance_type must be non-empty")
    metadata["identity_sha256"] = _distance_matrix_identity(matrix, matrix_labels)
    return matrix, point_to_index, metadata


def _load_distance_matrix_npz(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load a numeric NPZ matrix bundle without enabling pickle deserialization."""
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != {"matrix", "labels"}:
            raise ValueError("distance matrix NPZ must contain exactly 'matrix' and 'labels'")
        matrix = np.asarray(bundle["matrix"], dtype=float)
        labels = [str(label) for label in bundle["labels"].tolist()]
    return matrix, labels


def _distance_model_limitation(distance_type: str) -> str:
    if distance_type == "haversine":
        return "Haversine distances are straight-line estimates, not road distances."
    if distance_type.startswith("osmnx_"):
        return "The OSM-derived matrix comes from the identified cached graph and does not establish current road conditions or operational suitability."
    return "The supplied distance matrix is not independently validated against field conditions or operational requirements."


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


def run_fixed_benchmark(
    timeout_seconds: float | None = None, *, source_output_path: str | Path | None = None,
    distance_matrix: np.ndarray | None = None,
    distance_labels: list[str] | tuple[str, ...] | None = None,
    distance_source: str | None = None,
    distance_type: str | None = None,
    distance_attribution: str | None = None,
    distance_license: str | None = None,
) -> dict[str, Any]:
    """Return scenario runs; an optional GA budget applies independently to each seed/scenario."""
    if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be positive and finite")
    distance_matrix, point_to_index, distance_provenance = _resolve_distance_matrix(
        distance_matrix, distance_labels, distance_source, distance_type,
        distance_attribution, distance_license,
    )
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
                "seed": seed, "routes": routes, "route_distance_km": round(float(total_distance), 2),
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
                distance_matrix=distance_provenance,
            ),
            "baseline": {"name": "capacity-aware greedy nearest-stop heuristic", "route_distance_km": round(float(baseline_distance), 2), "feasible": _is_feasible(baseline_feasibility), "feasibility": baseline_feasibility},
            "optimizer": optimizer,
        }
    base = scenarios["base"]
    return {
        "schema_version": 4, "source": source,
        "runtime_environment": _runtime_environment(),
        "benchmark": "fixed-seed GA vs. greedy baseline demand sensitivity",
        "distance_model": (
            "Haversine from built-in default coordinates; saved coordinate edits and live routing are ignored"
            if distance_provenance["type"] == "haversine"
            else f"{distance_provenance['type']} distance matrix in km"
        ),
        "distance_matrix": distance_provenance,
        "data_note": "Bundled waste values are illustrative estimates, not validated operational measurements.",
        "limitations": [
            "The fixture uses six actual named barangays as an illustrative subset; waste demand is synthetic and illustrative, not measured field input.",
            _distance_model_limitation(distance_provenance["type"]),
            "The genetic algorithm is a heuristic and does not guarantee a globally optimal route plan.",
            "These results do not establish municipal savings, city-wide route quality, or service performance.",
        ],
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
        "evaluation_cases": _fixed_evaluation_cases(),
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fixed-input route sensitivity benchmark.")
    parser.add_argument("--json-out", help="Write the report to a JSON file; stdout is the default.")
    parser.add_argument("--timeout-seconds", type=float, help="Optional positive finite cooperative budget per scenario/seed GA run.")
    parser.add_argument("--distance-matrix", help="Optional NPZ bundle containing `matrix` values in km and ordered `labels`.")
    parser.add_argument("--distance-source", help="Required provenance description when --distance-matrix is supplied.")
    parser.add_argument("--distance-type", help="Required distance model identifier when --distance-matrix is supplied.")
    parser.add_argument("--distance-attribution", help="Optional source attribution recorded in the report.")
    parser.add_argument("--distance-license", help="Optional source license recorded in the report.")
    args = parser.parse_args(argv)
    if args.timeout_seconds is not None and (not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0):
        parser.error("--timeout-seconds must be positive and finite")
    distance_matrix = None
    distance_labels = None
    if args.distance_matrix:
        if not args.distance_source or not args.distance_type:
            parser.error("--distance-matrix requires --distance-source and --distance-type")
        try:
            distance_matrix, distance_labels = _load_distance_matrix_npz(args.distance_matrix)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            parser.error(f"could not load --distance-matrix: {exc}")
    elif any((args.distance_source, args.distance_type, args.distance_attribution, args.distance_license)):
        parser.error("distance provenance options require --distance-matrix")
    try:
        report = run_fixed_benchmark(
            args.timeout_seconds, source_output_path=args.json_out,
            distance_matrix=distance_matrix, distance_labels=distance_labels,
            distance_source=args.distance_source, distance_type=args.distance_type,
            distance_attribution=args.distance_attribution, distance_license=args.distance_license,
        )
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
