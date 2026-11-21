import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src import data
from src.benchmark import (
    BENCHMARK_CAPACITY_KG,
    BENCHMARK_GENERATIONS,
    BENCHMARK_POPULATION,
    BENCHMARK_SEEDS,
    BENCHMARK_STOPS,
    BENCHMARK_VEHICLES,
    CROSSOVER_RATE,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    MUTATION_RATE,
    _benchmark_distances,
    _evaluation_case,
    _scenario_fingerprint,
    _scenario_waste,
    _source_identity,
    run_fixed_benchmark,
)


def test_scenarios_scale_values_round_half_up_without_mutating_defaults() -> None:
    original = data.BARANGAY_WASTE_KG.copy()
    scenarios = [_scenario_waste(name, multiplier) for name, multiplier in (("minus_10_percent", 0.9), ("base", 1.0), ("plus_10_percent", 1.1))]
    assert list(scenarios[1]) == list(BENCHMARK_STOPS)
    assert scenarios[1] == {name: original[name] for name in BENCHMARK_STOPS}
    assert scenarios[0] == {name: int(original[name] * 0.9 + 0.5) for name in BENCHMARK_STOPS}
    assert scenarios[2] == {name: int(original[name] * 1.1 + 0.5) for name in BENCHMARK_STOPS}
    assert _scenario_waste("tie", 1.0, {"Test": 1}) == {"Test": 1}
    assert _scenario_waste("tie", 1.5, {"Test": 1}) == {"Test": 2}
    assert data.BARANGAY_WASTE_KG == original


def test_fingerprint_covers_all_settings_and_timeout() -> None:
    base = {"name": "base", "multiplier": 1.0, "rounding_rule": "nearest_integer_ties_up", "waste_kg_by_barangay": {"Amsic": 10}}
    matrix_identity = {"source": "synthetic test matrix", "type": "test", "units": "km", "labels": ["Depot", "Amsic"], "identity_sha256": "0" * 64}
    args = dict(selected_stops=["Amsic"], coordinates={"Amsic": [1.0, 2.0]}, depot=[0.0, 0.0], seeds=[42], vehicles=3, capacity_kg=10_000, population=20, generations=30, mutation_rate=0.05, crossover_rate=0.85, early_stopping_patience=30, early_stopping_min_delta=0.05, timeout_seconds=None, distance_matrix=matrix_identity)
    fingerprint = _scenario_fingerprint(base, **args)
    assert fingerprint == _scenario_fingerprint(base, **args)
    changed_matrix = dict(args, distance_matrix={**matrix_identity, "identity_sha256": "1" * 64})
    assert _scenario_fingerprint(base, **changed_matrix) != fingerprint
    for setting, value in (("mutation_rate", 0.06), ("crossover_rate", 0.86), ("early_stopping_patience", 31), ("early_stopping_min_delta", 0.06), ("population", 21), ("generations", 31), ("vehicles", 4), ("capacity_kg", 9999), ("seeds", [43]), ("timeout_seconds", 1.0)):
        changed = dict(args)
        changed[setting] = value
        assert _scenario_fingerprint(base, **changed) != fingerprint
    for setting, value in (("multiplier", 0.9), ("rounding_rule", "floor")):
        changed_scenario = {**base, setting: value}
        assert _scenario_fingerprint(changed_scenario, **args) != fingerprint


def test_report_has_scenario_runs_and_source_identity(monkeypatch) -> None:
    from src import benchmark

    early_stop_settings = []
    original_run_ga = benchmark.run_ga

    def recording_run_ga(*args, **kwargs):
        early_stop_settings.append((kwargs["early_stopping_patience"], kwargs["early_stopping_min_delta"]))
        return original_run_ga(*args, **kwargs)

    monkeypatch.setattr(benchmark, "run_ga", recording_run_ga)
    report = run_fixed_benchmark(timeout_seconds=0.5)
    assert report["schema_version"] == 4
    assert report["distance_matrix"]["type"] == "haversine"
    assert report["limitations"]
    revision = report["source"]["revision"]
    dirty = report["source"]["dirty"]
    if revision is None:
        assert dirty is None
    else:
        assert isinstance(revision, str) and revision
        assert isinstance(dirty, bool)
    tree_hash = report["source"]["tree_sha256"]
    assert len(tree_hash) == 64 and all(character in "0123456789abcdef" for character in tree_hash)
    assert report["configuration"]["selected_barangays"] == list(BENCHMARK_STOPS)
    assert report["configuration"]["seeds"] == list(BENCHMARK_SEEDS)
    assert report["configuration"]["vehicles"] == BENCHMARK_VEHICLES
    assert report["configuration"]["vehicle_capacity_kg"] == BENCHMARK_CAPACITY_KG
    assert report["configuration"]["population"] == BENCHMARK_POPULATION
    assert report["configuration"]["generations"] == BENCHMARK_GENERATIONS
    assert report["configuration"]["mutation_rate"] == MUTATION_RATE
    assert report["configuration"]["crossover_rate"] == CROSSOVER_RATE
    assert report["configuration"]["early_stopping_patience"] == EARLY_STOP_PATIENCE
    assert report["configuration"]["early_stopping_min_delta"] == EARLY_STOP_MIN_DELTA
    assert len(early_stop_settings) == len(BENCHMARK_SEEDS) * 3 + 3
    assert set(early_stop_settings) == {(EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA)}
    assert report["configuration"]["timeout_semantics"] == (
        "The per-run deadline starts at invocation; setup before the first generation check and an active "
        "generation cannot be interrupted, so runtime may exceed the budget in either case. "
        "The best-so-far route is retained."
    )
    scenarios = report["scenarios"]
    assert list(scenarios) == ["minus_10_percent", "base", "plus_10_percent"]
    assert len({scenario["input_fingerprint"] for scenario in scenarios.values()}) == 3
    for scenario in scenarios.values():
        assert len(scenario["optimizer"]["runs"]) == 3
        assert scenario["baseline"]["route_distance_km"] >= 0
        assert scenario["baseline"]["feasibility"]
        for seed, run in zip(BENCHMARK_SEEDS, scenario["optimizer"]["runs"], strict=True):
            assert run["seed"] == seed
            assert len(run["routes"]) == BENCHMARK_VEHICLES
            assert sorted(stop for route in run["routes"] for stop in route) == sorted(BENCHMARK_STOPS)
            assert run["route_distance_km"] >= 0
            assert run["runtime_seconds"] >= 0
            assert run["timeout_seconds"] == 0.5
            assert isinstance(run["timed_out"], bool)
            assert "feasible" in run
        assert "feasibility_rate" in scenario["optimizer"]["summary"]
        assert "timeout_rate" in scenario["optimizer"]["summary"]


def test_fixed_benchmark_uses_external_matrix_and_reports_provenance() -> None:
    from src import benchmark

    labels = [benchmark.DEPOT, *benchmark.BARANGAYS]
    matrix = np.full((len(labels), len(labels)), 0.25)
    np.fill_diagonal(matrix, 0.0)
    report = run_fixed_benchmark(
        distance_matrix=matrix, distance_labels=labels,
        distance_source="deterministic synthetic test matrix", distance_type="synthetic_road_km",
        distance_attribution="© example contributors", distance_license="ODbL",
    )
    assert report["distance_matrix"]["source"] == "deterministic synthetic test matrix"
    assert report["distance_matrix"]["type"] == "synthetic_road_km"
    assert report["distance_matrix"]["units"] == "km"
    assert report["distance_matrix"]["attribution"] == "© example contributors"
    assert report["distance_matrix"]["license"] == "ODbL"
    assert len(report["distance_matrix"]["identity_sha256"]) == 64
    assert report["distance_matrix"]["identity_sha256"] == benchmark._distance_matrix_identity(matrix, labels)
    assert not any("OSM-derived" in item for item in report["limitations"])
    for scenario in report["scenarios"].values():
        assert scenario["baseline"]["route_distance_km"] < 3.0
        assert all(run["route_distance_km"] < 3.0 for run in scenario["optimizer"]["runs"])
        assert all(run["feasible"] for run in scenario["optimizer"]["runs"])


def test_distance_model_limitations_match_matrix_provenance() -> None:
    from src.benchmark import _distance_model_limitation

    assert "Haversine" in _distance_model_limitation("haversine")
    assert "OSM-derived" not in _distance_model_limitation("haversine")
    assert "OSM-derived" in _distance_model_limitation("osmnx_drive_directed_shortest_path_km")
    assert "OSM-derived" not in _distance_model_limitation("synthetic_road_km")


@pytest.mark.parametrize("matrix,labels", [
    (np.zeros((2, 2)), ["Depot", "Amsic"]),
    (np.zeros((33, 33)), [data.DEPOT, *data.BARANGAYS[:-1]]),
    (np.full((34, 34), np.nan), [data.DEPOT, *data.BARANGAYS]),
])
def test_external_matrix_rejects_incomplete_or_invalid_data(matrix, labels) -> None:
    with pytest.raises(ValueError):
        run_fixed_benchmark(
            distance_matrix=matrix, distance_labels=labels,
            distance_source="fixture", distance_type="test_road_km",
        )


def test_cli_loads_external_matrix_and_records_attribution(tmp_path) -> None:
    labels = [data.DEPOT, *data.BARANGAYS]
    matrix = np.full((len(labels), len(labels)), 0.25)
    np.fill_diagonal(matrix, 0.0)
    matrix_path = tmp_path / "matrix.npz"
    report_path = tmp_path / "report.json"
    np.savez_compressed(matrix_path, matrix=matrix, labels=np.asarray(labels, dtype=str))
    result = subprocess.run([
        sys.executable, "-m", "src.benchmark", "--distance-matrix", str(matrix_path),
        "--distance-source", "fixture", "--distance-type", "test_road_km",
        "--distance-attribution", "© example contributors", "--distance-license", "ODbL",
        "--json-out", str(report_path),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["distance_matrix"]["source"] == "fixture"
    assert report["distance_matrix"]["attribution"] == "© example contributors"
    assert report["distance_matrix"]["license"] == "ODbL"
    assert report["distance_matrix"]["identity_sha256"]
    assert report["scenarios"]["base"]["baseline"]["route_distance_km"] < 3


def test_cli_reports_corrupt_external_matrix_without_traceback(tmp_path) -> None:
    matrix_path = tmp_path / "corrupt.npz"
    labels = [data.DEPOT, *data.BARANGAYS]
    matrix = np.zeros((len(labels), len(labels)))
    np.savez_compressed(matrix_path, matrix=matrix, labels=np.asarray(labels, dtype=str))
    damaged = bytearray(matrix_path.read_bytes())
    central_directory_offset = damaged.find(b"PK\x01\x02")
    assert central_directory_offset >= 0
    damaged[central_directory_offset : central_directory_offset + 4] = b"NO\x01\x02"
    matrix_path.write_bytes(damaged)

    with pytest.raises(zipfile.BadZipFile, match="central directory"):
        np.load(matrix_path, allow_pickle=False)

    result = subprocess.run([
        sys.executable, "-m", "src.benchmark", "--distance-matrix", str(matrix_path),
        "--distance-source", "fixture", "--distance-type", "test_road_km",
    ], capture_output=True, text=True)

    assert result.returncode != 0
    assert "could not load --distance-matrix" in result.stderr
    assert "BadZipFile" not in result.stderr
    assert "Traceback" not in result.stderr


def test_report_runtime_environment_includes_reference_hardware(monkeypatch) -> None:
    from src import benchmark

    monkeypatch.setattr(benchmark.platform, "system", lambda: "TestOS")
    monkeypatch.setattr(benchmark.platform, "release", lambda: "1.2.3")
    monkeypatch.setattr(benchmark.platform, "version", lambda: "test build")
    monkeypatch.setattr(benchmark.platform, "machine", lambda: "test-arch")
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "Test CPU family")
    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    environment = run_fixed_benchmark()["runtime_environment"]

    assert environment["python"] == sys.version.split()[0]
    assert environment["numpy"] == np.__version__
    assert environment["operating_system"] == {
        "name": "TestOS",
        "release": "1.2.3",
        "version": "test build",
    }
    assert environment["architecture"] == "test-arch"
    assert environment["cpu_family"] == "Test CPU family"
    assert environment["logical_cpu_count"] == 8


def test_evaluation_case_does_not_misclassify_solver_value_error(monkeypatch) -> None:
    from src import benchmark

    distance_matrix, point_to_index = _benchmark_distances()

    def fail_during_solver(*args, **kwargs):
        raise ValueError("solver failed after input validation")

    monkeypatch.setattr(benchmark, "run_ga", fail_during_solver)
    with pytest.raises(ValueError, match="solver failed after input validation"):
        _evaluation_case(
            "capacity_constrained", ["Amsic"], {"Amsic": 1000}, 1, 1000,
            distance_matrix, point_to_index,
        )


def test_evaluation_case_distinguishes_rejected_infeasibility_from_invalid_input() -> None:
    distance_matrix, point_to_index = _benchmark_distances()

    result = _evaluation_case(
        "infeasible_demand", ["Amsic", "Anunas", "Balibago"],
        {"Amsic": 700, "Anunas": 700, "Balibago": 700}, 2, 1000,
        distance_matrix, point_to_index,
    )

    assert result["outcome"] == "INFEASIBLE_INPUT"
    assert result["input_validation"]["valid"] is True
    assert result["input_validation"]["feasible"] is False
    assert result["input_validation"]["reason_codes"] == [
        "total_demand_exceeds_fleet_capacity",
    ]
    assert result["input_validation"]["details"]["fleet_capacity_kg"] == 2000
    assert result["validation_runtime_seconds"] >= 0
    assert result["solver_result"] is None


def test_report_includes_reproducible_capacity_and_input_outcomes() -> None:
    report = run_fixed_benchmark(timeout_seconds=1e-12)

    cases = report["evaluation_cases"]
    assert set(cases) == {
        "capacity_constrained",
        "infeasible_demand",
        "invalid_input",
        "oversized_stop",
    }
    assert len({case["input_fingerprint"] for case in cases.values()}) == 4
    assert all(len(case["input_fingerprint"]) == 64 for case in cases.values())

    capacity = cases["capacity_constrained"]
    assert capacity["input"] == {
        "selected_barangays": ["Amsic", "Anunas"],
        "waste_kg_by_barangay": {"Amsic": 1000, "Anunas": 1000},
        "vehicles": 2,
        "vehicle_capacity_kg": 1000,
        "distance_model": "Haversine from built-in default coordinates",
        "coordinates": {
            "Amsic": list(data.DEFAULT_BARANGAY_COORDS["Amsic"]),
            "Anunas": list(data.DEFAULT_BARANGAY_COORDS["Anunas"]),
        },
        "depot": {"name": data.DEPOT, "coordinates": list(data.DEPOT_COORD)},
    }
    assert capacity["configuration"] == {
        "seed": 42,
        "population": 4,
        "generations": 2,
        "mutation_rate": MUTATION_RATE,
        "crossover_rate": CROSSOVER_RATE,
        "early_stopping_patience": EARLY_STOP_PATIENCE,
        "early_stopping_min_delta": EARLY_STOP_MIN_DELTA,
        "timeout_seconds": None,
        "allow_infeasible": False,
    }
    expected_cases = {
        "capacity_constrained": (
            ["Amsic", "Anunas"], {"Amsic": 1000, "Anunas": 1000}, 2, 1000,
        ),
        "infeasible_demand": (
            ["Amsic", "Anunas", "Balibago"],
            {"Amsic": 700, "Anunas": 700, "Balibago": 700}, 2, 1000,
        ),
        "invalid_input": (["Unknown Zone"], {"Unknown Zone": 100}, 1, 1000),
        "oversized_stop": (["Amsic"], {"Amsic": 1001}, 1, 1000),
    }
    for name, (stops, waste, vehicles, vehicle_capacity) in expected_cases.items():
        case = cases[name]
        inputs = case["input"]
        assert {
            "selected_barangays": inputs["selected_barangays"],
            "waste_kg_by_barangay": inputs["waste_kg_by_barangay"],
            "vehicles": inputs["vehicles"],
            "vehicle_capacity_kg": inputs["vehicle_capacity_kg"],
        } == {
            "selected_barangays": stops,
            "waste_kg_by_barangay": waste,
            "vehicles": vehicles,
            "vehicle_capacity_kg": vehicle_capacity,
        }
        assert inputs["coordinates"] == {
            stop: list(data.DEFAULT_BARANGAY_COORDS[stop])
            for stop in stops
            if stop in data.DEFAULT_BARANGAY_COORDS
        }
        assert inputs["depot"] == {
            "name": data.DEPOT,
            "coordinates": list(data.DEPOT_COORD),
        }
        assert case["configuration"] == {
            **capacity["configuration"],
            "allow_infeasible": name in {"infeasible_demand", "oversized_stop"},
        }
        assert len(case["input_fingerprint"]) == 64
    assert capacity["outcome"] == "FEASIBLE_RESULT"
    assert capacity["solver_result"]["feasible"] is True
    assert sorted(capacity["solver_result"]["route_loads_kg"]) == [1000.0, 1000.0]

    infeasible = cases["infeasible_demand"]
    assert infeasible["input"]["waste_kg_by_barangay"] == {
        "Amsic": 700, "Anunas": 700, "Balibago": 700,
    }
    assert infeasible["outcome"] == "INFEASIBLE_DIAGNOSTIC"
    assert infeasible["input_validation"]["reason_codes"] == [
        "total_demand_exceeds_fleet_capacity",
    ]
    assert infeasible["input_validation"]["details"]["fleet_capacity_kg"] == 2000
    assert infeasible["solver_result"]["feasible"] is False

    invalid = cases["invalid_input"]
    assert invalid["outcome"] == "REJECTED_INVALID_INPUT"
    assert invalid["input_validation"]["error"] == {
        "type": "ValueError",
        "message": "unknown barangay name(s): Unknown Zone",
    }
    assert invalid["validation_runtime_seconds"] >= 0
    assert invalid["solver_result"] is None

    oversized = cases["oversized_stop"]
    assert oversized["outcome"] == "INFEASIBLE_DIAGNOSTIC"
    assert oversized["input_validation"]["reason_codes"] == [
        "stop_exceeds_vehicle_capacity",
        "total_demand_exceeds_fleet_capacity",
    ]
    assert oversized["input_validation"]["details"]["oversized_stops"] == [
        {"barangay": "Amsic", "waste_kg": 1001.0},
    ]
    assert oversized["solver_result"]["feasible"] is False


def test_evaluation_cases_remain_haversine_with_external_benchmark_matrix() -> None:
    labels = [data.DEPOT, *data.BARANGAYS]
    matrix = np.full((len(labels), len(labels)), 10.0)
    np.fill_diagonal(matrix, 0.0)

    haversine_case = run_fixed_benchmark()["evaluation_cases"]["capacity_constrained"]
    external_matrix_report = run_fixed_benchmark(
        distance_matrix=matrix,
        distance_labels=labels,
        distance_source="distinct test matrix",
        distance_type="synthetic_test_km",
    )
    external_matrix_case = external_matrix_report["evaluation_cases"]["capacity_constrained"]

    assert external_matrix_case["input"]["distance_model"] == "Haversine from built-in default coordinates"
    assert external_matrix_case["solver_result"]["route_distance_km"] == (
        haversine_case["solver_result"]["route_distance_km"]
    )


def test_tiny_cooperative_timeout_keeps_best_so_far_metrics() -> None:
    report = run_fixed_benchmark(timeout_seconds=1e-12)
    for scenario in report["scenarios"].values():
        for run in scenario["optimizer"]["runs"]:
            assert run["timed_out"] is True
            assert run["route_distance_km"] >= 0
            assert isinstance(run["feasible"], bool)
            assert run["feasibility"]


def test_capacity_feasibility_uses_scenario_values() -> None:
    from src.benchmark import _route_feasibility
    assert _route_feasibility([["Amsic"]], {"Amsic": BENCHMARK_CAPACITY_KG + 1})["all_routes_within_capacity"] is False
    assert _route_feasibility([["Amsic"]], {"Amsic": BENCHMARK_CAPACITY_KG})["all_routes_within_capacity"] is True


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_timeout(value: str) -> None:
    result = subprocess.run([sys.executable, "-m", "src.benchmark", "--timeout-seconds", value], capture_output=True, text=True)
    assert result.returncode == 2
    assert "positive and finite" in result.stderr


def test_cli_emits_json_with_timeout(tmp_path) -> None:
    output = tmp_path / "report.json"
    result = subprocess.run([sys.executable, "-m", "src.benchmark", "--timeout-seconds", "0.5", "--json-out", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["schema_version"] == 4
    assert report["evaluation_cases"]["capacity_constrained"]["outcome"] == "FEASIBLE_RESULT"
    assert report["evaluation_cases"]["invalid_input"]["outcome"] == "REJECTED_INVALID_INPUT"
    assert report["configuration"]["timeout_seconds_per_ga_run"] == 0.5
    assert report["scenarios"]["base"]["optimizer"]["runs"][0]["timeout_seconds"] == 0.5


def test_source_hash_changes_for_untracked_source_and_documentation(tmp_path, monkeypatch) -> None:
    from src import benchmark

    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("tracked source tree\n")
    benchmark_file = root / "src" / "benchmark.py"
    benchmark_file.write_text("# benchmark\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md", "src/benchmark.py"], cwd=root, check=True)
    monkeypatch.setattr(benchmark, "__file__", str(benchmark_file))

    before = _source_identity()["tree_sha256"]
    (root / "docs").mkdir()
    (root / "docs" / "release-notes.md").write_text("untracked documentation\n")
    after = _source_identity()["tree_sha256"]
    assert after != before


def test_source_identity_ignores_exact_output_for_hash_and_git_dirty_state(tmp_path, monkeypatch) -> None:
    from src import benchmark

    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    benchmark_file = root / "src" / "benchmark.py"
    benchmark_file.write_text("# benchmark\n")
    readme = root / "README.md"
    readme.write_text("tracked\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md", "src/benchmark.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)
    monkeypatch.setattr(benchmark, "__file__", str(benchmark_file))

    output = root / "report.json"
    before = _source_identity(output)
    assert before["dirty"] is False
    output.write_text("generated report\n")
    after_output = _source_identity(output)
    assert after_output == before

    docs = root / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("untracked documentation\n")
    after_docs = _source_identity(output)
    assert after_docs["dirty"] is True
    assert after_docs["tree_sha256"] != before["tree_sha256"]


def test_source_hash_is_stable_and_excludes_same_json_output_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="benchmark-report-", dir=project_root) as temp_dir:
        output = Path(temp_dir) / "report.json"
        command = [sys.executable, "-m", "src.benchmark", "--timeout-seconds", "1e-12", "--json-out", str(output)]
        first = subprocess.run(command, cwd=project_root, capture_output=True, text=True)
        assert first.returncode == 0, first.stderr
        first_hash = json.loads(output.read_text())["source"]["tree_sha256"]

        second = subprocess.run(command, cwd=project_root, capture_output=True, text=True)
        assert second.returncode == 0, second.stderr
        second_hash = json.loads(output.read_text())["source"]["tree_sha256"]

        assert _source_identity(output)["tree_sha256"] == second_hash
        assert second_hash == first_hash


def test_source_identity_falls_back_safely_outside_git(tmp_path, monkeypatch) -> None:
    from src import benchmark

    root = tmp_path / "plain-project"
    source_dir = root / "src"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "benchmark.py"
    source_file.write_text("source one\n")
    monkeypatch.setattr(benchmark, "__file__", str(source_file))

    first = _source_identity()
    repeated = _source_identity()
    assert first["revision"] is None
    assert first["dirty"] is None
    assert len(first["tree_sha256"]) == 64
    assert all(character in "0123456789abcdef" for character in first["tree_sha256"])
    assert repeated == first

    source_file.write_text("source two\n")
    changed = _source_identity()
    assert changed["revision"] is None
    assert changed["dirty"] is None
    assert changed["tree_sha256"] != first["tree_sha256"]


def test_benchmark_distances_ignore_user_saved_coordinate_edits(monkeypatch) -> None:
    expected, expected_index = _benchmark_distances()
    monkeypatch.setitem(data.BARANGAY_COORDS, "Amsic", (0.0, 0.0))
    actual, actual_index = _benchmark_distances()
    assert actual_index == expected_index
    np.testing.assert_array_equal(actual, expected)
