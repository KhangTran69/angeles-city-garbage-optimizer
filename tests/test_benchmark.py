import numpy as np

from src import data
from src.benchmark import _benchmark_distances, run_fixed_benchmark


def test_fixed_benchmark_is_deterministic_and_reports_comparable_feasibility() -> None:
    first = run_fixed_benchmark()
    second = run_fixed_benchmark()

    assert first["schema_version"] == second["schema_version"] == 1
    assert first["configuration"] == second["configuration"]
    assert first["baseline"] == second["baseline"]
    assert [
        {key: value for key, value in run.items() if key != "runtime_seconds"}
        for run in first["optimizer"]["runs"]
    ] == [
        {key: value for key, value in run.items() if key != "runtime_seconds"}
        for run in second["optimizer"]["runs"]
    ]
    assert "Haversine" in first["distance_model"]
    assert first["configuration"]["seeds"] == [42, 43, 44]
    assert first["baseline"]["name"]
    assert first["baseline"]["route_distance_km"] >= 0
    assert first["baseline"]["feasible"] is True
    assert first["optimizer"]["all_runs_feasible"] is True
    assert first["runtime_environment"]["python"]
    assert first["runtime_environment"]["numpy"]
    assert first["optimizer"]["distance_range_km"] == [36.32, 41.21]
    assert first["optimizer"]["distance_stddev_km"] > 0
    assert first["optimizer"]["mean_runtime_seconds"] >= 0
    assert [run["seed"] for run in first["optimizer"]["runs"]] == [42, 43, 44]
    assert all(run["route_distance_km"] >= 0 for run in first["optimizer"]["runs"])
    assert all(run["runtime_seconds"] >= 0 for run in first["optimizer"]["runs"])
    assert all(run["feasibility"]["all_selected_stops_once"] for run in first["optimizer"]["runs"])
    assert all(run["feasibility"]["all_routes_within_capacity"] for run in first["optimizer"]["runs"])


def test_benchmark_distances_ignore_user_saved_coordinate_edits(monkeypatch) -> None:
    expected, expected_index = _benchmark_distances()
    monkeypatch.setitem(data.BARANGAY_COORDS, "Amsic", (0.0, 0.0))

    actual, actual_index = _benchmark_distances()

    assert actual_index == expected_index
    np.testing.assert_array_equal(actual, expected)
