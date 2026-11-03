import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src import data
from src.benchmark import (
    BENCHMARK_CAPACITY_KG,
    CROSSOVER_RATE,
    BENCHMARK_GENERATIONS,
    BENCHMARK_POPULATION,
    BENCHMARK_SEEDS,
    BENCHMARK_STOPS,
    BENCHMARK_VEHICLES,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    MUTATION_RATE,
    _benchmark_distances,
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
    args = dict(selected_stops=["Amsic"], coordinates={"Amsic": [1.0, 2.0]}, depot=[0.0, 0.0], seeds=[42], vehicles=3, capacity_kg=10_000, population=20, generations=30, mutation_rate=0.05, crossover_rate=0.85, early_stopping_patience=30, early_stopping_min_delta=0.05, timeout_seconds=None)
    fingerprint = _scenario_fingerprint(base, **args)
    assert fingerprint == _scenario_fingerprint(base, **args)
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
    assert report["schema_version"] == 2
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
    assert len(early_stop_settings) == len(BENCHMARK_SEEDS) * 3
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
        assert scenario["baseline"]["feasibility"]
        for run in scenario["optimizer"]["runs"]:
            assert run["route_distance_km"] >= 0
            assert run["runtime_seconds"] >= 0
            assert run["timeout_seconds"] == 0.5
            assert isinstance(run["timed_out"], bool)
            assert "feasible" in run
        assert "feasibility_rate" in scenario["optimizer"]["summary"]
        assert "timeout_rate" in scenario["optimizer"]["summary"]


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
