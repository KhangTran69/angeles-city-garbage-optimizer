import numpy as np

from src.optimization import (
    build_seed_chromosomes,
    decode_chromosome,
    evaluate_manual_baseline,
    greedy_baseline,
    load_cached_distance_matrix,
    run_benchmark,
    run_ga,
)
import src.optimization as optimization_module
from src.utils import get_route_load

def test_greedy_baseline_simple():
    """Verify greedy baseline correctly partitions selected barangays and returns valid distances."""
    selected = ["Amsic", "Anunas", "Balibago"]
    n_vehicles = 2
    
    dist, routes = greedy_baseline(selected, n_vehicles)
    
    assert isinstance(dist, float)
    assert dist >= 0.0
    assert len(routes) == n_vehicles
    
    # Check that all barangays were visited exactly once
    flattened = [b for r in routes for b in r]
    assert len(flattened) == len(selected)
    for s in selected:
        assert s in flattened

def test_greedy_baseline_custom_waste():
    """Verify that greedy algorithm respects custom waste modifications."""
    selected = ["Amsic", "Anunas"]
    n_vehicles = 1
    custom_waste = {"Amsic": 1, "Anunas": 1}
    
    dist, routes = greedy_baseline(selected, n_vehicles, waste_dict=custom_waste)
    assert len(routes[0]) == 2


def test_decode_chromosome_respects_capacity_when_possible():
    """The decoder should keep routes within capacity when a feasible split exists."""
    chromosome = ["Amsic", "Anunas", "Balibago"]
    waste_dict = {"Amsic": 1000, "Anunas": 900, "Balibago": 1100}

    routes, _, _ = decode_chromosome(chromosome, n_vehicles=2, vehicle_capacity_kg=2000, waste_dict=waste_dict)

    flattened = [stop for route in routes for stop in route]
    assert sorted(flattened) == sorted(chromosome)
    assert all(get_route_load(route, waste_dict) <= 2000 for route in routes if route)


def test_run_ga_returns_complete_visit_set():
    """GA output should visit each selected barangay exactly once."""
    selected = ["Amsic", "Anunas", "Balibago", "Capaya"]
    custom_waste = {barangay: 500 for barangay in selected}

    routes, dists, total, history = run_ga(
        selected,
        n_vehicles=2,
        pop_size=20,
        generations=5,
        mutation_rate=0.05,
        crossover_rate=0.8,
        vehicle_capacity_kg=2000,
        waste_dict=custom_waste,
        road_graph=None,
        distance_matrix=None,
        point_to_index=None,
    )

    flattened = [stop for route in routes for stop in route]
    assert sorted(flattened) == sorted(selected)
    assert len(dists) == 2
    assert total >= 0.0
    assert len(history) == 5


def test_run_ga_handles_empty_selection():
    """Empty selections should return empty routes without crashing."""
    routes, dists, total, history = run_ga(
        [],
        n_vehicles=3,
        pop_size=10,
        generations=5,
        mutation_rate=0.05,
        crossover_rate=0.8,
    )

    assert routes == [[], [], []]
    assert dists == [0.0, 0.0, 0.0]
    assert total == 0.0
    assert history == []


def test_run_ga_single_vehicle_stays_complete():
    """Single-vehicle runs should keep all stops assigned exactly once."""
    selected = ["Amsic", "Anunas", "Balibago"]
    routes, dists, total, history = run_ga(
        selected,
        n_vehicles=1,
        pop_size=12,
        generations=4,
        mutation_rate=0.05,
        crossover_rate=0.8,
        vehicle_capacity_kg=10000,
    )

    assert len(routes) == 1
    assert sorted(routes[0]) == sorted(selected)
    assert len(dists) == 1
    assert total >= 0
    assert len(history) == 4


def test_decode_chromosome_handles_impossible_capacity():
    """Decoder should still assign all stops even if every route must overload."""
    chromosome = ["Amsic", "Anunas"]
    waste_dict = {"Amsic": 6000, "Anunas": 7000}

    routes, _, _ = decode_chromosome(chromosome, n_vehicles=1, vehicle_capacity_kg=1000, waste_dict=waste_dict)

    assert routes == [["Amsic", "Anunas"]]


def test_run_benchmark_is_reproducible():
    """Benchmark totals should be deterministic for a fixed base seed."""
    selected = ["Amsic", "Anunas", "Balibago"]

    first = run_benchmark(selected, 2, 10, 4, 0.05, 0.8, base_seed=99, runs=3)
    second = run_benchmark(selected, 2, 10, 4, 0.05, 0.8, base_seed=99, runs=3)

    assert first["totals"] == second["totals"]
    assert first["seeds"] == [99, 100, 101]


def test_evaluate_manual_baseline_respects_custom_capacity():
    """Manual baseline should split routes according to the active capacity setting."""
    selected = ["Agapito del Rosario", "Amsic", "Anunas", "Balibago", "Capaya"]
    custom_waste = {
        "Agapito del Rosario": 850,
        "Amsic": 1250,
        "Anunas": 980,
        "Balibago": 2450,
        "Capaya": 780,
    }

    _, routes = evaluate_manual_baseline(
        selected,
        None,
        None,
        None,
        vehicle_capacity_kg=3000,
        waste_dict=custom_waste,
    )

    assert len(routes) == 4
    assert all(get_route_load(route, custom_waste) <= 3000 for route in routes)


def test_load_cached_distance_matrix_uses_saved_matrix(tmp_path, monkeypatch):
    """The fast path should reuse a valid persisted matrix without needing the graph."""
    matrix_path = tmp_path / "distance_matrix.npy"
    expected = np.zeros((34, 34), dtype=np.float32)
    np.save(matrix_path, expected)
    load_cached_distance_matrix.clear()
    monkeypatch.setattr(optimization_module, "DISTANCE_MATRIX_PATH", str(matrix_path))

    loaded, point_to_idx = load_cached_distance_matrix()

    assert loaded is not None
    assert loaded.shape == expected.shape
    assert point_to_idx["Angeles City Transfer Station"] == 0
    assert point_to_idx["Virgen Delos Remedios"] == 33


def test_run_benchmark_reuses_seed_chromosomes(monkeypatch):
    """Repeated benchmark runs should not rebuild deterministic baseline seeds every trial."""
    call_counts = {"greedy": 0, "manual": 0}

    def fake_greedy(selected, n_v, waste_dict=None, road_graph=None, dist_matrix=None, pt_to_idx=None, vehicle_capacity_kg=5000):
        call_counts["greedy"] += 1
        return 0.0, [selected[:]]

    def fake_manual(selected, road_graph=None, dist_matrix=None, pt_to_idx=None, vehicle_capacity_kg=5000, waste_dict=None):
        call_counts["manual"] += 1
        return 0.0, [selected[:]]

    monkeypatch.setattr(optimization_module, "greedy_baseline", fake_greedy)
    monkeypatch.setattr(optimization_module, "evaluate_manual_baseline", fake_manual)

    benchmark = run_benchmark(["Amsic", "Anunas"], 1, 6, 2, 0.05, 0.8, base_seed=7, runs=3)

    assert benchmark["seeds"] == [7, 8, 9]
    assert call_counts == {"greedy": 1, "manual": 1}


def test_run_ga_caches_duplicate_chromosome_fitness(monkeypatch):
    """Duplicate chromosomes should only pay one fitness decode plus the final route decode."""
    selected = ["Amsic", "Anunas", "Balibago"]
    fitness_calls = {"decode": 0}

    def fake_decode(chromosome, n_vehicles, vehicle_capacity_kg, waste_dict=None):
        fitness_calls["decode"] += 1
        return [chromosome[:]], [float(len(chromosome))], [sum(waste_dict.values()) if waste_dict else 0]

    def fake_distance(route, road_graph, distance_matrix, point_to_index):
        return float(len(route))

    def fake_sample(self, population, k):
        return selected[:]

    monkeypatch.setattr(optimization_module, "decode_chromosome", fake_decode)
    monkeypatch.setattr(optimization_module, "route_distance_road", fake_distance)
    monkeypatch.setattr(optimization_module.random.Random, "sample", fake_sample)

    routes, dists, total, history = run_ga(
        selected,
        n_vehicles=1,
        pop_size=6,
        generations=4,
        mutation_rate=0.0,
        crossover_rate=0.0,
        random_seed=11,
        seed_chromosomes=[selected[:], selected[:]],
    )

    assert routes == [selected]
    assert dists == [3.0]
    assert total == 3.0
    assert len(history) == 4
    assert fitness_calls["decode"] == 2


def test_run_benchmark_reuses_route_distance_cache_across_runs(monkeypatch):
    """Benchmark runs should share cached route distances when the decoded routes repeat."""
    selected = ["Amsic", "Anunas", "Balibago"]
    route_distance_calls = {"count": 0}

    def fake_decode(chromosome, n_vehicles, vehicle_capacity_kg, waste_dict=None, **kwargs):
        return [chromosome[:]], [3.0], [0]

    def fake_distance(route, road_graph, distance_matrix, point_to_index):
        route_distance_calls["count"] += 1
        return float(len(route))

    def fake_sample(self, population, k):
        return selected[:]

    monkeypatch.setattr(optimization_module, "decode_chromosome", fake_decode)
    monkeypatch.setattr(optimization_module, "route_distance_road", fake_distance)
    monkeypatch.setattr(optimization_module, "two_opt", lambda route, road_graph, distance_matrix, point_to_index: route)
    monkeypatch.setattr(optimization_module, "build_seed_chromosomes", lambda *args, **kwargs: [selected[:]])
    monkeypatch.setattr(optimization_module.random.Random, "sample", fake_sample)

    benchmark = run_benchmark(
        selected,
        n_vehicles=1,
        pop_size=6,
        generations=4,
        mutation_rate=0.0,
        crossover_rate=0.0,
        base_seed=7,
        runs=3,
    )

    assert benchmark["totals"] == [3.0, 3.0, 3.0]
    assert route_distance_calls["count"] == 1
