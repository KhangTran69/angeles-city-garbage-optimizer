"""
Genetic Algorithm and Distance Matrix Optimization for Angeles City Garbage Routing.
Optimization Recovery: GA Seeding + Unified Road Physics.
"""
import os
import random
from math import ceil
import numpy as np
import streamlit as st
from typing import List, Tuple, Dict, Any, Optional

from src.config import (
    DISTANCE_MATRIX_PATH, GRAPH_PATH, OSM_BBOX, 
    DEFAULT_VEHICLE_CAPACITY_KG, CVRP_OVERLOAD_PENALTY_PER_KG
)
from src.data import BARANGAYS, DEPOT, BARANGAY_WASTE_KG
from src.utils import (
    haversine, get_coord, route_distance_road, 
    get_route_load, two_opt
)

def _build_point_index() -> Dict[str, int]:
    """Create the shared depot/barangay lookup used by the distance matrix."""
    return {point: idx for idx, point in enumerate([DEPOT] + BARANGAYS)}


@st.cache_data(show_spinner=False)
def load_cached_distance_matrix() -> Tuple[Optional[np.ndarray], Dict[str, int]]:
    """Load the persisted distance matrix without touching the road graph."""
    point_to_idx = _build_point_index()
    n_points = len(point_to_idx)
    if not os.path.exists(DISTANCE_MATRIX_PATH):
        return None, point_to_idx

    try:
        dist_matrix = np.load(DISTANCE_MATRIX_PATH)
    except Exception:
        return None, point_to_idx

    if dist_matrix.shape != (n_points, n_points):
        return None, point_to_idx
    return dist_matrix, point_to_idx


def decode_chromosome(
    chromosome_indices: List[int],
    n_vehicles: int,
    vehicle_capacity_kg: int,
    waste_list: List[int],
    dist_matrix: List[List[float]],
    depot_idx: int,
) -> Tuple[List[List[int]], List[float], List[int]]:
    """
    Extremely fast decoding loop using integer indices and native Python List-of-Lists.
    Avoids string lookups and NumPy overhead in the tightest O(N*M) loop.
    """
    if n_vehicles <= 0: return [], [], []

    routes: List[List[int]] = [[] for _ in range(n_vehicles)]
    loads = [0 for _ in range(n_vehicles)]
    route_dist_proxies = [0.0 for _ in range(n_vehicles)]
    last_stop_indices = [depot_idx for _ in range(n_vehicles)]
    target_stops = max(1, len(chromosome_indices) // n_vehicles)
    
    # Cache local variables for speed
    cap = vehicle_capacity_kg
    
    for stop_idx in chromosome_indices:
        stop_load = waste_list[stop_idx]
        best_idx = 0
        best_score = (999999999.0, 999999999.0, 999999999.0, 999999999.0)
        
        for v_idx in range(n_vehicles):
            curr_load = loads[v_idx]
            overload = max(0, curr_load + stop_load - cap)
            inc_dist = dist_matrix[last_stop_indices[v_idx]][stop_idx]
            
            # Simple score components
            proj_ratio = (curr_load + stop_load) / cap
            bal_gap = max(0, len(routes[v_idx]) + 1 - target_stops)
            
            # Priority: Overload > Balanced Load > Cumulative Route Distance > Stop Count
            score = (
                float(overload),
                float(proj_ratio + bal_gap * 0.15),
                float(route_dist_proxies[v_idx] + inc_dist),
                float(len(routes[v_idx]))
            )
            
            if score < best_score:
                best_score = score
                best_idx = v_idx

        # Dispatch
        chosen = best_idx
        routes[chosen].append(stop_idx)
        loads[chosen] += stop_load
        route_dist_proxies[chosen] += (best_score[2] - route_dist_proxies[chosen])
        last_stop_indices[chosen] = stop_idx

    # Add return-to-depot distances
    for i in range(n_vehicles):
        if routes[i]:
            route_dist_proxies[i] += dist_matrix[last_stop_indices[i]][depot_idx]

    return routes, route_dist_proxies, loads


def build_seed_chromosomes(
    selected_barangays: List[str],
    n_vehicles: int,
    vehicle_capacity_kg: int = DEFAULT_VEHICLE_CAPACITY_KG,
    waste_dict: Optional[Dict[str, int]] = None,
    road_graph=None,
    distance_matrix: Optional[np.ndarray] = None,
    point_to_index: Optional[Dict[str, int]] = None,
) -> List[List[str]]:
    """Precompute deterministic baseline seed chromosomes for repeated GA runs."""
    if not selected_barangays:
        return []

    w_dict = waste_dict if waste_dict else BARANGAY_WASTE_KG
    seed_chromosomes: List[List[str]] = []

    _, greedy_routes = greedy_baseline(
        selected_barangays,
        n_vehicles,
        w_dict,
        road_graph,
        distance_matrix,
        point_to_index,
        vehicle_capacity_kg,
    )
    seed_chromosomes.append([node for route in greedy_routes for node in route])

    _, manual_routes = evaluate_manual_baseline(
        selected_barangays,
        road_graph,
        distance_matrix,
        point_to_index,
        vehicle_capacity_kg,
        w_dict,
    )
    manual_flat = [node for route in manual_routes for node in route]
    if len(manual_flat) == len(selected_barangays):
        seed_chromosomes.append(manual_flat)

    return seed_chromosomes

@st.cache_resource
def load_road_network() -> Optional[Any]:
    """Load or download the Angeles City road network."""
    import osmnx as ox
    if os.path.exists(GRAPH_PATH):
        try: return ox.load_graphml(GRAPH_PATH)
        except Exception: pass
    try:
        graph = ox.project_graph(ox.graph_from_bbox(*OSM_BBOX, network_type="drive"))
        ox.save_graphml(ox.routing.add_edge_travel_times(ox.routing.add_edge_speeds(graph)), GRAPH_PATH)
        return graph
    except Exception: return None

@st.cache_data
def get_distance_matrix(_graph) -> Tuple[np.ndarray, Dict[str, int]]:
    import networkx as nx
    import osmnx as ox
    import pyproj
    
    points = [DEPOT] + BARANGAYS
    point_to_idx = _build_point_index()
    n = len(points)
    cached_matrix, cached_index = load_cached_distance_matrix()
    if cached_matrix is not None:
        return cached_matrix, cached_index

    # 🚀 TIER 1: HIGH-FIDELITY TRUCK ROUTING (ORS API)
    from src.config import ORS_API_KEY
    if ORS_API_KEY:
        from src.map_utils import fetch_ors_matrix
        st.info("📡 Using OpenRouteService HGV Matrix for Truck-Feasible Optimization...")
        coords = [get_coord(p) for p in points]
        matrix = fetch_ors_matrix([list(c) for c in coords], ORS_API_KEY)
        if matrix:
            dist_matrix = np.array(matrix, dtype=np.float32) / 1000.0
            os.makedirs(os.path.dirname(DISTANCE_MATRIX_PATH), exist_ok=True)
            np.save(DISTANCE_MATRIX_PATH, dist_matrix)
            return dist_matrix, point_to_idx
        else:
            st.warning("⚠️ ORS Matrix API failed. Falling back to local road network approximation.")

    # 🚀 TIER 2: LOCAL ROAD NETWORK (OSMNX)
    target_crs = _graph.graph["crs"]

    projector = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True).transform
    node_map = {p: ox.distance.nearest_nodes(_graph, *projector(get_coord(p)[1], get_coord(p)[0])) for p in points}
    relevant_nodes = set(node_map.values())
    dist_matrix = np.full((n, n), fill_value=0.0, dtype=np.float32)
    found_distances = {}
    matrix_prog = st.progress(0, text="Optimizing Road Network Pathfinding...")
    for i, (point_name, start_node) in enumerate(node_map.items()):
        lengths = nx.single_source_dijkstra_path_length(_graph, start_node, weight="length")
        found_distances[start_node] = {target: d for target, d in lengths.items() if target in relevant_nodes}
        matrix_prog.progress((i + 1) / n, text=f"Mapping Connectivity: {point_name}")
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i == j: continue
            d = found_distances.get(node_map[p1], {}).get(node_map[p2])
            dist_matrix[i, j] = (d / 1000.0) if d is not None else haversine(get_coord(p1), get_coord(p2))
    matrix_prog.empty()
    os.makedirs(os.path.dirname(DISTANCE_MATRIX_PATH), exist_ok=True)
    np.save(DISTANCE_MATRIX_PATH, dist_matrix)
    return dist_matrix, point_to_idx

def run_ga(
    selected_barangays: List[str], n_vehicles: int, pop_size: int, generations: int,
    mutation_rate: float, crossover_rate: float, progress_placeholder=None, log_placeholder=None,
    vehicle_capacity_kg: int = DEFAULT_VEHICLE_CAPACITY_KG, waste_dict: Optional[Dict[str, int]] = None,
    road_graph=None, distance_matrix: Optional[np.ndarray] = None, point_to_index: Optional[Dict[str, int]] = None,
    random_seed: Optional[int] = None, seed_chromosomes: Optional[List[List[str]]] = None,
    route_distance_cache: Optional[Dict[Tuple[str, ...], float]] = None,
) -> Tuple[List[List[str]], List[float], float, List[float]]:
    """Execute CVRP solver with Seeding (Greedy + Manual GSO)."""
    if not selected_barangays:
        return [[] for _ in range(n_vehicles)], [0.0 for _ in range(n_vehicles)], 0.0, []

    rng = random.Random(random_seed)
    fitness_cache: Dict[Tuple[int, ...], float] = {}
    
    # 1. PRE-CALCULATE ALL MAPPINGS (Massive Performance Gain)
    # Convert matrix to native Python list-of-lists (faster for individual access than NumPy)
    dist_list = distance_matrix.tolist() if distance_matrix is not None else None
    if dist_list is None:
        # Emergency recovery: Build a haversine list if no matrix is available
        points = [DEPOT] + BARANGAYS
        n = len(points)
        dist_list = [[haversine(get_coord(p1), get_coord(p2)) for p2 in points] for p1 in points]
        pt_idx = {p: i for i, p in enumerate(points)}
    else:
        pt_idx = point_to_index if point_to_index else {}
        
    depot_idx = pt_idx.get(DEPOT, 0)
    w_dict = waste_dict if waste_dict else BARANGAY_WASTE_KG
    
    # Global index maps for selected barangays
    id_to_name = {i: name for i, name in enumerate([DEPOT] + BARANGAYS)}
    name_to_id = {name: i for i, name in id_to_name.items()}
    waste_list = [w_dict.get(id_to_name[i], 0) for i in range(len(id_to_name))]
    
    selected_indices = [name_to_id[b] for b in selected_barangays]

    def fitness(chromosome_indices: List[int]) -> float:
        chrom_key = tuple(chromosome_indices)
        if chrom_key in fitness_cache: return fitness_cache[chrom_key]

        _, route_dists, route_loads = decode_chromosome(
            chromosome_indices, n_vehicles, vehicle_capacity_kg, waste_list, dist_list, depot_idx # type: ignore
        )
        total_dist, total_penalty = 0.0, 0.0
        active_dists = []
        for d, load in zip(route_dists, route_loads):
            if d <= 0: continue
            total_dist += d
            active_dists.append(d)
            if load > vehicle_capacity_kg: 
                total_penalty += (load - vehicle_capacity_kg) * CVRP_OVERLOAD_PENALTY_PER_KG
        
        imbalance = max(active_dists) - min(active_dists) if active_dists else 0
        score = total_dist + total_penalty + (imbalance * 0.1)
        fitness_cache[chrom_key] = score
        return score

    def crossover(p1: List[int], p2: List[int]) -> List[int]:
        if len(p1) < 2 or rng.random() > crossover_rate:
            return p1[:]
        a, b = sorted(rng.sample(range(len(p1)), 2))
        child = [None] * len(p1)
        child[a:b] = p1[a:b]
        child_set = set(p1[a:b])
        p2_f = [x for x in p2 if x not in child_set]
        idx = 0
        for i in range(len(p1)):
            if child[i] is None: child[i] = p2_f[idx]; idx += 1
        return child # type: ignore

    population = [rng.sample(selected_indices, len(selected_indices)) for _ in range(max(pop_size, 2))]
    
    # Handle Seeding (Convert names to indices)
    if not seed_chromosomes:
        baseline_seed_chroms_names = build_seed_chromosomes(
            selected_barangays, n_vehicles, vehicle_capacity_kg, w_dict, 
            road_graph, distance_matrix, point_to_index
        )
        pop_indices = [[name_to_id[n] for n in chrom] for chrom in baseline_seed_chroms_names]
    else:
        pop_indices = [[name_to_id[n] for n in chrom] for chrom in seed_chromosomes]

    for idx, chromosome in enumerate(pop_indices[:2]):
        if len(chromosome) == len(selected_indices):
            population[idx] = chromosome[:]

    history = []
    best_chrom, best_fit = population[0], fitness(population[0])
    patience = 30
    gen_no_improve = 0
    min_delta = 0.05

    for gen in range(generations):
        # Elitism & Sorting
        ranked_population = sorted((fitness(chrom), chrom) for chrom in population)
        curr_fit, best_in_gen = ranked_population[0]
        
        # Early Stopping Logic
        if curr_fit < best_fit - min_delta:
            best_fit, best_chrom = float(curr_fit), best_in_gen[:]
            gen_no_improve = 0
        else:
            gen_no_improve += 1
            
        history.append(float(round(best_fit, 2)))
        
        # Breed next generation from sorted survivors
        population = [chrom for _, chrom in ranked_population]
        new_pop = population[:max(2, pop_size // 10)]
        while len(new_pop) < pop_size:
            p1, p2 = rng.choices(population[:max(2, pop_size // 2)], k=2)
            child = crossover(p1, p2)
            if len(child) >= 2 and rng.random() < mutation_rate:
                i, j = rng.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        population = new_pop

        if progress_placeholder: progress_placeholder.progress((gen + 1) / generations)
        if log_placeholder and gen % 10 == 0: log_placeholder.markdown(f"Gen {gen+1}: {best_fit:.2f} km")
        
        if gen_no_improve >= patience:
            if log_placeholder: log_placeholder.markdown(f"Early stop at gen {gen+1} (Convergence)")
            break

    # MAP BACK TO STRINGS (Final Step)
    best_decoded_indices, _, _ = decode_chromosome(
        best_chrom, n_vehicles, vehicle_capacity_kg, waste_list, dist_list, depot_idx # type: ignore
    )
    final_routes_names = [[id_to_name[idx] for idx in route] for route in best_decoded_indices]
    final_routes = [two_opt(route, road_graph, distance_matrix, point_to_index) for route in final_routes_names]
    
    # Correct distance calculation with road network
    final_dists = [route_distance_road(r, road_graph, distance_matrix, point_to_index) for r in final_routes]
    return final_routes, final_dists, float(sum(final_dists)), history


def run_benchmark(
    selected_barangays: List[str],
    n_vehicles: int,
    pop_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    vehicle_capacity_kg: int = DEFAULT_VEHICLE_CAPACITY_KG,
    waste_dict: Optional[Dict[str, int]] = None,
    road_graph=None,
    distance_matrix: Optional[np.ndarray] = None,
    point_to_index: Optional[Dict[str, int]] = None,
    base_seed: int = 0,
    runs: int = 5,
) -> Dict[str, Any]:
    """Execute a reproducible benchmark suite across sequential seeds."""
    seeds = [base_seed + offset for offset in range(max(1, runs))]
    totals: List[float] = []
    histories: List[List[float]] = []
    shared_route_distance_cache: Dict[Tuple[str, ...], float] = {}
    seed_chromosomes = build_seed_chromosomes(
        selected_barangays,
        n_vehicles,
        vehicle_capacity_kg,
        waste_dict,
        road_graph,
        distance_matrix,
        point_to_index,
    )

    for seed in seeds:
        _, _, total, history = run_ga(
            selected_barangays,
            n_vehicles,
            pop_size,
            generations,
            mutation_rate,
            crossover_rate,
            None,
            None,
            vehicle_capacity_kg,
            waste_dict,
            road_graph,
            distance_matrix,
            point_to_index,
            seed,
            seed_chromosomes,
            shared_route_distance_cache,
        )
        totals.append(total)
        histories.append(history)

    return {
        "seeds": seeds,
        "totals": totals,
        "mean_total": float(np.mean(totals)) if totals else 0.0,
        "std_total": float(np.std(totals)) if totals else 0.0,
        "best_total": float(min(totals)) if totals else 0.0,
        "worst_total": float(max(totals)) if totals else 0.0,
        "histories": histories,
    }

def greedy_baseline(
    selected,
    n_v,
    waste_dict=None,
    road_graph=None,
    dist_matrix=None,
    pt_to_idx=None,
    vehicle_capacity_kg: int = DEFAULT_VEHICLE_CAPACITY_KG,
):
    """Capacity-Aware Greedy Heuristic using high-precision Road Matrix."""
    rem, routes = selected[:], [[] for _ in range(n_v)]
    w_dict = waste_dict if waste_dict else BARANGAY_WASTE_KG
    cap = vehicle_capacity_kg
    
    curr_v = 0
    curr_p = [DEPOT] * n_v
    curr_loads = [0] * n_v

    while rem:
        # Try to find the nearest point for current vehicle that fits capacity
        def get_dist(p1, p2):
            if dist_matrix is not None and pt_to_idx is not None:
                idx1, idx2 = pt_to_idx.get(p1), pt_to_idx.get(p2)
                if idx1 is not None and idx2 is not None: return dist_matrix[idx1][idx2]
            return haversine(get_coord(p1), get_coord(p2))

        # Sort remaining by closeness to current vehicle's last position
        rem.sort(key=lambda p: get_dist(curr_p[curr_v], p))
        
        found = False
        for p in rem:
            if curr_loads[curr_v] + w_dict.get(p, 0) <= cap:
                routes[curr_v].append(p)
                curr_loads[curr_v] += w_dict.get(p, 0)
                curr_p[curr_v] = p
                rem.remove(p)
                found = True
                break
        
        if not found:
            # Current vehicle is full or can't fit any remaining. Move to next.
            curr_v = (curr_v + 1) % n_v
            # If we've circled back and still can't fit anything, we must "force" a new trip 
            # or just accept an overload in this baseline (less ideal).
            # For simplicity in this heuristic, if all vehicles are "stuck", the last one takes the load.
            if all(curr_loads[v] + min([w_dict.get(x, 99999) for x in rem]) > cap for v in range(n_v)):
                 # Force pick the nearest to current vehicle
                 p = min(rem, key=lambda x: get_dist(curr_p[curr_v], x))
                 routes[curr_v].append(p)
                 curr_loads[curr_v] += w_dict.get(p, 0)
                 curr_p[curr_v] = p
                 rem.remove(p)
        
        curr_v = (curr_v + 1) % n_v

    return float(sum([route_distance_road(r, road_graph, dist_matrix, pt_to_idx) for r in routes])), routes

def evaluate_manual_baseline(
    selected,
    road_graph=None,
    dist_matrix=None,
    pt_to_idx=None,
    vehicle_capacity_kg: int = DEFAULT_VEHICLE_CAPACITY_KG,
    waste_dict: Optional[Dict[str, int]] = None,
):
    """
    Evaluates Manual GSO sectors with Trip Splitting.
    If a sector's total waste exceeds capacity, it is split into multiple trips.
    """
    from src.data import MANUAL_GSO_BASELINE
    cap = vehicle_capacity_kg
    w_dict = waste_dict if waste_dict is not None else BARANGAY_WASTE_KG
    
    raw_routes = [[b for b in s if b in selected] for s in MANUAL_GSO_BASELINE if [b for b in s if b in selected]]
    final_routes = []
    
    for r in raw_routes:
        if not r: continue
        # Split into multiple trips if overloaded
        curr_trip = []
        curr_load = 0
        for stop in r:
            stop_waste = w_dict.get(stop, 0)
            if curr_load + stop_waste > cap and curr_trip:
                final_routes.append(curr_trip)
                curr_trip = [stop]
                curr_load = stop_waste
            else:
                curr_trip.append(stop)
                curr_load += stop_waste
        if curr_trip:
            final_routes.append(curr_trip)
            
    return float(sum([route_distance_road(r, road_graph, dist_matrix, pt_to_idx) for r in final_routes])), final_routes
