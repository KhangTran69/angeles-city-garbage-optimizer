
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from src.data import BARANGAYS, DEPOT, BARANGAY_WASTE_KG
from src.utils import route_distance_road, haversine, get_coord
from src.optimization import evaluate_manual_baseline, greedy_baseline

# Mock road graph and distance matrix for simple testing if needed
# But let's try to use the actual ones if they exist
from src.optimization import load_road_network, get_distance_matrix

def verify():
    print("--- Baseline Accuracy Verification ---")
    
    graph = load_road_network()
    if graph is None:
        print("Error: Could not load road network.")
        return
        
    dist_matrix, pt_to_idx = get_distance_matrix(graph)
    
    # Test with a set that exceeds 5000kg
    # Total: Agapito(850)+Amsic(1250)+Anunas(980)+Balibago(2450)+Capaya(780) = 6310kg
    selected = ['Agapito del Rosario', 'Amsic', 'Anunas', 'Balibago', 'Capaya']
    n_vehicles = 2
    
    print(f"Selected Barangays: {selected}")
    print(f"Number of Vehicles: {n_vehicles}")
    
    # 1. Manual GSO
    gso_dist, gso_routes = evaluate_manual_baseline(selected, graph, dist_matrix, pt_to_idx)
    print(f"\n[Manual GSO - Now Capacity-Aware]")
    print(f"Routes (Trips): {gso_routes}")
    print(f"Total Distance: {gso_dist:.2f} km")
    
    # Verify each trip is under 5000kg
    for i, r in enumerate(gso_routes):
        load = sum(BARANGAY_WASTE_KG[p] for p in r)
        print(f"  Trip {i} Load: {load} kg {'✅' if load <= 5000 else '❌'}")

    # 2. Greedy Baseline
    greedy_dist, greedy_routes = greedy_baseline(selected, n_vehicles, BARANGAY_WASTE_KG, graph, dist_matrix, pt_to_idx)
    print(f"\n[Greedy Algorithm - Now Capacity-Aware]")
    print(f"Routes: {greedy_routes}")
    print(f"Total Distance: {greedy_dist:.2f} km")
    
    for i, r in enumerate(greedy_routes):
        load = sum(BARANGAY_WASTE_KG[p] for p in r)
        print(f"  Vehicle {i} Load: {load} kg {'✅' if load <= 5000 else '❌'}")

if __name__ == "__main__":
    verify()
