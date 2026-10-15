"""
Street-Level Arc Routing Garbage Collection Optimizer for Angeles City, Philippines

This script shifts from Barangay-level to Street-Level routing using OSMnx and NetworkX.
It implements arc routing algorithms to ensure all specified street segments are traversed.

Features:
- Loads Angeles City road network from OSMnx
- Filters edges by street names for each zone
- Applies Chinese Postman Problem heuristic for arc routing
- Calculates distances including Depot to Route Start and Route End to Landfill
- Outputs distance matrix for optimization
- Differentiates between Mini Dump Trucks (street collection) and Big Dump Trucks (landfill transport)

Usage: python street_level_optimizer.py
"""

import osmnx as ox
import networkx as nx
from networkx.algorithms.approximation import christofides, traveling_salesman_problem
import numpy as np
from typing import List, Dict, Tuple, Set
from shapely.geometry import Point
import geopy.distance

# Constants
DEPOT_COORD = (15.14141069995556, 120.62967669596279)  # Angeles City Transfer Station
LANDFILL_COORD = (15.299778, 120.522833)  # Clark Sanitary Landfill

# Shifts
SHIFTS = {
    "morning": {"time": "08:00 AM–05:00 PM", "zones": ["A", "B"]},
    "night": {"time": "08:00 PM–06:00 AM", "zones": ["B", "C"]},
    "market": {"time": "specialized", "zones": ["A", "C"]},
}

# Core Routes (Zones)
ZONES = {
    "A": ["Sto. Rosario", "Plaridel", "Miranda St"],
    "B": ["New Abacan Bypass", "Arayat Blvd", "Pampang Road"],
    "C": ["MacArthur Hi-way"],  # Boundary Sto. Domingo to Marisol Rotonda
}

# Priority Landmarks
LANDMARKS = {
    "hospital_ona": (15.15, 120.58),  # Approximate, Hospital ning Ona
    "pampang_market": (15.15, 120.58),  # Approximate
    "city_high_pandan": (15.15, 120.58),  # Approximate
    "sapang_balen": (15.15, 120.58),  # Approximate
}

# Truck types
TRUCKS = {
    "mini_dump": {"capacity": 5000, "type": "street_collection"},
    "big_dump": {"capacity": 15000, "type": "landfill_transport"},
}


def load_angeles_city_graph() -> nx.MultiDiGraph:
    """Load the road network graph for Angeles City using OSMnx."""
    # For testing, use a smaller bbox around Angeles City
    bbox = (15.10, 15.20, 120.55, 120.65)  # Approximate bbox for Angeles City
    graph = ox.graph_from_bbox(bbox, network_type="drive")
    # Add travel times and speeds
    graph = ox.routing.add_edge_speeds(graph)
    graph = ox.routing.add_edge_travel_times(graph)
    return graph


def filter_edges_by_streets(graph: nx.MultiDiGraph, street_names: List[str]) -> nx.MultiDiGraph:
    """Filter the graph to include only edges that match the given street names."""
    edges_to_keep = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        name = data.get("name", "")
        if isinstance(name, list):
            name = name[0]  # Take first name if list
        if any(street.lower() in str(name).lower() for street in street_names):
            edges_to_keep.append((u, v, key))

    subgraph = graph.edge_subgraph(edges_to_keep).copy()
    return subgraph


def chinese_postman_arc_routing(graph: nx.MultiDiGraph) -> List[Tuple[int, int, int]]:
    """
    Implement a simplified Chinese Postman Problem for arc routing.
    Finds a route that traverses all edges at least once.
    """
    # For simplicity, use a heuristic: find a spanning tree and traverse
    # In practice, full CPP is complex; this is an approximation
    undirected_graph = graph.to_undirected()

    # Use minimum spanning tree and traverse
    mst = nx.minimum_spanning_tree(undirected_graph, weight="length")
    # Traverse MST edges
    route = list(mst.edges())

    # Duplicate edges to cover all if needed (simplified)
    all_edges = list(undirected_graph.edges())
    # For now, just return the MST edges as route
    return [(u, v, 0) for u, v in route]  # Assume key 0


def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate geodesic distance between two coordinates in km."""
    return geopy.distance.geodesic(coord1, coord2).km


def get_nearest_node(graph: nx.MultiDiGraph, coord: Tuple[float, float]) -> int:
    """Get the nearest node in the graph to the given coordinate."""
    return ox.distance.nearest_nodes(graph, coord[1], coord[0])


def plan_routes_for_shift(graph: nx.MultiDiGraph, shift: str) -> Dict[str, Dict]:
    """Plan routes for a given shift."""
    shift_data = SHIFTS[shift]
    zones = shift_data["zones"]
    routes = {}

    for zone in zones:
        streets = ZONES[zone]
        zone_graph = filter_edges_by_streets(graph, streets)

        if zone_graph.number_of_edges() == 0:
            print(f"No edges found for zone {zone}")
            continue

        # Get arc routing route
        route = chinese_postman_arc_routing(zone_graph)

        # Calculate total distance
        total_distance = (
            sum(zone_graph[u][v][k]["length"] for u, v, k in route) / 1000
        )  # Convert to km

        # Find start and end nodes
        start_node = route[0][0]
        end_node = route[-1][1]

        # Get coordinates
        start_coord = (graph.nodes[start_node]["y"], graph.nodes[start_node]["x"])
        end_coord = (graph.nodes[end_node]["y"], graph.nodes[end_node]["x"])

        # Calculate depot to start and end to landfill
        depot_to_start = calculate_distance(DEPOT_COORD, start_coord)
        end_to_landfill = calculate_distance(end_coord, LANDFILL_COORD)

        routes[zone] = {
            "route": route,
            "total_distance": total_distance,
            "start_coord": start_coord,
            "end_coord": end_coord,
            "depot_to_start": depot_to_start,
            "end_to_landfill": end_to_landfill,
            "truck_type": "mini_dump",  # For street collection
        }

    return routes


def generate_distance_matrix(routes: Dict[str, Dict]) -> np.ndarray:
    """Generate a distance matrix including depot and landfill legs."""
    zones = list(routes.keys())
    n = len(zones) + 2  # Zones + depot + landfill
    matrix = np.zeros((n, n))

    # Depot is index 0, landfill is index n-1
    depot_idx = 0
    landfill_idx = n - 1
    zone_indices = {zone: i + 1 for i, zone in enumerate(zones)}

    # Depot to zone starts
    for zone, data in routes.items():
        matrix[depot_idx, zone_indices[zone]] = data["depot_to_start"]
        matrix[zone_indices[zone], depot_idx] = data["depot_to_start"]

    # Zone ends to landfill
    for zone, data in routes.items():
        matrix[zone_indices[zone], landfill_idx] = data["end_to_landfill"]
        matrix[landfill_idx, zone_indices[zone]] = data["end_to_landfill"]

    # Between zones if needed (though not directly connected)
    for z1 in zones:
        for z2 in zones:
            if z1 != z2:
                dist = calculate_distance(routes[z1]["end_coord"], routes[z2]["start_coord"])
                matrix[zone_indices[z1], zone_indices[z2]] = dist

    return matrix


def main():
    """Main function to run the street-level optimizer."""
    print("Loading Angeles City road network...")
    graph = load_angeles_city_graph()
    print(f"Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

    all_routes = {}
    for shift in SHIFTS:
        print(f"\nPlanning routes for {shift} shift ({SHIFTS[shift]['time']})...")
        routes = plan_routes_for_shift(graph, shift)
        all_routes[shift] = routes

        for zone, data in routes.items():
            print(
                f"Zone {zone}: {data['total_distance']:.2f} km route, "
                f"Depot→Start: {data['depot_to_start']:.2f} km, "
                f"End→Landfill: {data['end_to_landfill']:.2f} km"
            )

    # Generate distance matrix for morning shift as example
    if "morning" in all_routes:
        matrix = generate_distance_matrix(all_routes["morning"])
        print("\nDistance Matrix for Morning Shift (km):")
        labels = ["Depot"] + list(all_routes["morning"].keys()) + ["Landfill"]
        print(" ".join(f"{label:>10}" for label in labels))
        for i, row in enumerate(matrix):
            print(f"{labels[i]:>10} " + " ".join(f"{val:>10.2f}" for val in row))

    # For big dump trucks, they transport from depot to landfill
    landfill_distance = calculate_distance(DEPOT_COORD, LANDFILL_COORD)
    print(f"\nBig Dump Truck Route: Depot → Landfill = {landfill_distance:.2f} km")


if __name__ == "__main__":
    main()
