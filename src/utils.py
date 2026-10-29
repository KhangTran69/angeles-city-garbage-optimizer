"""
Mathematical and routing utility functions for Angeles City Garbage Route Optimizer.
Includes Haversine distance, road distance calculators, and metrics formatting.
"""
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from src.data import BARANGAY_COORDS, DEPOT_COORD, DEPOT, BARANGAY_WASTE_KG

def haversine(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0  # Earth radius in KM

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 3)

def get_coord(point_name: str) -> Tuple[float, float]:
    """Get (lat, lon) for a given barangay name or the Depot."""
    if point_name == DEPOT:
        return DEPOT_COORD
    return BARANGAY_COORDS.get(point_name, (0.0, 0.0))

def route_distance_road(
    route: List[str], 
    road_graph, 
    distance_matrix: Optional[List[List[float]] | np.ndarray],
    point_to_index: Optional[Dict[str, int]]
) -> float:
    """
    Calculate the total road-network distance for a route (Depot -> Stops -> Depot).
    Uses distance_matrix if provided, otherwise fall back to haversine.
    """
    if not route:
        return 0.0
    
    full_route = [DEPOT] + route + [DEPOT]
    total_km = 0.0
    
    for i in range(len(full_route) - 1):
        p1, p2 = full_route[i], full_route[i+1]
        
        if distance_matrix is not None and point_to_index is not None:
            idx1, idx2 = point_to_index.get(p1), point_to_index.get(p2)
            if idx1 is not None and idx2 is not None:
                total_km += distance_matrix[idx1][idx2]
                continue
        
        # Fallback to Haversine if matrix lookup fails
        total_km += haversine(get_coord(p1), get_coord(p2))
    
    return round(total_km, 2)

def get_route_load(route: List[str], waste_dict: Optional[Dict[str, int]] = None) -> int:
    """Sum total waste load (kg) for a list of stops."""
    w_dict = waste_dict if waste_dict is not None else BARANGAY_WASTE_KG
    return sum(w_dict.get(stop, 0) for stop in route)

def get_load_utilization_pct(route: List[str], capacity: int, waste_dict: Optional[Dict[str, int]] = None) -> int:
    """Calculate percentage of vehicle capacity utilized."""
    if capacity <= 0: return 0
    load = get_route_load(route, waste_dict)
    return int((load / capacity) * 100)

def get_utilization_style(pct: int) -> str:
    """Return a CSS color code based on utilization percentage."""
    if pct < 80:
        return "#69f0ae"  # Healthy (Accent Green)
    if pct < 95:
        return "#ffb74d"  # Warning (Orange)
    return "#ff6b6b"      # Overload/Critical (Red)

def format_eta_minutes(minutes: float) -> str:
    """Convert floating minutes into a readable 'XH Ym' format."""
    h = int(minutes // 60)
    m = int(minutes % 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"

def estimate_route_duration_minutes(distance_km: float, n_stops: int) -> float:
    """Estimate total time including travel and collection service time."""
    from src.config import MAP_AVG_SPEED_KMH, STOP_SERVICE_MINUTES
    travel_time = (distance_km / MAP_AVG_SPEED_KMH) * 60
    service_time = n_stops * STOP_SERVICE_MINUTES
    return travel_time + service_time

def two_opt(
    route: List[str], 
    road_graph, 
    distance_matrix: Optional[List[List[float]] | np.ndarray],
    point_to_index: Optional[Dict[str, int]]
) -> List[str]:
    """Basic 2-opt local search to refine an individual vehicle's route."""
    best_route = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route)):
                if j - i == 1: continue
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                if route_distance_road(new_route, road_graph, distance_matrix, point_to_index) < \
                   route_distance_road(best_route, road_graph, distance_matrix, point_to_index):
                    best_route = new_route
                    improved = True
        if not improved: break
    return best_route

def calculate_route_time_penalty(
    route: List[str], 
    distance_matrix: Optional[List[List[float]]], 
    point_to_index: Optional[Dict[str, int]]
) -> float:
    """
    Placeholder for time-window calculations if needed for thesis.
    Currently returns 0.0 (could be expanded to penalize late arrivals).
    """
    return 0.0

def split_into_vehicles(selected: List[str], n_vehicles: int) -> List[List[str]]:
    """Naively split a list of stops into N approximately equal chunks."""
    if not selected: return [[] for _ in range(n_vehicles)]
    size = len(selected) // n_vehicles
    remainder = len(selected) % n_vehicles
    routes = []
    start = 0
    for i in range(n_vehicles):
        end = start + size + (1 if i < remainder else 0)
        routes.append(selected[start:end])
        start = end
    return routes
