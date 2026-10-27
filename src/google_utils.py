"""
Google Maps Directions API utilities for Angeles City Garbage Route Optimizer.
Provides high-fidelity route verification against Google's routing engine.
"""
import requests
from typing import List, Tuple, Dict, Any, Optional
from src.logger import logger

GOOGLE_MAPS_BASE_URL = "https://maps.googleapis.com/maps/api/directions/json"

def fetch_google_route(
    coords: List[Tuple[float, float]], 
    api_key: str
) -> Optional[Dict[str, Any]]:
    """
    Fetch a route from Google Maps Directions API.
    
    Args:
        coords: List of (lat, lon) coordinates.
        api_key: Google Maps API Key.
        
    Returns:
        Dictionary with distance (km), duration (mins), and encoded polyline.
    """
    if not api_key:
        logger.warning("Google Maps API key not provided")
        return None

    if len(coords) < 2:
        return None

    origin = f"{coords[0][0]},{coords[0][1]}"
    destination = f"{coords[-1][0]},{coords[-1][1]}"
    waypoints = "|".join([f"{c[0]},{c[1]}" for c in coords[1:-1]])

    params = {
        "origin": origin,
        "destination": destination,
        "key": api_key,
        "mode": "driving"
    }
    if waypoints:
        params["waypoints"] = f"optimize:false|{waypoints}"

    try:
        response = requests.get(GOOGLE_MAPS_BASE_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "OK":
                route = data["routes"][0]
                total_distance_m = 0
                total_duration_s = 0
                
                for leg in route["legs"]:
                    total_distance_m += leg["distance"]["value"]
                    total_duration_s += leg["duration"]["value"]
                
                return {
                    "distance_km": total_distance_m / 1000.0,
                    "duration_mins": total_duration_s / 60.0,
                    "polyline": route["overview_polyline"]["points"],
                    "status": "OK"
                }
            else:
                logger.error(f"Google Maps API error: {data.get('status')} - {data.get('error_message')}")
                return {"status": data.get("status"), "error": data.get("error_message")}
        else:
            logger.error(f"Google Maps HTTP error: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Failed to fetch Google Maps route: {str(e)}")
        return None

def verify_optimized_route(
    route_name: str,
    coords: List[Tuple[float, float]],
    current_distance_km: float,
    api_key: str
) -> Dict[str, Any]:
    """
    Compare an existing route distance with Google Maps verification.
    """
    google_data = fetch_google_route(coords, api_key)
    if not google_data or google_data.get("status") != "OK":
        return {"success": False, "error": google_data.get("error") if google_data else "Unknown error"}

    google_dist = google_data["distance_km"]
    delta_km = current_distance_km - google_dist
    delta_pct = (delta_km / google_dist) * 100 if google_dist > 0 else 0

    return {
        "success": True,
        "route_name": route_name,
        "current_distance_km": round(current_distance_km, 3),
        "google_distance_km": round(google_dist, 3),
        "delta_km": round(delta_km, 3),
        "delta_pct": round(delta_pct, 2),
        "google_duration_mins": round(google_data["duration_mins"], 1),
        "polyline": google_data["polyline"]
    }
