"""
GPX Route Parsing and Comparison Utilities for Angeles City Garbage Route Optimizer.
Thesis AL-1: Real-World Driver Behavior vs Optimized Routes.
"""
import gpxpy
import gpxpy.gpx
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import math

from src.utils import haversine
from src.logger import logger, log_error


@dataclass
class RouteMetrics:
    """Computed metrics for a single GPX route."""
    route_type: str  # "baseline", "optimized", "local_knowledge"
    total_distance_km: float
    total_duration_minutes: float
    avg_speed_kmh: float
    num_waypoints: int
    num_stops: int
    estimated_fuel_liters: float
    estimated_co2_kg: float
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    start_coord: Optional[Tuple[float, float]] = None
    end_coord: Optional[Tuple[float, float]] = None
    polyline: List[Tuple[float, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_type": self.route_type,
            "total_distance_km": round(self.total_distance_km, 3),
            "total_duration_minutes": round(self.total_duration_minutes, 2),
            "avg_speed_kmh": round(self.avg_speed_kmh, 2),
            "num_waypoints": self.num_waypoints,
            "num_stops": self.num_stops,
            "estimated_fuel_liters": round(self.estimated_fuel_liters, 3),
            "estimated_co2_kg": round(self.estimated_co2_kg, 3),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "start_coord": self.start_coord,
            "end_coord": self.end_coord,
        }


@dataclass
class RouteComparison:
    """Comparison results between baseline, optimized, and local knowledge routes."""
    baseline: Optional[RouteMetrics]
    optimized: Optional[RouteMetrics]
    local_knowledge: Optional[RouteMetrics]
    
    # Computed comparisons
    optimized_savings_km: Optional[float] = None
    optimized_savings_pct: Optional[float] = None
    local_knowledge_savings_km: Optional[float] = None
    local_knowledge_savings_pct: Optional[float] = None
    efficiency_ranking: List[str] = field(default_factory=list)  # ["optimized", "local_knowledge", "baseline"]
    deviation_score_optimized: Optional[float] = None  # How much optimized differs from baseline
    deviation_score_local: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "optimized": self.optimized.to_dict() if self.optimized else None,
            "local_knowledge": self.local_knowledge.to_dict() if self.local_knowledge else None,
            "optimized_savings_km": round(self.optimized_savings_km, 3) if self.optimized_savings_km else None,
            "optimized_savings_pct": round(self.optimized_savings_pct, 2) if self.optimized_savings_pct else None,
            "local_knowledge_savings_km": round(self.local_knowledge_savings_km, 3) if self.local_knowledge_savings_km else None,
            "local_knowledge_savings_pct": round(self.local_knowledge_savings_pct, 2) if self.local_knowledge_savings_pct else None,
            "efficiency_ranking": self.efficiency_ranking,
            "deviation_score_optimized": round(self.deviation_score_optimized, 2) if self.deviation_score_optimized else None,
            "deviation_score_local": round(self.deviation_score_local, 2) if self.deviation_score_local else None,
        }


# Constants for estimation formulas
FUEL_PER_KM = 0.15  # Liters per km for garbage truck
CO2_PER_LITER_DIESEL = 2.68  # kg CO2 per liter diesel
STOP_THRESHOLD_SECONDS = 30  # Minimum stop duration to count as a "stop"
STOP_SPEED_THRESHOLD_KMH = 3  # Speed below which we consider it a "stop"


def parse_gpx_file(file_path: str) -> Optional[gpxpy.gpx.GPX]:
    """
    Parse a GPX file and return the GPX object.
    
    Args:
        file_path: Path to the GPX file
        
    Returns:
        Parsed GPX object or None if parsing failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            gpx = gpxpy.parse(f)
        logger.info(f"Successfully parsed GPX file: {file_path}")
        return gpx
    except Exception as e:
        log_error(logger, e, f"Parsing GPX file: {file_path}")
        return None


def parse_gpx_content(gpx_content: str) -> Optional[gpxpy.gpx.GPX]:
    """
    Parse GPX content from a string (e.g., file upload).
    
    Args:
        gpx_content: GPX file content as string
        
    Returns:
        Parsed GPX object or None if parsing failed
    """
    try:
        gpx = gpxpy.parse(gpx_content)
        logger.info("Successfully parsed GPX content from upload")
        return gpx
    except Exception as e:
        log_error(logger, e, "Parsing GPX content from upload")
        return None


def extract_route_from_gpx(gpx: gpxpy.gpx.GPX) -> List[Tuple[float, float]]:
    """
    Extract polyline (lat, lon) from GPX track.
    
    Args:
        gpx: Parsed GPX object
        
    Returns:
        List of (latitude, longitude) tuples
    """
    polyline = []
    
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                polyline.append((point.latitude, point.longitude))
    
    logger.info(f"Extracted {len(polyline)} waypoints from GPX")
    return polyline


def compute_route_metrics(
    polyline: List[Tuple[float, float]],
    gpx: gpxpy.gpx.GPX,
    route_type: str,
) -> RouteMetrics:
    """
    Compute comprehensive metrics for a route.
    
    Args:
        polyline: List of (lat, lon) coordinates
        gpx: Original GPX object (for timestamps)
        route_type: One of "baseline", "optimized", "local_knowledge"
        
    Returns:
        RouteMetrics object with all computed values
    """
    if not polyline:
        return RouteMetrics(
            route_type=route_type,
            total_distance_km=0,
            total_duration_minutes=0,
            avg_speed_kmh=0,
            num_waypoints=0,
            num_stops=0,
            estimated_fuel_liters=0,
            estimated_co2_kg=0,
        )
    
    # Calculate total distance
    total_distance_km = 0.0
    for i in range(1, len(polyline)):
        total_distance_km += haversine(polyline[i-1], polyline[i])
    
    # Extract timestamps from GPX
    timestamps = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time:
                    timestamps.append(point.time)
    
    # Calculate duration
    total_duration_minutes = 0.0
    start_time = None
    end_time = None
    
    if timestamps:
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = (end_time - start_time).total_seconds()
        total_duration_minutes = duration / 60.0
    
    # Calculate average speed
    avg_speed_kmh = 0.0
    if total_duration_minutes > 0:
        avg_speed_kmh = (total_distance_km / total_duration_minutes) * 60.0
    
    # Count stops (points where speed is very low or time gap is large)
    num_stops = 0
    if timestamps and len(timestamps) > 1:
        for i in range(1, len(timestamps)):
            time_gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            if time_gap > STOP_THRESHOLD_SECONDS:
                num_stops += 1
    
    # Estimate fuel and CO2
    estimated_fuel_liters = total_distance_km * FUEL_PER_KM
    estimated_co2_kg = estimated_fuel_liters * CO2_PER_LITER_DIESEL
    
    return RouteMetrics(
        route_type=route_type,
        total_distance_km=total_distance_km,
        total_duration_minutes=total_duration_minutes,
        avg_speed_kmh=avg_speed_kmh,
        num_waypoints=len(polyline),
        num_stops=num_stops,
        estimated_fuel_liters=estimated_fuel_liters,
        estimated_co2_kg=estimated_co2_kg,
        start_time=start_time,
        end_time=end_time,
        start_coord=polyline[0] if polyline else None,
        end_coord=polyline[-1] if polyline else None,
        polyline=polyline,
    )


def compute_deviation_score(
    baseline_polyline: List[Tuple[float, float]],
    comparison_polyline: List[Tuple[float, float]],
) -> float:
    """
    Compute how much a route deviates from the baseline.
    Uses average point-to-route distance.
    
    Args:
        baseline_polyline: Baseline route coordinates
        comparison_polyline: Route to compare against baseline
        
    Returns:
        Deviation score in meters (lower = more similar)
    """
    if not baseline_polyline or not comparison_polyline:
        return float('inf')
    
    total_deviation = 0.0
    sample_points = comparison_polyline[::max(1, len(comparison_polyline) // 50)]  # Sample up to 50 points
    baseline_segments = list(zip(baseline_polyline, baseline_polyline[1:]))
    
    for point in sample_points:
        if baseline_segments:
            min_dist = min(
                _point_to_segment_distance_km(point, start, end)
                for start, end in baseline_segments
            )
        else:
            min_dist = haversine(point, baseline_polyline[0])
        total_deviation += min_dist
    
    avg_deviation_km = total_deviation / len(sample_points)
    return avg_deviation_km * 1000  # Convert to meters


def _angular_distance_radians(
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> float:
    """Calculate great-circle angular distance for latitude/longitude pairs."""
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    haversine_value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    haversine_value = max(0.0, min(1.0, haversine_value))
    return 2 * math.atan2(math.sqrt(haversine_value), math.sqrt(1 - haversine_value))


def _initial_bearing_radians(
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> float:
    """Calculate the initial bearing from one coordinate to another."""
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    delta_lon = lon2 - lon1
    return math.atan2(
        math.sin(delta_lon) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon),
    )


def _point_to_segment_distance_km(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    """Return the shortest spherical distance from a point to a route segment."""
    earth_radius_km = 6371.0
    segment_angle = _angular_distance_radians(start, end)
    point_angle = _angular_distance_radians(start, point)
    if segment_angle == 0:
        return point_angle * earth_radius_km

    bearing_delta = _initial_bearing_radians(start, point) - _initial_bearing_radians(start, end)
    cross_track_angle = math.asin(
        max(-1.0, min(1.0, math.sin(point_angle) * math.sin(bearing_delta)))
    )
    along_track_angle = math.atan2(
        math.sin(point_angle) * math.cos(bearing_delta),
        math.cos(point_angle),
    )
    if 0 <= along_track_angle <= segment_angle:
        return abs(cross_track_angle) * earth_radius_km

    return min(
        _angular_distance_radians(start, point),
        _angular_distance_radians(end, point),
    ) * earth_radius_km


def compare_routes(
    baseline_gpx: Union[gpxpy.gpx.GPX, RouteMetrics, None],
    optimized_gpx: Union[gpxpy.gpx.GPX, RouteMetrics, None],
    local_knowledge_gpx: Union[gpxpy.gpx.GPX, RouteMetrics, None],
) -> RouteComparison:
    """
    Compare up to three routes and compute all metrics.
    
    Args:
        baseline_gpx: Baseline/manual route GPX or pre-computed RouteMetrics
        optimized_gpx: Optimized route GPX or pre-computed RouteMetrics
        local_knowledge_gpx: Local knowledge route GPX or pre-computed RouteMetrics
        
    Returns:
        RouteComparison object with all metrics and comparisons
    """
    # Parse metrics for each route
    baseline_metrics = None
    optimized_metrics = None
    local_metrics = None
    baseline_polyline = None
    optimized_polyline = None
    local_polyline = None
    
    # Process Baseline
    if baseline_gpx:
        if isinstance(baseline_gpx, RouteMetrics):
            baseline_metrics = baseline_gpx
            baseline_polyline = baseline_gpx.polyline
        else:
            baseline_polyline = extract_route_from_gpx(baseline_gpx)
            baseline_metrics = compute_route_metrics(baseline_polyline, baseline_gpx, "baseline")
    
    # Process Optimized
    if optimized_gpx:
        if isinstance(optimized_gpx, RouteMetrics):
            optimized_metrics = optimized_gpx
            optimized_polyline = optimized_gpx.polyline
        else:
            optimized_polyline = extract_route_from_gpx(optimized_gpx)
            optimized_metrics = compute_route_metrics(optimized_polyline, optimized_gpx, "optimized")
    
    # Process Local Knowledge
    if local_knowledge_gpx:
        if isinstance(local_knowledge_gpx, RouteMetrics):
            local_metrics = local_knowledge_gpx
            local_polyline = local_knowledge_gpx.polyline
        else:
            local_polyline = extract_route_from_gpx(local_knowledge_gpx)
            local_metrics = compute_route_metrics(local_polyline, local_knowledge_gpx, "local_knowledge")
    
    # Compute comparisons
    optimized_savings_km = None
    optimized_savings_pct = None
    local_savings_km = None
    local_savings_pct = None
    deviation_optimized = None
    deviation_local = None
    
    if baseline_metrics and optimized_metrics:
        optimized_savings_km = baseline_metrics.total_distance_km - optimized_metrics.total_distance_km
        if baseline_metrics.total_distance_km > 0:
            optimized_savings_pct = (optimized_savings_km / baseline_metrics.total_distance_km) * 100
    
    if baseline_metrics and local_metrics:
        local_savings_km = baseline_metrics.total_distance_km - local_metrics.total_distance_km
        if baseline_metrics.total_distance_km > 0:
            local_savings_pct = (local_savings_km / baseline_metrics.total_distance_km) * 100
    
    # Compute deviation scores
    if baseline_polyline and optimized_polyline:
        deviation_optimized = compute_deviation_score(baseline_polyline, optimized_polyline)
    
    if baseline_polyline and local_polyline:
        deviation_local = compute_deviation_score(baseline_polyline, local_polyline)
    
    # Efficiency ranking (by distance, lower is better)
    efficiency_ranking = []
    routes_with_distance = []
    if baseline_metrics:
        routes_with_distance.append(("baseline", baseline_metrics.total_distance_km))
    if optimized_metrics:
        routes_with_distance.append(("optimized", optimized_metrics.total_distance_km))
    if local_metrics:
        routes_with_distance.append(("local_knowledge", local_metrics.total_distance_km))
    
    routes_with_distance.sort(key=lambda x: x[1])
    efficiency_ranking = [r[0] for r in routes_with_distance]
    
    return RouteComparison(
        baseline=baseline_metrics,
        optimized=optimized_metrics,
        local_knowledge=local_metrics,
        optimized_savings_km=optimized_savings_km,
        optimized_savings_pct=optimized_savings_pct,
        local_knowledge_savings_km=local_savings_km,
        local_knowledge_savings_pct=local_savings_pct,
        efficiency_ranking=efficiency_ranking,
        deviation_score_optimized=deviation_optimized,
        deviation_score_local=deviation_local,
    )


def save_comparison_project(
    comparison: RouteComparison,
    project_name: str,
    output_dir: Path,
) -> str:
    """
    Save a comparison project to JSON for later analysis.
    
    Args:
        comparison: RouteComparison object
        project_name: Name for the project
        output_dir: Directory to save the project
        
    Returns:
        Path to the saved JSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = output_dir.resolve()
    normalized_name = project_name.strip()
    if not normalized_name or "/" in normalized_name or "\\" in normalized_name:
        raise ValueError("Project name must be a filename without directory separators.")
    
    project_data = {
        "project_name": project_name,
        "created_at": datetime.now().isoformat(),
        "comparison": comparison.to_dict(),
    }
    
    output_path = (project_root / f"{normalized_name.replace(' ', '_')}.json").resolve()
    if output_path.parent != project_root:
        raise ValueError("Project path must stay inside the GPX projects directory.")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(project_data, f, indent=2)
    
    logger.info(f"Saved comparison project: {output_path}")
    return str(output_path)


def load_comparison_project(project_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a previously saved comparison project.
    
    Args:
        project_path: Path to the JSON file
        
    Returns:
        Project data dict or None if loading failed
    """
    try:
        with open(project_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log_error(logger, e, f"Loading comparison project: {project_path}")
        return None
