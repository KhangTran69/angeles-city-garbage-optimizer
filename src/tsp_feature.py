"""GPX mission-point TSP analysis for the Streamlit dashboard."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import gpxpy
from gpxpy.gpx import GPX
import pandas as pd
import streamlit as st
from geopy.distance import great_circle
from shapely.geometry import LineString

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except Exception:  # pragma: no cover - exercised only when OR-Tools is unavailable.
    pywrapcp = None
    routing_enums_pb2 = None

Coordinate = tuple[float, float]
PointRecord = dict[str, Any]
MissionPoint = dict[str, Any]
WaypointNode = dict[str, Any]

STOP_SPEED_THRESHOLD_KMH = 1.0
STOP_MIN_DURATION_SECONDS = 5 * 60
NODE_DEDUPLICATION_METERS = 60.0
SIMPLIFICATION_TOLERANCE_METERS = 20.0


def _haversine_km(coord_a: Coordinate, coord_b: Coordinate) -> float:
    """Calculate the great-circle distance between two coordinates in kilometers."""
    import math

    lat1, lon1 = coord_a
    lat2, lon2 = coord_b
    radius_km = 6371.0

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def _path_distance_km(path: Sequence[Coordinate]) -> float:
    """Sum segment distances across a coordinate path."""
    return sum(_haversine_km(path[idx - 1], path[idx]) for idx in range(1, len(path)))


def _format_timestamp(timestamp_seconds: Optional[float]) -> str:
    """Render a timestamp for UI tables."""
    if timestamp_seconds is None:
        return "N/A"

    timestamp = datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc).astimezone()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def _meters_to_latlon_tolerance(meters: float) -> float:
    """Approximate meter-to-degree conversion for simplification."""
    return meters / 111_320.0


def _resolve_graph_path() -> Optional[str]:
    """Return the packaged Angeles City road graph if present."""
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "data" / "angeles_city_network.graphml",
        project_root / "data" / "angeles_city_network_latest.graphml",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _bbox_from_coords(
    coords: Sequence[Coordinate],
    padding_degrees: float = 0.01,
) -> tuple[float, float, float, float]:
    """Return a padded WGS84 bounding box as south, west, north, east."""
    latitudes = [coord[0] for coord in coords]
    longitudes = [coord[1] for coord in coords]
    return (
        min(latitudes) - padding_degrees,
        min(longitudes) - padding_degrees,
        max(latitudes) + padding_degrees,
        max(longitudes) + padding_degrees,
    )


def _build_bbox_graph_key(coords: Sequence[Coordinate]) -> str:
    """Serialize a dynamic OSM graph request into a stable cache key."""
    south, west, north, east = _bbox_from_coords(coords)
    return f"bbox:{south:.5f}:{west:.5f}:{north:.5f}:{east:.5f}"


def _crs_is_wgs84(crs_value: Any) -> bool:
    """Return True when the graph CRS is geographic WGS84."""
    crs_text = str(crs_value).lower()
    return "4326" in crs_text or "wgs84" in crs_text


@st.cache_resource(show_spinner=False)
def _load_osmnx_graph(graph_key: str) -> Any:
    """Load a packaged graph or download one for a GPX-specific bounding box."""
    import osmnx as ox

    if graph_key.startswith("file:"):
        return ox.load_graphml(graph_key.removeprefix("file:"))

    if graph_key.startswith("bbox:"):
        _, south, west, north, east = graph_key.split(":")
        return ox.graph_from_bbox(
            bbox=(
                float(west),
                float(south),
                float(east),
                float(north),
            ),
            network_type="drive",
            simplify=True,
            retain_all=True,
            truncate_by_edge=True,
        )

    raise ValueError(f"Unsupported graph key: {graph_key}")


@st.cache_data(show_spinner=False)
def _graph_bounds_latlon(graph_key: str) -> tuple[float, float, float, float]:
    """Return graph bounds in WGS84 as south, west, north, east."""
    import pyproj

    graph = _load_osmnx_graph(graph_key)
    graph_crs = graph.graph.get("crs", "EPSG:4326")

    longitudes: list[float] = []
    latitudes: list[float] = []

    if _crs_is_wgs84(graph_crs):
        for _, data in graph.nodes(data=True):
            longitudes.append(float(data["x"]))
            latitudes.append(float(data["y"]))
    else:
        transformer = pyproj.Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)
        for _, data in graph.nodes(data=True):
            lon, lat = transformer.transform(float(data["x"]), float(data["y"]))
            longitudes.append(lon)
            latitudes.append(lat)

    return min(latitudes), min(longitudes), max(latitudes), max(longitudes)


def _coords_within_bounds(
    coords: Sequence[Coordinate],
    bounds: tuple[float, float, float, float],
    margin_degrees: float = 0.002,
) -> bool:
    """Check whether all coordinates fall inside a bounding box."""
    south, west, north, east = bounds
    for lat, lon in coords:
        if lat < south - margin_degrees or lat > north + margin_degrees:
            return False
        if lon < west - margin_degrees or lon > east + margin_degrees:
            return False
    return True


def _resolve_osmnx_graph_key(
    coords: Sequence[Coordinate],
    preferred_graph_path: Optional[str],
) -> tuple[str, str]:
    """Choose the best road graph source for the uploaded GPX extent."""
    if preferred_graph_path:
        local_graph_key = f"file:{preferred_graph_path}"
        try:
            local_bounds = _graph_bounds_latlon(local_graph_key)
            if _coords_within_bounds(coords, local_bounds):
                return local_graph_key, "Using packaged Angeles City road graph."
        except Exception:
            pass

    if len(coords) < 2:
        raise ValueError("At least 2 coordinates are required to resolve a road network.")

    dynamic_graph_key = _build_bbox_graph_key(coords)
    _load_osmnx_graph(dynamic_graph_key)
    return dynamic_graph_key, "Using an OSM road graph downloaded for the uploaded GPX area."


def _nearest_node_for_coord(graph: Any, coord: Coordinate) -> Any:
    """Find the nearest graph node, projecting coordinates when needed."""
    import osmnx as ox
    import pyproj

    graph_crs = graph.graph.get("crs", "EPSG:4326")
    if _crs_is_wgs84(graph_crs):
        x_coord, y_coord = coord[1], coord[0]
    else:
        transformer = pyproj.Transformer.from_crs("EPSG:4326", graph_crs, always_xy=True)
        x_coord, y_coord = transformer.transform(coord[1], coord[0])

    return ox.distance.nearest_nodes(graph, X=x_coord, Y=y_coord)


def _build_point_records(
    gpx_points: Sequence[Any],
    point_source: str,
) -> list[PointRecord]:
    """Normalize GPX point objects from tracks, routes, or waypoints."""
    point_records: list[PointRecord] = []
    for point in gpx_points:
        timestamp_value = getattr(point, "time", None)
        timestamp = timestamp_value.timestamp() if timestamp_value else None
        coord = (float(point.latitude), float(point.longitude))
        point_records.append(
            {
                "lat": coord[0],
                "lon": coord[1],
                "coord": coord,
                "timestamp": timestamp,
                "source_type": point_source,
            }
        )
    return point_records


def extract_gpx_point_records(gpx: GPX) -> tuple[list[PointRecord], str]:
    """Extract points from tracks first, then routes, then waypoints."""
    track_points: list[Any] = []
    for track in gpx.tracks:
        for segment in track.segments:
            track_points.extend(segment.points)
    if track_points:
        return _build_point_records(track_points, "track"), "track"

    route_points: list[Any] = []
    for route in gpx.routes:
        route_points.extend(route.points)
    if route_points:
        return _build_point_records(route_points, "route"), "route"

    if gpx.waypoints:
        return _build_point_records(gpx.waypoints, "waypoint"), "waypoint"

    return [], "unknown"


def detect_mission_points(
    point_records: Sequence[PointRecord],
    speed_threshold_kmh: float = STOP_SPEED_THRESHOLD_KMH,
    min_stop_duration_seconds: int = STOP_MIN_DURATION_SECONDS,
) -> list[MissionPoint]:
    """Detect stationary mission-point clusters from timestamped GPX samples."""

    def finalize_cluster(
        start_index: int,
        end_index: int,
        existing_points: list[MissionPoint],
    ) -> None:
        start_time = point_records[start_index]["timestamp"]
        end_time = point_records[end_index]["timestamp"]
        if start_time is None or end_time is None:
            return

        dwell_seconds = end_time - start_time
        if dwell_seconds < min_stop_duration_seconds:
            return

        cluster_records = point_records[start_index : end_index + 1]
        latitudes = [record["lat"] for record in cluster_records]
        longitudes = [record["lon"] for record in cluster_records]
        centroid = (sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes))

        for existing_point in existing_points:
            if _haversine_km(existing_point["coord"], centroid) * 1000 <= NODE_DEDUPLICATION_METERS:
                return

        existing_points.append(
            {
                "label": f"Mission {len(existing_points) + 1}",
                "coord": centroid,
                "arrival_timestamp": start_time,
                "departure_timestamp": end_time,
                "dwell_minutes": dwell_seconds / 60.0,
                "sample_count": len(cluster_records),
                "start_index": start_index,
                "end_index": end_index,
            }
        )

    mission_points: list[MissionPoint] = []
    cluster_start: Optional[int] = None
    cluster_end: Optional[int] = None

    for index in range(1, len(point_records)):
        previous_point = point_records[index - 1]
        current_point = point_records[index]
        previous_timestamp = previous_point["timestamp"]
        current_timestamp = current_point["timestamp"]

        if previous_timestamp is None or current_timestamp is None:
            if cluster_start is not None and cluster_end is not None:
                finalize_cluster(cluster_start, cluster_end, mission_points)
            cluster_start = None
            cluster_end = None
            continue

        elapsed_seconds = current_timestamp - previous_timestamp
        if elapsed_seconds <= 0:
            current_speed_kmh = 0.0
        else:
            distance_km = _haversine_km(previous_point["coord"], current_point["coord"])
            current_speed_kmh = distance_km / (elapsed_seconds / 3600.0)

        is_stationary = current_speed_kmh < speed_threshold_kmh
        if is_stationary:
            if cluster_start is None:
                cluster_start = index - 1
            cluster_end = index
            continue

        if cluster_start is not None and cluster_end is not None:
            finalize_cluster(cluster_start, cluster_end, mission_points)
        cluster_start = None
        cluster_end = None

    if cluster_start is not None and cluster_end is not None:
        finalize_cluster(cluster_start, cluster_end, mission_points)

    return mission_points


def simplify_transit_path(
    point_records: Sequence[PointRecord],
    mission_points: Sequence[MissionPoint],
    tolerance_meters: float = SIMPLIFICATION_TOLERANCE_METERS,
) -> list[Coordinate]:
    """Simplify transit geometry after removing stationary stop samples."""
    mission_indexes: set[int] = set()
    for mission_point in mission_points:
        mission_indexes.update(
            range(mission_point["start_index"], mission_point["end_index"] + 1)
        )

    transit_path = [
        record["coord"]
        for index, record in enumerate(point_records)
        if index not in mission_indexes
    ]
    if len(transit_path) < 2:
        transit_path = [record["coord"] for record in point_records]
    if len(transit_path) < 2:
        return transit_path

    line = LineString([(lon, lat) for lat, lon in transit_path])
    simplified_line = line.simplify(
        _meters_to_latlon_tolerance(tolerance_meters),
        preserve_topology=False,
    )

    if simplified_line.geom_type != "LineString":
        return transit_path

    simplified_path = [(lat, lon) for lon, lat in simplified_line.coords]
    if len(simplified_path) < 2:
        return transit_path
    return simplified_path


def build_tsp_nodes(
    start_coord: Coordinate,
    mission_points: Sequence[MissionPoint],
    dedupe_radius_meters: float = NODE_DEDUPLICATION_METERS,
) -> list[MissionPoint]:
    """Compose the depot plus mission nodes to be visited exactly once."""
    nodes: list[MissionPoint] = [{"label": "Start / Depot", "coord": start_coord}]

    for mission_point in mission_points:
        coord = mission_point["coord"]
        if _haversine_km(start_coord, coord) * 1000 <= dedupe_radius_meters:
            continue

        if any(
            _haversine_km(existing_node["coord"], coord) * 1000 <= dedupe_radius_meters
            for existing_node in nodes
        ):
            continue

        nodes.append(
            {
                "label": f"Mission {len(nodes)}",
                "coord": coord,
                "arrival_timestamp": mission_point["arrival_timestamp"],
                "departure_timestamp": mission_point["departure_timestamp"],
                "dwell_minutes": mission_point["dwell_minutes"],
                "sample_count": mission_point["sample_count"],
            }
        )

    return nodes


@st.cache_data(show_spinner=False)
def parse_gpx_upload_data(
    gpx_bytes: bytes,
    filename: str,
    simplification_tolerance_meters: float = SIMPLIFICATION_TOLERANCE_METERS,
) -> dict[str, Any]:
    """Parse GPX bytes into raw coordinates, timestamps, mission points, and display geometry."""
    if not gpx_bytes.strip():
        raise ValueError(f"{filename} is empty.")

    gpx_text = gpx_bytes.decode("utf-8", errors="ignore").strip()
    if not gpx_text:
        raise ValueError(f"{filename} is empty.")

    gpx = gpxpy.parse(gpx_text)
    point_records, point_source = extract_gpx_point_records(gpx)

    if not point_records:
        raise ValueError(
            f"{filename} does not contain any GPX track, route, or waypoint points."
        )
    if len(point_records) < 3:
        raise ValueError(f"{filename} needs at least 3 GPX points for route analysis.")

    mission_points = detect_mission_points(point_records)
    display_path = simplify_transit_path(
        point_records,
        mission_points,
        tolerance_meters=simplification_tolerance_meters,
    )
    original_path = [record["coord"] for record in point_records]

    return {
        "filename": filename,
        "point_records": point_records,
        "original_path": original_path,
        "display_path": display_path,
        "mission_points": mission_points,
        "original_distance_km": _path_distance_km(original_path),
        "start_coord": original_path[0],
        "end_coord": original_path[-1],
        "timestamped_samples": sum(record["timestamp"] is not None for record in point_records),
        "point_source": point_source,
    }


@st.cache_data(show_spinner=False)
def build_distance_matrix_meters(
    node_coords: tuple[Coordinate, ...],
    method: str = "haversine",
    graph_key: Optional[str] = None,
) -> list[list[int]]:
    """Build an integer distance matrix for OR-Tools."""
    if not node_coords:
        return []

    if method == "ors":
        from src.config import ORS_API_KEY
        from src.map_utils import fetch_ors_matrix
        if not ORS_API_KEY:
            raise ValueError("ORS mode requires a valid ORS_API_KEY in .env.")
        
        matrix = fetch_ors_matrix([list(c) for c in node_coords], ORS_API_KEY)
        if matrix:
            # ORS matrix is in meters
            return [[int(round(dist)) for dist in row] for row in matrix]
        
        # Fallback to OSMnx if ORS fails but graph is available
        if graph_key:
            method = "road"

    if method == "road":
        if not graph_key:
            raise ValueError("Road-network mode requires a resolved OSMnx graph.")

        import networkx as nx

        graph = _load_osmnx_graph(graph_key)
        path_graph = graph.to_undirected()
        nearest_graph_nodes = [
            _nearest_node_for_coord(graph, coord) for coord in node_coords
        ]
        road_matrix: list[list[int]] = []
        for source_index, source_coord in enumerate(node_coords):
            row: list[int] = []
            for target_index, target_coord in enumerate(node_coords):
                if source_index == target_index:
                    row.append(0)
                    continue
                try:
                    meters = nx.shortest_path_length(
                        path_graph,
                        nearest_graph_nodes[source_index],
                        nearest_graph_nodes[target_index],
                        weight="length",
                    )
                    row.append(int(round(meters)))
                except Exception:
                    row.append(int(round(_haversine_km(source_coord, target_coord) * 1000)))
            road_matrix.append(row)
        return road_matrix

    return [
        [int(round(_haversine_km(c1, c2) * 1000)) for c2 in node_coords]
        for c1 in node_coords
    ]




def solve_closed_tsp(distance_matrix_meters: Sequence[Sequence[int]]) -> list[int]:
    """Solve a closed TSP tour that starts and ends at node 0."""
    node_count = len(distance_matrix_meters)
    if node_count == 0:
        return []
    if node_count == 1:
        return [0, 0]
    if pywrapcp is None or routing_enums_pb2 is None:
        raise RuntimeError("OR-Tools is required to solve a TSP tour")

    manager = pywrapcp.RoutingIndexManager(node_count, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix_meters[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.time_limit.seconds = 5

    assignment = routing.SolveWithParameters(search_parameters)
    if assignment is None:
        raise ValueError("OR-Tools could not find a feasible TSP tour.")

    tour: list[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        tour.append(manager.IndexToNode(index))
        index = assignment.Value(routing.NextVar(index))
    tour.append(manager.IndexToNode(index))
    return tour


@st.cache_data(show_spinner=False)
def build_optimized_route_polyline(
    node_coords: tuple[Coordinate, ...],
    visit_order: tuple[int, ...],
    method: str = "haversine",
    graph_key: Optional[str] = None,
) -> list[Coordinate]:
    """Expand an optimized node order into a display polyline."""
    ordered_coords = [node_coords[index] for index in visit_order]
    if len(ordered_coords) < 2:
        return ordered_coords

    if method == "ors":
        from src.config import ORS_API_KEY
        from src.map_utils import fetch_ors_route
        if ORS_API_KEY:
            # fetch_ors_route handles HGV profile and truck dimensions
            route = fetch_ors_route([list(c) for c in ordered_coords], ORS_API_KEY)
            if route:
                return [(float(c[0]), float(c[1])) for c in route]
        
        # Fallback to OSMnx road routing if ORS fails but graph is available
        if graph_key:
            method = "road"

    if method != "road" or not graph_key:
        return ordered_coords



    try:
        import networkx as nx
        import pyproj

        graph = _load_osmnx_graph(graph_key)
        path_graph = graph.to_undirected()
        graph_crs = graph.graph.get("crs", "EPSG:4326")
        transformer = None
        if not _crs_is_wgs84(graph_crs):
            transformer = pyproj.Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)

        def to_latlon(x_coord: float, y_coord: float) -> Coordinate:
            if transformer is None:
                return float(y_coord), float(x_coord)
            lon, lat = transformer.transform(float(x_coord), float(y_coord))
            return float(lat), float(lon)

        def node_coord(node: Any) -> Coordinate:
            node_data = graph.nodes[node]
            return to_latlon(float(node_data["x"]), float(node_data["y"]))

        def edge_segment_coords(source_node: Any, target_node: Any) -> list[Coordinate]:
            edge_bundle = (
                graph.get_edge_data(source_node, target_node)
                or graph.get_edge_data(target_node, source_node)
                or path_graph.get_edge_data(source_node, target_node)
            )
            if not edge_bundle:
                return [node_coord(source_node), node_coord(target_node)]

            edge_data = min(edge_bundle.values(), key=lambda item: item.get("length", 1))
            geometry = edge_data.get("geometry")
            if geometry is None:
                return [node_coord(source_node), node_coord(target_node)]

            coords = [to_latlon(x_coord, y_coord) for x_coord, y_coord in geometry.coords]
            if len(coords) < 2:
                return [node_coord(source_node), node_coord(target_node)]

            source_coord = node_coord(source_node)
            target_coord = node_coord(target_node)
            forward_error = _haversine_km(coords[0], source_coord) + _haversine_km(coords[-1], target_coord)
            reverse_error = _haversine_km(coords[0], target_coord) + _haversine_km(coords[-1], source_coord)
            if reverse_error < forward_error:
                coords.reverse()
            return coords

        route_polyline: list[Coordinate] = []
        graph_nodes = [_nearest_node_for_coord(graph, coord) for coord in ordered_coords]

        for segment_index in range(1, len(graph_nodes)):
            source = graph_nodes[segment_index - 1]
            target = graph_nodes[segment_index]
            shortest_path = nx.shortest_path(path_graph, source, target, weight="length")
            for source_node, target_node in zip(shortest_path[:-1], shortest_path[1:]):
                segment_coords = edge_segment_coords(source_node, target_node)
                if route_polyline and segment_coords:
                    route_polyline.extend(segment_coords[1:])
                else:
                    route_polyline.extend(segment_coords)

        return route_polyline or ordered_coords
    except Exception:
        return ordered_coords


def _build_visit_order_lookup(visit_order: Sequence[int]) -> dict[int, int]:
    """Map mission node indexes to their optimized visit position."""
    return {node_index: order for order, node_index in enumerate(visit_order[1:-1], start=1)}


def _build_tsp_map(
    original_path: Sequence[Coordinate],
    optimized_path: Sequence[Coordinate],
    nodes: Sequence[MissionPoint],
    visit_order: Sequence[int],
) -> Any:
    """Create the folium map for the TSP comparison view."""
    import folium

    center = original_path[0] if original_path else optimized_path[0]
    tsp_map = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron", control_scale=True)

    folium.PolyLine(
        locations=list(original_path),
        color="#7f8c8d",
        weight=4,
        opacity=0.8,
        dash_array="8, 12",
        tooltip="Original Path",
    ).add_to(tsp_map)

    folium.PolyLine(
        locations=list(optimized_path),
        color="#1565c0",
        weight=5,
        opacity=0.95,
        tooltip="Optimized Path",
    ).add_to(tsp_map)

    visit_lookup = _build_visit_order_lookup(visit_order)

    if nodes:
        folium.Marker(
            location=nodes[0]["coord"],
            tooltip="Start / End Depot",
            icon=folium.Icon(color="green", icon="home", prefix="fa"),
        ).add_to(tsp_map)

    for node_index, node in enumerate(nodes[1:], start=1):
        visit_rank = visit_lookup.get(node_index, "-")
        popup_lines = [
            f"<b>{node['label']}</b>",
            f"Visit order: {visit_rank}",
        ]
        if "dwell_minutes" in node:
            popup_lines.append(f"Dwell time: {node['dwell_minutes']:.1f} min")
        if "arrival_timestamp" in node:
            popup_lines.append(f"Arrival: {_format_timestamp(node['arrival_timestamp'])}")
        if "departure_timestamp" in node:
            popup_lines.append(f"Departure: {_format_timestamp(node['departure_timestamp'])}")

        folium.CircleMarker(
            location=node["coord"],
            radius=7,
            color="#0d47a1",
            fill=True,
            fill_color="#42a5f5",
            fill_opacity=0.95,
            tooltip=f"{node['label']} (Visit {visit_rank})",
            popup=folium.Popup("<br>".join(popup_lines), max_width=320),
        ).add_to(tsp_map)

    bounds = [list(coord) for coord in original_path] + [list(coord) for coord in optimized_path]
    if bounds:
        tsp_map.fit_bounds(bounds)

    folium.LayerControl(collapsed=False).add_to(tsp_map)
    return tsp_map


def _render_truck_tab(
    uploaded_file: Any,
    truck_number: int,
    distance_method: str,
    graph_path: Optional[str],
) -> None:
    """Render one uploaded GPX file as its own truck analysis tab."""
    from streamlit_folium import st_folium

    try:
        parsed_route = parse_gpx_upload_data(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as error:
        st.error(str(error))
        return

    if parsed_route["timestamped_samples"] < 2:
        st.warning(
            f"This GPX file was parsed from {parsed_route['point_source']} points, "
            "but it has insufficient timestamps for stop detection. "
            "Mission points require recorded GPX samples with timestamps."
        )
        if parsed_route["point_source"] != "track":
            st.caption(
                "This usually means the file is a planned route export rather than a recorded trip. "
                "For TSP mission-point detection, upload a recorded tracking GPX with timestamps."
            )
        return

    nodes = build_tsp_nodes(parsed_route["start_coord"], parsed_route["mission_points"])
    if len(nodes) < 2:
        st.warning(
            "No mission points were detected with the current rules "
            "(speed below 1 km/h for more than 5 minutes)."
        )
        st.caption(
            "The original path still parsed correctly, but there are not enough stop clusters "
            "to construct a meaningful TSP tour."
        )
        return

    node_coords = tuple(node["coord"] for node in nodes)
    active_distance_method = distance_method
    graph_key: Optional[str] = None

    if distance_method == "road":
        try:
            with st.spinner("Resolving OSM road network for this GPX..."):
                graph_key, graph_status = _resolve_osmnx_graph_key(node_coords, graph_path)
            st.caption(graph_status)
        except Exception as error:
            active_distance_method = "haversine"
            st.warning(
                "OSMnx road-network routing could not be prepared for this file. "
                f"Falling back to Haversine distance. Details: {error}"
            )

    distance_matrix = build_distance_matrix_meters(
        node_coords,
        method=active_distance_method,
        graph_key=graph_key,
    )
    visit_order = solve_closed_tsp(distance_matrix)
    optimized_path = build_optimized_route_polyline(
        node_coords,
        tuple(visit_order),
        method=active_distance_method,
        graph_key=graph_key,
    )

    original_distance_km = parsed_route["original_distance_km"]
    optimized_distance_km = (
        sum(distance_matrix[visit_order[idx - 1]][visit_order[idx]] for idx in range(1, len(visit_order)))
        / 1000.0
    )
    efficiency_gain_pct = (
        ((original_distance_km - optimized_distance_km) / original_distance_km) * 100
        if original_distance_km > 0
        else 0.0
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Original Total Distance", f"{original_distance_km:.2f} km")
    metric_col2.metric(
        "Optimized Total Distance",
        f"{optimized_distance_km:.2f} km",
        delta=f"{original_distance_km - optimized_distance_km:+.2f} km",
    )
    metric_col3.metric("Efficiency Gained", f"{efficiency_gain_pct:.1f}%")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.caption(f"Truck {truck_number}")
    summary_col2.caption(f"Raw GPX points: {len(parsed_route['original_path'])}")
    summary_col3.caption(f"Detected mission points: {len(nodes) - 1}")

    st.caption(
        "Transit geometry was simplified with Douglas-Peucker for cleaner display "
        f"({len(parsed_route['original_path'])} to {len(parsed_route['display_path'])} points)."
    )

    visit_lookup = _build_visit_order_lookup(visit_order)
    mission_rows = []
    for node_index, node in enumerate(nodes[1:], start=1):
        mission_rows.append(
            {
                "Mission Point": node["label"],
                "Visit Order": visit_lookup.get(node_index, "-"),
                "Latitude": round(node["coord"][0], 6),
                "Longitude": round(node["coord"][1], 6),
                "Arrival": _format_timestamp(node.get("arrival_timestamp")),
                "Departure": _format_timestamp(node.get("departure_timestamp")),
                "Dwell (min)": round(float(node.get("dwell_minutes", 0.0)), 1),
            }
        )

    if mission_rows:
        st.dataframe(pd.DataFrame(mission_rows), hide_index=True, use_container_width=True)

    tsp_map = _build_tsp_map(
        parsed_route["display_path"],
        optimized_path,
        nodes,
        visit_order,
    )
    st_folium(
        tsp_map,
        use_container_width=True,
        height=540,
        key=f"tsp-map-{truck_number}-{uploaded_file.name}",
    )


def _integrate_mission_point_tsp_feature() -> None:
    """Render the legacy stop-cluster TSP page inside the Streamlit dashboard."""
    st.markdown(
        """
        <div class='section-title'>TSP Mission Point Optimizer</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='glass-card'>
            <p><b>Upload up to 5 GPX truck tracks and optimize the detected stop sequence as a closed TSP tour.</b></p>
            <p style='margin-bottom: 0;'>
                Mission points are detected from stop clusters where speed stays below 1 km/h for more than 5 minutes.
                The optimized tour always starts and ends at the first GPX coordinate.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    graph_path = _resolve_graph_path()
    distance_options = ["Haversine (fast)"]
    if graph_path:
        distance_options.append("OSMnx Road Network (accurate)")

    controls_col1, controls_col2 = st.columns([3, 2])
    with controls_col1:
        uploaded_files = st.file_uploader(
            "Upload GPX Files (up to 5 trucks)",
            type=["gpx"],
            accept_multiple_files=True,
            key="tsp_gpx_uploads",
        )
    with controls_col2:
        distance_choice = st.selectbox(
            "Distance Matrix Method",
            options=distance_options,
            index=0,
            help=(
                "Use Haversine for fast analysis or the packaged OSMnx road graph "
                "for road-network distances."
            ),
        )

    if not uploaded_files:
        st.info("Upload one or more GPX files to build truck tabs and run the TSP analysis.")
        return

    if len(uploaded_files) > 5:
        st.warning("Only the first 5 GPX files will be processed.")
        uploaded_files = uploaded_files[:5]

    distance_method = "road" if distance_choice.startswith("OSMnx") else "haversine"
    truck_tabs = st.tabs(
        [f"Truck {index + 1}: {uploaded_file.name}" for index, uploaded_file in enumerate(uploaded_files)]
    )

    for index, truck_tab in enumerate(truck_tabs):
        with truck_tab:
            _render_truck_tab(
                uploaded_files[index],
                truck_number=index + 1,
                distance_method=distance_method,
                graph_path=graph_path,
            )

# --- Waypoint GPX + RDP + TSP workflow ---------------------------------------

def extract_track_and_waypoint_records(
    gpx: GPX,
) -> tuple[list[PointRecord], list[WaypointNode], str]:
    """Extract route track points and explicit GPX waypoints from one GPX file."""
    track_points: list[Any] = []
    for track in gpx.tracks:
        for segment in track.segments:
            track_points.extend(segment.points)
    point_source = "track"

    if not track_points:
        for route in gpx.routes:
            track_points.extend(route.points)
        point_source = "route"

    track_records = _build_point_records(track_points, point_source) if track_points else []

    waypoint_records: list[WaypointNode] = []
    for index, waypoint in enumerate(gpx.waypoints, start=1):
        raw_name = getattr(waypoint, "name", None)
        label = str(raw_name).strip() if raw_name else f"Waypoint {index}"
        timestamp_value = getattr(waypoint, "time", None)
        timestamp = timestamp_value.timestamp() if timestamp_value else None
        coord = (float(waypoint.latitude), float(waypoint.longitude))
        waypoint_records.append(
            {
                "label": label,
                "coord": coord,
                "lat": coord[0],
                "lon": coord[1],
                "timestamp": timestamp,
                "elevation_m": getattr(waypoint, "elevation", None),
                "description": getattr(waypoint, "description", None),
                "source_index": index,
            }
        )

    return track_records, waypoint_records, point_source


@st.cache_data(show_spinner=False)
def parse_gpx_spatial_upload_data(gpx_bytes: bytes, filename: str) -> dict[str, Any]:
    """Parse a GPX upload into dense track geometry and named visit waypoints."""
    if not gpx_bytes.strip():
        raise ValueError(f"{filename} is empty.")

    gpx_text = gpx_bytes.decode("utf-8", errors="ignore").strip()
    if not gpx_text:
        raise ValueError(f"{filename} is empty.")

    gpx = gpxpy.parse(gpx_text)
    track_records, waypoints, point_source = extract_track_and_waypoint_records(gpx)

    if len(track_records) < 2:
        raise ValueError(
            f"{filename} needs at least 2 GPX track or route points for track display."
        )
    if not waypoints:
        raise ValueError(f"{filename} does not contain named GPX waypoints/placemarks.")

    original_path = [record["coord"] for record in track_records]
    all_coords = original_path + [waypoint["coord"] for waypoint in waypoints]
    center = (
        sum(coord[0] for coord in all_coords) / len(all_coords),
        sum(coord[1] for coord in all_coords) / len(all_coords),
    )

    return {
        "filename": filename,
        "point_source": point_source,
        "track_records": track_records,
        "original_path": original_path,
        "waypoints": waypoints,
        "original_distance_km": _path_distance_km(original_path),
        "center": center,
        "timestamped_track_samples": sum(
            record["timestamp"] is not None for record in track_records
        ),
    }


def _point_line_distance_degrees(
    point: Coordinate,
    line_start: Coordinate,
    line_end: Coordinate,
) -> float:
    """Return perpendicular point-to-segment distance in coordinate degrees."""
    x0, y0 = point[1], point[0]
    x1, y1 = line_start[1], line_start[0]
    x2, y2 = line_end[1], line_end[0]
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.hypot(x0 - x1, y0 - y1)

    projection = ((x0 - x1) * dx + (y0 - y1) * dy) / ((dx * dx) + (dy * dy))
    projection = max(0.0, min(1.0, projection))
    closest_x = x1 + projection * dx
    closest_y = y1 + projection * dy
    return math.hypot(x0 - closest_x, y0 - closest_y)


def ramer_douglas_peucker(
    points: Sequence[Coordinate],
    epsilon: float,
) -> list[Coordinate]:
    """Simplify a coordinate path using the Ramer-Douglas-Peucker algorithm."""
    if len(points) <= 2 or epsilon <= 0:
        return list(points)

    keep_indexes: set[int] = {0, len(points) - 1}
    stack: list[tuple[int, int]] = [(0, len(points) - 1)]

    while stack:
        start_index, end_index = stack.pop()
        if end_index <= start_index + 1:
            continue

        start_point = points[start_index]
        end_point = points[end_index]
        max_distance = -1.0
        farthest_index: Optional[int] = None

        for index in range(start_index + 1, end_index):
            distance = _point_line_distance_degrees(points[index], start_point, end_point)
            if distance > max_distance:
                max_distance = distance
                farthest_index = index

        if farthest_index is not None and max_distance > epsilon:
            keep_indexes.add(farthest_index)
            stack.append((start_index, farthest_index))
            stack.append((farthest_index, end_index))

    return [points[index] for index in sorted(keep_indexes)]


@st.cache_data(show_spinner=False)
def simplify_path_with_rdp(
    path: tuple[Coordinate, ...],
    epsilon: float,
) -> list[Coordinate]:
    """Cache RDP simplification so Streamlit slider updates stay responsive."""
    return ramer_douglas_peucker(path, epsilon)


@st.cache_data(show_spinner=False)
def build_waypoint_distance_matrix_meters(
    waypoint_coords: tuple[Coordinate, ...],
) -> list[list[int]]:
    """Build a Haversine/great-circle distance matrix for waypoint TSP solving."""
    return [
        [
            0
            if source_index == target_index
            else int(round(great_circle(source_coord, target_coord).meters))
            for target_index, target_coord in enumerate(waypoint_coords)
        ]
        for source_index, source_coord in enumerate(waypoint_coords)
    ]


def solve_waypoint_tsp_nearest_neighbor(
    distance_matrix_meters: Sequence[Sequence[int]],
    return_to_start: bool = False,
) -> list[int]:
    """Fast nearest-neighbor TSP heuristic starting at waypoint 1."""
    node_count = len(distance_matrix_meters)
    if node_count == 0:
        return []
    if node_count == 1:
        return [0, 0] if return_to_start else [0]

    unvisited = set(range(1, node_count))
    order = [0]
    current = 0
    while unvisited:
        next_node = min(unvisited, key=lambda node: distance_matrix_meters[current][node])
        order.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    if return_to_start:
        order.append(0)
    return order


@st.cache_data(show_spinner=False)
def solve_waypoint_tsp(
    distance_matrix_meters: tuple[tuple[int, ...], ...],
    return_to_start: bool = False,
    time_limit_seconds: int = 2,
) -> list[int]:
    """Solve waypoint order with OR-Tools, falling back to nearest neighbor."""
    node_count = len(distance_matrix_meters)
    if node_count == 0:
        return []
    if node_count == 1:
        return [0, 0] if return_to_start else [0]

    if pywrapcp is None or routing_enums_pb2 is None:
        return solve_waypoint_tsp_nearest_neighbor(distance_matrix_meters, return_to_start)

    try:
        dummy_end: Optional[int] = None
        if return_to_start:
            manager = pywrapcp.RoutingIndexManager(node_count, 1, 0)
        else:
            dummy_end = node_count
            manager = pywrapcp.RoutingIndexManager(node_count + 1, 1, [0], [dummy_end])

        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            if dummy_end is not None and (from_node == dummy_end or to_node == dummy_end):
                return 0
            return int(distance_matrix_meters[from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = max(1, int(time_limit_seconds))

        assignment = routing.SolveWithParameters(search_parameters)
        if assignment is None:
            return solve_waypoint_tsp_nearest_neighbor(
                distance_matrix_meters,
                return_to_start,
            )

        order: list[int] = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node < node_count:
                order.append(node)
            index = assignment.Value(routing.NextVar(index))

        end_node = manager.IndexToNode(index)
        if end_node < node_count:
            order.append(end_node)

        if set(order) != set(range(node_count)):
            return solve_waypoint_tsp_nearest_neighbor(
                distance_matrix_meters,
                return_to_start,
            )
        if return_to_start and order[-1] != 0:
            order.append(0)
        return order
    except Exception:
        return solve_waypoint_tsp_nearest_neighbor(distance_matrix_meters, return_to_start)


def calculate_order_distance_km(
    distance_matrix_meters: Sequence[Sequence[int]],
    visit_order: Sequence[int],
) -> float:
    """Calculate total distance for a waypoint visit order."""
    if len(visit_order) < 2:
        return 0.0
    return (
        sum(
            distance_matrix_meters[visit_order[index - 1]][visit_order[index]]
            for index in range(1, len(visit_order))
        )
        / 1000.0
    )


def _waypoint_visit_lookup(visit_order: Sequence[int]) -> dict[int, int]:
    """Map waypoint indexes to their display visit position."""
    display_order = list(visit_order)
    if len(display_order) > 1 and display_order[-1] == display_order[0]:
        display_order = display_order[:-1]
    return {node_index: order for order, node_index in enumerate(display_order, start=1)}


def _build_waypoint_tsp_map(
    original_path: Sequence[Coordinate],
    simplified_path: Sequence[Coordinate],
    optimized_route_path: Sequence[Coordinate],
    waypoints: Sequence[WaypointNode],
    visit_order: Sequence[int],
    center: Coordinate,
) -> Any:
    """Build a Folium map with original track, RDP track, and waypoint TSP route."""
    import folium

    tsp_map = folium.Map(
        location=center,
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )

    original_layer = folium.FeatureGroup(name="Original Track", show=True)
    simplified_layer = folium.FeatureGroup(name="Simplified Track", show=True)
    tsp_layer = folium.FeatureGroup(name="Optimized TSP Route", show=True)
    waypoint_layer = folium.FeatureGroup(name="Waypoints", show=True)

    if original_path:
        folium.PolyLine(
            locations=list(original_path),
            color="#5f6b7a",
            weight=3,
            opacity=0.25,
            tooltip="Original Track",
            smooth_factor=2,
        ).add_to(original_layer)

    if simplified_path:
        folium.PolyLine(
            locations=list(simplified_path),
            color="#ff8f00",
            weight=4,
            opacity=0.9,
            tooltip="Simplified Track (RDP)",
            smooth_factor=1,
        ).add_to(simplified_layer)

    if optimized_route_path:
        folium.PolyLine(
            locations=list(optimized_route_path),
            color="#1565c0",
            weight=4,
            opacity=0.95,
            tooltip="Optimized TSP Road Route",
        ).add_to(tsp_layer)

    visit_lookup = _waypoint_visit_lookup(visit_order)
    for waypoint_index, waypoint in enumerate(waypoints):
        visit_rank = visit_lookup.get(waypoint_index, "-")
        popup_lines = [
            f"<b>{waypoint['label']}</b>",
            f"Optimized visit: {visit_rank}",
            f"Original placemark: {waypoint['source_index']}",
            f"Lat/Lon: {waypoint['lat']:.6f}, {waypoint['lon']:.6f}",
        ]
        if waypoint.get("timestamp") is not None:
            popup_lines.append(f"Time: {_format_timestamp(waypoint['timestamp'])}")

        folium.CircleMarker(
            location=waypoint["coord"],
            radius=7,
            color="#0d47a1",
            fill=True,
            fill_color="#64b5f6",
            fill_opacity=0.95,
            tooltip=f"{visit_rank}. {waypoint['label']}",
            popup=folium.Popup("<br>".join(popup_lines), max_width=320),
        ).add_to(waypoint_layer)

    original_layer.add_to(tsp_map)
    simplified_layer.add_to(tsp_map)
    tsp_layer.add_to(tsp_map)
    waypoint_layer.add_to(tsp_map)

    bounds = [list(coord) for coord in original_path] + [
        list(waypoint["coord"]) for waypoint in waypoints
    ]
    if bounds:
        tsp_map.fit_bounds(bounds)

    folium.LayerControl(collapsed=False).add_to(tsp_map)
    return tsp_map


def _render_waypoint_tsp_analysis(
    uploaded_file: Any,
    file_index: int = 0,
) -> None:
    """Render one GPX simplification and waypoint TSP analysis."""
    from streamlit_folium import st_folium

    try:
        parsed = parse_gpx_spatial_upload_data(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as error:
        st.error(str(error))
        return

    widget_suffix = f"{file_index}-{uploaded_file.name}"
    controls_col1, controls_col2, controls_col3 = st.columns([2, 1, 1])
    with controls_col1:
        epsilon = st.slider(
            "Epsilon",
            min_value=0.00001,
            max_value=0.00500,
            value=0.00050,
            step=0.00001,
            format="%.5f",
            help="Coordinate-degree tolerance for Ramer-Douglas-Peucker simplification.",
            key=f"waypoint-tsp-epsilon-{widget_suffix}",
        )
    with controls_col2:
        return_to_start = st.checkbox(
            "Return to first waypoint",
            value=False,
            key=f"waypoint-tsp-return-{widget_suffix}",
        )
    with controls_col3:
        solver_seconds = st.number_input(
            "TSP seconds",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            key=f"waypoint-tsp-seconds-{widget_suffix}",
        )
    
    # Routing Engine Selector
    routing_col1, routing_col2 = st.columns(2)
    with routing_col1:
        from src.config import ORS_API_KEY
        engine_options = ["OSMnx Road Network", "Straight Line (Haversine)"]
        if ORS_API_KEY:
            engine_options.insert(0, "OpenRouteService (Truck Feasible)")
            
        selected_engine = st.selectbox(
            "Routing Engine",
            engine_options,
            key=f"waypoint-tsp-engine-{widget_suffix}",
            help="Choose between high-fidelity truck routing (ORS), local road network (OSMnx), or straight-line distance."
        )
        
        if selected_engine.startswith("OpenRouteService"):
            active_distance_method = "ors"
        elif selected_engine.startswith("OSMnx"):
            active_distance_method = "road"
        else:
            active_distance_method = "haversine"


    original_path = parsed["original_path"]
    waypoints = parsed["waypoints"]
    simplified_path = simplify_path_with_rdp(tuple(original_path), epsilon)
    waypoint_coords = tuple(waypoint["coord"] for waypoint in waypoints)
    graph_path = _resolve_graph_path()
    graph_key: Optional[str] = None
    graph_status = ""

    try:
        # Always try to resolve graph key if using a road-network method
        if active_distance_method in ["road", "ors"]:
            try:
                graph_key, graph_status = _resolve_osmnx_graph_key(waypoint_coords, graph_path)
            except Exception:
                # If graph resolution fails but we are using ORS, we can still proceed
                if active_distance_method == "road":
                    raise
        
        distance_matrix = build_distance_matrix_meters(
            waypoint_coords,
            method=active_distance_method,
            graph_key=graph_key,
        )
    except Exception as error:
        active_distance_method = "haversine"
        distance_matrix = build_waypoint_distance_matrix_meters(waypoint_coords)
        st.warning(
            "Road-network routing could not be prepared for this GPX, so the app fell back "
            f"to Haversine waypoint distances. Details: {error}"
        )



    visit_order = solve_waypoint_tsp(
        tuple(tuple(row) for row in distance_matrix),
        return_to_start=return_to_start,
        time_limit_seconds=int(solver_seconds),
    )
    optimized_route_path = build_optimized_route_polyline(
        waypoint_coords,
        tuple(visit_order),
        method=active_distance_method,
        graph_key=graph_key,
    )
    tsp_distance_km = calculate_order_distance_km(distance_matrix, visit_order)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Original Points", f"{len(original_path):,}")
    metric_col2.metric("Simplified (RDP)", f"{len(simplified_path):,}")
    metric_col3.metric("Visit Waypoints", f"{len(waypoints):,}")
    metric_col4.metric("RDP Reduction", f"{(1 - len(simplified_path)/len(original_path))*100:.1f}%")

    # New Distance Comparison Feature
    st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)
    orig_dist_km = parsed["original_distance_km"]
    dist_diff_km = orig_dist_km - tsp_distance_km
    efficiency_gain = (dist_diff_km / orig_dist_km * 100) if orig_dist_km > 0 else 0

    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    d_col1.metric("Original Track", f"{orig_dist_km:.2f} km")
    d_col2.metric("TSP Optimized", f"{tsp_distance_km:.2f} km")
    d_col3.metric("Distance Difference", f"{dist_diff_km:+.2f} km", delta=f"{dist_diff_km:+.2f} km")
    d_col4.metric("Efficiency Gain", f"{efficiency_gain:.1f}%")

    st.caption(
        f"Parsed {len(original_path):,} {parsed['point_source']} points and "
        f"{len(waypoints):,} named waypoints. Comparison measures the recorded track "
        "against the optimal road-network sequence through waypoints."
    )

    # Google Maps API Verification for TSP
    import os
    google_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if google_api_key and "your_google_maps" not in google_api_key:
        with st.expander("🔍 Verify TSP Route with Google Maps API"):
            st.write("Cross-reference the optimized TSP sequence with Google's proprietary routing engine.")
            if st.button("🚀 Run Google Verification", key=f"verify-google-{widget_suffix}"):
                from src.google_utils import verify_optimized_route
                with st.spinner("Fetching Google Maps data..."):
                    v_report = verify_optimized_route(
                        f"TSP Route ({uploaded_file.name})",
                        [waypoint_coords[i] for i in visit_order],
                        tsp_distance_km,
                        google_api_key
                    )

                
                if v_report["success"]:
                    st.success("✅ Google Maps verification complete!")
                    vc1, vc2, vc3 = st.columns(3)
                    vc1.metric("GA Distance", f"{v_report['current_distance_km']:.2f} km")
                    vc2.metric("Google Distance", f"{v_report['google_distance_km']:.2f} km")
                    vc3.metric("Delta", f"{v_report['delta_km']:+.2f} km", delta=f"{v_report['delta_pct']:+.1f}%")
                    st.caption(f"Google Estimated Duration: {v_report['google_duration_mins']:.1f} mins")
                else:
                    st.error(f"Google Verification Error: {v_report['error']}")


    if active_distance_method == "road":
        st.success(
            f"Road routing active: {graph_status} TSP ordering, distance, and blue route "
            "use shortest paths on the road network."
        )
    visit_lookup = _waypoint_visit_lookup(visit_order)
    order_rows = []
    for waypoint_index, waypoint in enumerate(waypoints):
        order_rows.append(
            {
                "Optimized Visit": visit_lookup.get(waypoint_index, "-"),
                "Placemark": waypoint["label"],
                "Original Order": waypoint["source_index"],
                "Latitude": round(waypoint["lat"], 7),
                "Longitude": round(waypoint["lon"], 7),
            }
        )
    st.dataframe(
        pd.DataFrame(order_rows).sort_values("Optimized Visit"),
        hide_index=True,
        use_container_width=True,
    )

    route_map = _build_waypoint_tsp_map(
        original_path=original_path,
        simplified_path=simplified_path,
        optimized_route_path=optimized_route_path,
        waypoints=waypoints,
        visit_order=visit_order,
        center=parsed["center"],
    )
    st_folium(
        route_map,
        use_container_width=True,
        height=620,
        key=f"waypoint-tsp-map-{widget_suffix}-{epsilon}-{return_to_start}",
    )


def integrate_tsp_feature() -> None:
    """Render the GPX waypoint simplification and TSP page."""
    st.markdown(
        """
        <div class='section-title'>GPX Simplification + Waypoint TSP</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='glass-card'>
            <p><b>Upload one GPX file with dense track points and named waypoints.</b></p>
            <p style='margin-bottom: 0;'>
                The map overlays the original track, the RDP simplified track, and a road-network optimized route through the GPX placemarks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload GPX Files",
        type=["gpx"],
        accept_multiple_files=True,
        key="waypoint_tsp_gpx_uploads",
    )

    if not uploaded_files:
        st.info("Upload one or more GPX files to solve each route with road-network TSP.")
        return

    if len(uploaded_files) == 1:
        _render_waypoint_tsp_analysis(uploaded_files[0], file_index=0)
        return

    st.caption(f"Solving {len(uploaded_files)} GPX files. Each tab has its own Epsilon and TSP controls.")
    tabs = st.tabs([f"{index + 1}. {uploaded_file.name}" for index, uploaded_file in enumerate(uploaded_files)])
    for index, tab in enumerate(tabs):
        with tab:
            _render_waypoint_tsp_analysis(uploaded_files[index], file_index=index)

