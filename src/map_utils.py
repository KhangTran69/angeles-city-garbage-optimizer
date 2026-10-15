"""
Folium map rendering utilities for Angeles City Garbage Route Optimizer.
Realistic Road Routing: OpenRouteService (ORS) + Standalone Polyline Decoder.
"""
import requests
import json
import html
import hashlib
import os
import streamlit as st
from typing import List, Optional, Dict, Any
import numpy as np
from src.data import DEPOT_COORD, BARANGAY_COORDS, DEPOT, GOOGLE_ROUTES
from src.config import VEHICLE_COLORS, ORS_BASE_URL
from src.logger import log_api_call, logger

MAP_CACHE_VERSION = "2026-04-21-map-cache-v3"
FULL_FLEET_MAX_ROUTE_POINTS = 120


def decode_polyline(polyline_str: str) -> List[List[float]]:
    """Decode a Google-style encoded polyline string into coordinates."""
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'lat': 0, 'lng': 0}

    while index < len(polyline_str):
        for unit in ['lat', 'lng']:
            shift, result = 0, 0
            while True:
                byte = ord(polyline_str[index]) - 63
                index += 1
                result |= (byte & 0x1f) << shift
                shift += 5
                if not byte >= 0x20:
                    break
            if (result & 1):
                changes[unit] = ~(result >> 1)
            else:
                changes[unit] = (result >> 1)

        lat += changes['lat']
        lng += changes['lng']
        coordinates.append([lat / 100000.0, lng / 100000.0])

    return coordinates

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_ors_route(coords: List[List[float]], api_key: str) -> Optional[List[List[float]]]:
    """Fetch high-fidelity road routes from OpenRouteService (GeoJSON Protocol)."""
    import time
    if not api_key:
        logger.warning("ORS API key not provided, using straight-line fallback")
        return None
    
    # Logic Sync: Standardize on GeoJSON for more robust coordinate handling
    url = f"{ORS_BASE_URL}/geojson"
    
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': api_key,
        'Content-Type': 'application/json; charset=utf-8'
    }
    # ORS expects [lon, lat]
    from src.config import (
        DEFAULT_TRUCK_HEIGHT, DEFAULT_TRUCK_WIDTH, 
        DEFAULT_TRUCK_LENGTH, DEFAULT_TRUCK_WEIGHT
    )
    
    body = {
        "coordinates": [[c[1], c[0]] for c in coords],
        "options": {
            "profile_params": {
                "restrictions": {
                    "height": DEFAULT_TRUCK_HEIGHT,
                    "width": DEFAULT_TRUCK_WIDTH,
                    "length": DEFAULT_TRUCK_LENGTH,
                    "weight": DEFAULT_TRUCK_WEIGHT
                }
            }
        }
    }

    
    start_time = time.time()
    try:
        response = requests.post(url, json=body, headers=headers, timeout=10)
        response_time_ms = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            log_api_call(logger, "ORS", "/geojson", True, response_time_ms=response_time_ms)
            data = response.json()
            # Extract road-curved coordinates from GeoJSON LineString
            line_coords = data['features'][0]['geometry']['coordinates']
            # Reformat to [lat, lon] for Folium
            return [[c[1], c[0]] for c in line_coords]
        elif response.status_code == 401:
            log_api_call(logger, "ORS", "/geojson", False, "Unauthorized - check API key", response_time_ms)
            st.error("🔑 ORS API Key Unauthorized. Check your .env credentials.")
            return None
        elif response.status_code == 429:
            log_api_call(logger, "ORS", "/geojson", False, "Rate limit exceeded", response_time_ms)
            st.warning("⏳ ORS Rate Limit Reached. Falling back to local network approximation.")
            return None
        else:
            log_api_call(logger, "ORS", "/geojson", False, f"HTTP {response.status_code}", response_time_ms)
            return None
    except requests.Timeout:
        response_time_ms = (time.time() - start_time) * 1000
        log_api_call(logger, "ORS", "/geojson", False, "Timeout", response_time_ms)
        st.warning("⏱️ ORS request timed out. Using straight-line fallback.")
        return None
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        log_api_call(logger, "ORS", "/geojson", False, str(e), response_time_ms)
        return None


def _map_view_label(view_mode: Any) -> str:
    """Normalize a view selector into the cache file suffix."""
    if view_mode == "all":
        return "all"
    return str(int(view_mode))


def _reduce_route_coords(coords: List[List[float]], max_points: int = FULL_FLEET_MAX_ROUTE_POINTS) -> List[List[float]]:
    """Trim overview routes so the full-fleet map stays responsive."""
    if len(coords) <= max_points:
        return coords

    sample_indices = np.linspace(0, len(coords) - 1, num=max_points, dtype=int)
    reduced = [coords[idx] for idx in sample_indices]
    return reduced


def _build_stop_popup_html(stop: str, color: str, vehicle_idx: int, stop_idx: int, load_kg: int, eta: str) -> str:
    """Render the shared stop popup UI for both single-truck and full-fleet views."""
    return f"""
    <div style='font-family:sans-serif; min-width:180px;'>
        <div style='background:{color}; color:white; padding:8px; border-radius:6px 6px 0 0; font-weight:700;'>🏘️ {html.escape(stop)}</div>
        <div style='padding:10px; background:white; border:1px solid #ddd; border-top:none; border-radius:0 0 6px 6px; font-size:12px;'>
            <b>Vehicle:</b> {vehicle_idx + 1} | <b>Stop:</b> {stop_idx + 1}<br>
            <b>Load:</b> {load_kg} kg | <b>ETA:</b> {eta}
        </div>
    </div>
    """


def _build_vehicle_popup_html(
    route: List[str],
    color: str,
    vehicle_idx: int,
    distance_km: float,
    waste_dict: Optional[Dict[str, int]] = None,
) -> str:
    """Render a clickable route summary card for full-fleet selection."""
    total_load = sum((waste_dict or {}).get(stop, 0) for stop in route)
    stop_preview = ", ".join(route[:4]) if route else "Depot standby"
    if len(route) > 4:
        stop_preview = f"{stop_preview}, ..."

    return f"""
    <div style='font-family:sans-serif; min-width:210px;'>
        <div style='background:{color}; color:white; padding:10px; border-radius:8px 8px 0 0; font-weight:700;'>🚛 Vehicle {vehicle_idx + 1} Dispatch</div>
        <div style='padding:12px; background:white; border:1px solid #ddd; border-top:none; border-radius:0 0 8px 8px; font-size:12px; line-height:1.5;'>
            <b>Distance:</b> {distance_km:.2f} km<br>
            <b>Stops:</b> {len(route)} | <b>Load:</b> {total_load} kg<br>
            <b>Coverage:</b> {html.escape(stop_preview)}
        </div>
    </div>
    """

def build_folium_route_map(
    routes: List[List[str]],
    dists: List[float],
    colors: List[str] = VEHICLE_COLORS,
    view_mode: Any = "all",
    road_graph = None,
    distance_matrix: Optional[np.ndarray] = None,
    point_to_index: Optional[Dict[str, int]] = None,
    truck_params: Optional[Dict[str, Any]] = None,
    vehicle_capacity_kg: Optional[int] = None,
    waste_dict: Optional[Dict[str, int]] = None,
    simulation_time_min: int = 9999,
    ors_key: str = ""
) -> Any:
    """Create an interactive Tactical Dashboard Map with Realistic Road Routing."""
    import folium
    from folium import plugins

    is_full_fleet_view = view_mode == "all" and len(routes) > 1
    m = folium.Map(
        location=DEPOT_COORD, 
        zoom_start=13, 
        tiles="CartoDB dark_matter", 
        attribution_control=False,
        control_scale=True,
        prefer_canvas=is_full_fleet_view,
    )

    plugins.Fullscreen(position="topright", title="Presentation Mode").add_to(m)
    plugins.MiniMap(toggle_display=True, position="bottomright").add_to(m)

    # Depot hub
    folium.Marker(
        location=DEPOT_COORD,
        popup="<div style='background:#b71c1c; color:white; padding:5px 10px; border-radius:4px; font-weight:700;'>🏆 GSO DEPOT (HUB)</div>",
        icon=folium.Icon(color="red", icon="home", prefix="fa")
    ).add_to(m)

    to_render = []
    if view_mode == "all":
        to_render = list(enumerate(routes))
    else:
        try:
            idx = int(view_mode)
            if 0 <= idx < len(routes): to_render = [(idx, routes[idx])]
        except: pass

    for i, route in to_render:
        if not route: continue
        color = colors[i % len(colors)]
        group = folium.FeatureGroup(name=f"🚛 Vehicle {i+1} Dispatch")
        
        full_stops = [DEPOT] + route + [DEPOT]
        base_coords = [BARANGAY_COORDS.get(s, DEPOT_COORD) if s != DEPOT else DEPOT_COORD for s in full_stops]

        # Override the base plotting path with the realistic ground-truth maps route (Truck 1-5).
        if i < len(GOOGLE_ROUTES):
            base_coords = GOOGLE_ROUTES[i]
        # ---------------------------------

        # 🚀 Tiered Routing Logic
        road_coords = None
        if ors_key:
            road_coords = fetch_ors_route(base_coords, ors_key)
        
        # Fallback to base polyline if ORS fails
        final_coords = road_coords if road_coords else base_coords
        distance_km = float(dists[i] if i < len(dists) else 0.0)
        route_tooltip = f"Vehicle {i+1} • {len(route)} stops • {distance_km:.2f} km"
        route_popup_html = _build_vehicle_popup_html(route, color, i, distance_km, waste_dict)

        if is_full_fleet_view:
            overview_coords = _reduce_route_coords(final_coords)
            folium.GeoJson(
                data={
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[coord[1], coord[0]] for coord in overview_coords],
                    },
                },
                style_function=lambda _, route_color=color: {
                    "color": route_color,
                    "weight": 5,
                    "opacity": 0.72,
                    "lineCap": "round",
                    "lineJoin": "round",
                },
                highlight_function=lambda _, route_color=color: {
                    "color": route_color,
                    "weight": 8,
                    "opacity": 1.0,
                },
                tooltip=folium.Tooltip(route_tooltip, sticky=False),
            ).add_to(group)

            dispatch_loc = overview_coords[len(overview_coords) // 2]
            dispatch_icon_html = f"""
            <div style="
                background:{color};
                border:2px solid white;
                border-radius:18px;
                min-width:38px;
                height:24px;
                display:flex;
                align-items:center;
                justify-content:center;
                color:white;
                font-weight:800;
                font-size:11px;
                font-family:'Outfit', sans-serif;
                box-shadow:0 4px 8px rgba(0,0,0,0.35);
                padding:0 8px;
            ">T-{i + 1}</div>
            """
            folium.Marker(
                location=dispatch_loc,
                icon=folium.DivIcon(
                    icon_size=(38, 24),
                    icon_anchor=(19, 12),
                    html=dispatch_icon_html,
                ),
                popup=folium.Popup(route_popup_html, max_width=280),
                tooltip=folium.Tooltip(f"Open Vehicle {i+1} dispatch", sticky=False),
            ).add_to(group)
        else:
            # Glow layer for focused single-truck inspection.
            folium.PolyLine(
                locations=final_coords,
                color=color,
                weight=10,
                opacity=0.2,
                tooltip=f"Glow Effect: V{i+1}"
            ).add_to(group)

            plugins.AntPath(
                locations=final_coords,
                color=color,
                weight=5,
                opacity=0.9,
                pulse_color="#ffffff",
                dash_array=[15, 30],
                delay=1500,
                tooltip=f"Optimized Road Route: Vehicle {i+1}"
            ).add_to(group)

        # 📍 Visual Stops & Popups
        sched = compute_stop_schedule(route, dists[i] if i < len(dists) else 0.0, road_graph, distance_matrix, point_to_index, vehicle_idx=i)
        for idx, stop in enumerate(route):
            loc = BARANGAY_COORDS.get(stop)
            if not loc:
                try:
                    # Snapping to the closest visual milestone along the custom map line
                    if i < len(GOOGLE_ROUTES) and len(GOOGLE_ROUTES[i]) > 0:
                        ratio = (idx + 1) / (len(route) + 1)
                        loc_idx = int(ratio * len(GOOGLE_ROUTES[i]))
                        loc = GOOGLE_ROUTES[i][min(loc_idx, len(GOOGLE_ROUTES[i])-1)]
                    else:
                        loc = DEPOT_COORD
                except Exception:
                    loc = DEPOT_COORD

            load_kg = waste_dict.get(stop, 0) if waste_dict else 0
            eta = sched[idx]["eta_label"] if idx < len(sched) else "N/A"
            p_html = _build_stop_popup_html(stop, color, i, idx, load_kg, eta)

            if is_full_fleet_view:
                full_fleet_tooltip = (
                    f"Vehicle {i+1} • Stop {idx+1}<br>"
                    f"{html.escape(stop)}<br>"
                    f"Load: {load_kg} kg • ETA: {eta}"
                )
                folium.CircleMarker(
                    location=loc,
                    radius=5,
                    color="#ffffff",
                    weight=1,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    tooltip=folium.Tooltip(full_fleet_tooltip, sticky=False),
                    popup=folium.Popup(p_html, max_width=250),
                ).add_to(group)
            else:
                icon_html = f"""
                <div style="
                    background-color: {color};
                    border: 2px solid white;
                    border-radius: 50%;
                    width: 30px;
                    height: 30px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: 800;
                    font-size: 14px;
                    font-family: 'Outfit', sans-serif;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.4);
                ">{idx + 1}</div>
                """

                folium.Marker(
                    location=loc,
                    icon=folium.DivIcon(
                        icon_size=(30, 30),
                        icon_anchor=(15, 15),
                        html=icon_html
                    ),
                    popup=folium.Popup(p_html, max_width=250)
                ).add_to(group)

        group.add_to(m)

    folium.LayerControl(position="topright").add_to(m)
    return m


def _map_cache_dir() -> str:
    """Return the on-disk cache directory for rendered tactical map views."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(project_root, "cache", "map_views")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def build_map_cache_signature(
    routes: List[List[str]],
    dists: List[float],
    waste_dict: Optional[Dict[str, int]] = None,
    ors_key: str = "",
) -> str:
    """Create a stable signature so cached map files match the active optimization results."""
    payload = {
        "cache_version": MAP_CACHE_VERSION,
        "routes": routes,
        "dists": [round(float(distance), 3) for distance in dists],
        "waste_dict": waste_dict or {},
        "ors_enabled": bool(ors_key),
        "coords": BARANGAY_COORDS,
        "colors": VEHICLE_COLORS,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def get_map_view_cache_path(signature: str, view_mode: Any) -> str:
    """Return the on-disk HTML path for a specific tactical map view."""
    return os.path.join(_map_cache_dir(), f"{signature}_{_map_view_label(view_mode)}.html")


def get_missing_map_views(signature: str, view_modes: List[Any]) -> List[Any]:
    """List which requested view caches are still missing on disk."""
    return [view_mode for view_mode in view_modes if not os.path.exists(get_map_view_cache_path(signature, view_mode))]


def get_cached_map_view_html(
    routes: List[List[str]],
    dists: List[float],
    colors: List[str] = VEHICLE_COLORS,
    view_mode: Any = "all",
    road_graph = None,
    distance_matrix: Optional[np.ndarray] = None,
    point_to_index: Optional[Dict[str, int]] = None,
    waste_dict: Optional[Dict[str, int]] = None,
    ors_key: str = "",
    signature: str = "",
) -> str:
    """Return a rendered map view from disk cache, writing it once on first use."""
    active_signature = signature or build_map_cache_signature(routes, dists, waste_dict, ors_key)
    cache_path = get_map_view_cache_path(active_signature, view_mode)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            return cache_file.read()

    map_obj = build_folium_route_map(
        routes,
        dists,
        colors,
        view_mode,
        road_graph,
        distance_matrix,
        point_to_index,
        waste_dict=waste_dict,
        ors_key=ors_key,
    )
    rendered_html = map_obj.get_root().render()
    temp_cache_path = f"{cache_path}.tmp"
    with open(temp_cache_path, "w", encoding="utf-8") as cache_file:
        cache_file.write(rendered_html)
    os.replace(temp_cache_path, cache_path)
    return rendered_html


def warm_cached_map_views(
    routes: List[List[str]],
    dists: List[float],
    view_modes: List[Any],
    colors: List[str] = VEHICLE_COLORS,
    road_graph = None,
    distance_matrix: Optional[np.ndarray] = None,
    point_to_index: Optional[Dict[str, int]] = None,
    waste_dict: Optional[Dict[str, int]] = None,
    ors_key: str = "",
    signature: str = "",
) -> None:
    """Ensure a set of tactical map views has been rendered to disk."""
    for view_mode in view_modes:
        get_cached_map_view_html(
            routes=routes,
            dists=dists,
            colors=colors,
            view_mode=view_mode,
            road_graph=road_graph,
            distance_matrix=distance_matrix,
            point_to_index=point_to_index,
            waste_dict=waste_dict,
            ors_key=ors_key,
            signature=signature,
        )

def compute_stop_schedule(
    route: List[str],
    total_dist_km: float,
    road_graph,
    distance_matrix: Optional[np.ndarray],
    point_to_index: Optional[Dict[str, int]],
    vehicle_idx: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Estimate arrival times for each stop."""
    from src.config import MAP_AVG_SPEED_KMH, STOP_SERVICE_MINUTES
    from src.utils import haversine
    
    schedule, curr_time, curr_point = [], 0.0, DEPOT
    
    # 🎯 GROUND-TRUTH ETA Logic (Truck 1-5)
    # If this is an override vehicle, we calculate distance by snapping to the Google path
    is_gt = vehicle_idx is not None and vehicle_idx < len(GOOGLE_ROUTES)
    gt_path = GOOGLE_ROUTES[vehicle_idx] if is_gt else []

    for i, stop in enumerate(route):
        d = 0.0
        
        if is_gt:
            # Snap previous point and current point to the GOOGLE_ROUTES milestones
            p1_idx = int((i) / (len(route) + 1) * len(gt_path))
            p2_idx = int((i + 1) / (len(route) + 1) * len(gt_path))
            
            p1 = gt_path[min(p1_idx, len(gt_path)-1)]
            p2 = gt_path[min(p2_idx, len(gt_path)-1)]
            d = haversine(p1, p2)
        elif distance_matrix is not None and point_to_index is not None:
            idx1, idx2 = point_to_index.get(curr_point), point_to_index.get(stop)
            if idx1 is not None and idx2 is not None: d = float(distance_matrix[idx1][idx2])
        
        # 🚗 Speed adjustment: Manual routes usually in higher traffic zones
        effective_speed = (MAP_AVG_SPEED_KMH * 0.7) if is_gt else (MAP_AVG_SPEED_KMH or 1)
        
        curr_time += (d / effective_speed) * 60
        h, m = int(curr_time // 60), int(curr_time % 60)
        schedule.append({"order": i+1, "name": stop, "eta_label": f"{h}h {m}m" if h>0 else f"{m}m"})
        curr_time += STOP_SERVICE_MINUTES
        curr_point = stop
    return schedule
def build_gpx_comparison_map(comparison: Any) -> Any:
    """Build a Folium map for comparing baseline, optimized, and local knowledge routes."""
    import folium
    from folium import plugins

    # Find a reasonable center
    all_points = []
    if comparison.baseline: all_points.extend(comparison.baseline.polyline)
    if comparison.optimized: all_points.extend(comparison.optimized.polyline)
    if comparison.local_knowledge: all_points.extend(comparison.local_knowledge.polyline)

    if not all_points:
        center = DEPOT_COORD
    else:
        center = [
            sum(p[0] for p in all_points) / len(all_points),
            sum(p[1] for p in all_points) / len(all_points)
        ]

    m = folium.Map(
        location=center,
        zoom_start=14,
        tiles="CartoDB dark_matter",
        attribution_control=False,
    )

    plugins.Fullscreen().add_to(m)

    if comparison.baseline and comparison.baseline.polyline:
        folium.PolyLine(
            locations=comparison.baseline.polyline,
            color="#ef5350",
            weight=5,
            opacity=0.8,
            tooltip="🔴 Baseline Route",
            name="Baseline (Manual)"
        ).add_to(m)
        
        # Start/End markers
        folium.CircleMarker(
            location=comparison.baseline.polyline[0],
            radius=6, color="#ef5350", fill=True, popup="Baseline Start"
        ).add_to(m)
        folium.Marker(
            location=comparison.baseline.polyline[-1],
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
            popup="Baseline End"
        ).add_to(m)

    if comparison.optimized and comparison.optimized.polyline:
        folium.PolyLine(
            locations=comparison.optimized.polyline,
            color="#42a5f5",
            weight=5,
            opacity=0.9,
            tooltip="🔵 Optimized Route",
            name="Optimized (GA)"
        ).add_to(m)
        
        folium.CircleMarker(
            location=comparison.optimized.polyline[0],
            radius=6, color="#42a5f5", fill=True, popup="Optimized Start"
        ).add_to(m)
        folium.Marker(
            location=comparison.optimized.polyline[-1],
            icon=folium.Icon(color="blue", icon="flag", prefix="fa"),
            popup="Optimized End"
        ).add_to(m)

    if comparison.local_knowledge and comparison.local_knowledge.polyline:
        folium.PolyLine(
            locations=comparison.local_knowledge.polyline,
            color="#66bb6a",
            weight=5,
            opacity=0.8,
            tooltip="🟢 Local Knowledge Route",
            name="Local Knowledge"
        ).add_to(m)
        
        folium.CircleMarker(
            location=comparison.local_knowledge.polyline[0],
            radius=6, color="#66bb6a", fill=True, popup="Local Knowledge Start"
        ).add_to(m)
        folium.Marker(
            location=comparison.local_knowledge.polyline[-1],
            icon=folium.Icon(color="green", icon="flag", prefix="fa"),
            popup="Local Knowledge End"
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    
    # Fit bounds
    if all_points:
        m.fit_bounds(all_points)

    return m


def fetch_ors_matrix(coords: List[List[float]], api_key: str) -> Optional[List[List[float]]]:
    """Fetch a distance matrix from OpenRouteService using the HGV profile."""
    if not api_key:
        return None
    
    from src.config import ORS_ROUTING_PROFILE
    url = f"https://api.openrouteservice.org/v2/matrix/{ORS_ROUTING_PROFILE}"
    
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': api_key,
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    # ORS expects [lon, lat]
    body = {
        "locations": [[c[1], c[0]] for c in coords],
        "metrics": ["distance"]
    }
    
    try:
        response = requests.post(url, json=body, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return data.get("distances")
        else:
            logger.error(f"ORS Matrix API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Failed to fetch ORS matrix: {str(e)}")
        return None

