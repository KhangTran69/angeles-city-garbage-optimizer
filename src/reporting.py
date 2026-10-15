"""Helpers for export payloads and route-level reporting."""
from typing import Any, Dict, List

from src.utils import (
    estimate_route_duration_minutes,
    format_eta_minutes,
    get_load_utilization_pct,
    get_route_load,
)


def build_route_summary_rows(
    routes: List[List[str]],
    dists: List[float],
    waste_dict: Dict[str, int],
    capacity: int,
) -> List[Dict[str, Any]]:
    """Create route-level summary rows for tables and exports."""
    rows: List[Dict[str, Any]] = []
    for index, (route, distance_km) in enumerate(zip(routes, dists), start=1):
        load_kg = get_route_load(route, waste_dict)
        eta_minutes = estimate_route_duration_minutes(distance_km, len(route))
        rows.append(
            {
                "vehicle": index,
                "stops": len(route),
                "distance_km": round(float(distance_km), 2),
                "load_kg": load_kg,
                "capacity_kg": capacity,
                "utilization_pct": get_load_utilization_pct(route, capacity, waste_dict),
                "estimated_minutes": round(float(eta_minutes), 2),
                "estimated_duration_label": format_eta_minutes(eta_minutes),
                "route": route,
            }
        )
    return rows


def build_export_payload(title: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a richer export payload from optimization results."""
    route_summaries = build_route_summary_rows(
        results["routes"],
        results["dists"],
        results["waste_dict"],
        results["capacity"],
    )
    total_load = sum(row["load_kg"] for row in route_summaries)

    return {
        "title": title,
        "params": results.get("params", {}),
        "metadata": results.get("metadata", {}),
        "total_km": round(float(results["total"]), 2),
        "runtime_seconds": round(float(results.get("time", 0.0)), 2),
        "total_load_kg": total_load,
        "routes": results["routes"],
        "route_summaries": route_summaries,
        "history": results.get("history", []),
    }
