import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.map_utils import (
    _reduce_route_coords,
    build_folium_route_map,
    build_map_cache_signature,
    fetch_ors_route,
    get_map_view_cache_path,
    get_missing_map_views,
)


def test_fetch_ors_route_uses_requested_truck_height(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"features": [{"geometry": {"coordinates": [[120.6, 15.1]]}}]}

    def post(url, *, json, headers, timeout):
        captured["body"] = json
        return Response()

    monkeypatch.setattr("src.map_utils.requests.post", post)

    route = fetch_ors_route([[15.1, 120.6], [15.2, 120.7]], "test-key", 4.25)

    assert route == [[15.1, 120.6]]
    assert captured["body"]["options"]["profile_params"]["restrictions"]["height"] == 4.25


def test_full_fleet_map_marks_diagnostic_candidates_and_feasible_trucks():
    routes = [["Amsic"], ["Balibago"]]
    distances = [1.0, 2.0]
    waste_dict = {"Amsic": 100, "Balibago": 200}

    diagnostic_html = build_folium_route_map(
        routes, distances, view_mode="all", waste_dict=waste_dict, feasible=False
    ).get_root().render()
    feasible_html = build_folium_route_map(
        routes, distances, view_mode="all", waste_dict=waste_dict, feasible=True
    ).get_root().render()

    assert "C-1" in diagnostic_html
    assert "Vehicle 1 Diagnostic Candidate" in diagnostic_html
    assert "Vehicle 1 Dispatch" not in diagnostic_html
    assert "T-1" in feasible_html
    assert "Vehicle 1 Dispatch" in feasible_html


def test_build_map_cache_signature_is_stable_for_same_payload():
    routes = [["Amsic", "Anunas"], ["Balibago"]]
    dists = [12.34, 5.67]
    waste_dict = {"Amsic": 1200, "Anunas": 950, "Balibago": 800}

    first = build_map_cache_signature(routes, dists, waste_dict, "")
    second = build_map_cache_signature(routes, dists, waste_dict, "")

    assert first == second


def test_build_map_cache_signature_changes_when_routes_change():
    waste_dict = {"Amsic": 1200, "Anunas": 950}

    first = build_map_cache_signature([["Amsic"]], [12.34], waste_dict, "")
    second = build_map_cache_signature([["Anunas"]], [12.34], waste_dict, "")

    assert first != second


def test_build_map_cache_signature_changes_when_feasibility_changes():
    routes = [["Amsic"]]
    dists = [12.34]
    waste_dict = {"Amsic": 1200}

    feasible_signature = build_map_cache_signature(
        routes, dists, waste_dict, "", feasible=True
    )
    diagnostic_signature = build_map_cache_signature(
        routes, dists, waste_dict, "", feasible=False
    )

    assert feasible_signature != diagnostic_signature


def test_build_map_cache_signature_includes_feasibility_and_truck_height():
    routes = [["Amsic"]]
    dists = [12.34]
    waste_dict = {"Amsic": 1200}

    feasible_signature = build_map_cache_signature(
        routes, dists, waste_dict, "", truck_height_m=3.5, feasible=True
    )
    diagnostic_signature = build_map_cache_signature(
        routes, dists, waste_dict, "", truck_height_m=3.5, feasible=False
    )
    taller_truck_signature = build_map_cache_signature(
        routes, dists, waste_dict, "", truck_height_m=4.0, feasible=True
    )

    assert feasible_signature != diagnostic_signature
    assert feasible_signature != taller_truck_signature


def test_get_map_view_cache_path_normalizes_all_and_numeric_views(monkeypatch, tmp_path):
    monkeypatch.setattr("src.map_utils._map_cache_dir", lambda: str(tmp_path))

    assert get_map_view_cache_path("sig123", "all") == str(tmp_path / "sig123_all.html")
    assert get_map_view_cache_path("sig123", 2) == str(tmp_path / "sig123_2.html")


def test_get_missing_map_views_only_returns_missing_entries(monkeypatch, tmp_path):
    monkeypatch.setattr("src.map_utils._map_cache_dir", lambda: str(tmp_path))
    existing = tmp_path / "sig123_all.html"
    existing.write_text("<html></html>", encoding="utf-8")

    missing = get_missing_map_views("sig123", ["all", 0, 1])

    assert missing == [0, 1]


def test_reduce_route_coords_preserves_route_endpoints():
    coords = [[float(idx), float(idx)] for idx in range(200)]

    reduced = _reduce_route_coords(coords, max_points=25)

    assert len(reduced) == 25
    assert reduced[0] == coords[0]
    assert reduced[-1] == coords[-1]
