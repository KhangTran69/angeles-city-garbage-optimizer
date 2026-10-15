import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.map_utils import (
    _reduce_route_coords,
    build_map_cache_signature,
    get_map_view_cache_path,
    get_missing_map_views,
)


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
