"""Unit tests for the GPX mission-point TSP feature."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tsp_feature import (
    _build_bbox_graph_key,
    _coords_within_bounds,
    build_distance_matrix_meters,
    extract_gpx_point_records,
    parse_gpx_upload_data,
    simplify_transit_path,
    solve_closed_tsp,
)
from src.tsp_feature import (
    build_waypoint_distance_matrix_meters,
    calculate_order_distance_km,
    parse_gpx_spatial_upload_data,
    ramer_douglas_peucker,
    solve_waypoint_tsp,
)



SAMPLE_TSP_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Test">
  <trk>
    <name>Truck Route</name>
    <trkseg>
      <trkpt lat="15.1450" lon="120.5880"><time>2026-04-13T08:00:00Z</time></trkpt>
      <trkpt lat="15.1458" lon="120.5888"><time>2026-04-13T08:02:00Z</time></trkpt>
      <trkpt lat="15.1470" lon="120.5900"><time>2026-04-13T08:04:00Z</time></trkpt>
      <trkpt lat="15.14701" lon="120.59001"><time>2026-04-13T08:05:00Z</time></trkpt>
      <trkpt lat="15.14700" lon="120.59002"><time>2026-04-13T08:06:00Z</time></trkpt>
      <trkpt lat="15.14702" lon="120.59001"><time>2026-04-13T08:07:00Z</time></trkpt>
      <trkpt lat="15.14701" lon="120.59000"><time>2026-04-13T08:08:00Z</time></trkpt>
      <trkpt lat="15.14700" lon="120.59001"><time>2026-04-13T08:09:00Z</time></trkpt>
      <trkpt lat="15.14701" lon="120.59002"><time>2026-04-13T08:10:00Z</time></trkpt>
      <trkpt lat="15.1490" lon="120.5920"><time>2026-04-13T08:13:00Z</time></trkpt>
      <trkpt lat="15.14901" lon="120.59201"><time>2026-04-13T08:14:00Z</time></trkpt>
      <trkpt lat="15.14900" lon="120.59200"><time>2026-04-13T08:15:00Z</time></trkpt>
      <trkpt lat="15.14902" lon="120.59201"><time>2026-04-13T08:16:00Z</time></trkpt>
      <trkpt lat="15.14901" lon="120.59202"><time>2026-04-13T08:17:00Z</time></trkpt>
      <trkpt lat="15.14900" lon="120.59201"><time>2026-04-13T08:18:00Z</time></trkpt>
      <trkpt lat="15.14902" lon="120.59200"><time>2026-04-13T08:19:00Z</time></trkpt>
      <trkpt lat="15.1450" lon="120.5880"><time>2026-04-13T08:22:00Z</time></trkpt>
    </trkseg>
  </trk>
</gpx>
"""

TWO_POINT_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Test">
  <trk>
    <trkseg>
      <trkpt lat="15.1450" lon="120.5880"><time>2026-04-13T08:00:00Z</time></trkpt>
      <trkpt lat="15.1460" lon="120.5890"><time>2026-04-13T08:05:00Z</time></trkpt>
    </trkseg>
  </trk>
</gpx>
"""

ROUTE_POINT_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Test">
  <rte>
    <name>Route Export</name>
    <rtept lat="15.1450" lon="120.5880" />
    <rtept lat="15.1460" lon="120.5890" />
    <rtept lat="15.1470" lon="120.5900" />
    <rtept lat="15.1480" lon="120.5910" />
  </rte>
</gpx>
"""


def test_parse_gpx_upload_detects_two_mission_points():
    """Stationary clusters longer than 5 minutes should become TSP mission points."""
    parsed = parse_gpx_upload_data(SAMPLE_TSP_GPX.encode("utf-8"), "truck.gpx")

    assert len(parsed["original_path"]) == 17
    assert parsed["start_coord"] == (15.1450, 120.5880)
    assert parsed["original_distance_km"] > 0
    assert len(parsed["mission_points"]) == 2
    assert parsed["mission_points"][0]["dwell_minutes"] >= 5.0
    assert parsed["mission_points"][1]["dwell_minutes"] >= 5.0


def test_parse_gpx_upload_rejects_routes_with_fewer_than_three_points():
    """Routes shorter than 3 samples should fail fast with a clear error."""
    with pytest.raises(ValueError, match="at least 3 GPX points"):
        parse_gpx_upload_data(TWO_POINT_GPX.encode("utf-8"), "short.gpx")


def test_parse_gpx_upload_accepts_route_points():
    """Route-point GPX files should parse instead of being reported as empty."""
    parsed = parse_gpx_upload_data(ROUTE_POINT_GPX.encode("utf-8"), "route.gpx")

    assert len(parsed["original_path"]) == 4
    assert parsed["point_source"] == "route"
    assert parsed["timestamped_samples"] == 0


def test_extract_gpx_point_records_falls_back_to_routes():
    """Extraction should use route points when no tracks are present."""
    import gpxpy

    gpx = gpxpy.parse(ROUTE_POINT_GPX)
    point_records, point_source = extract_gpx_point_records(gpx)

    assert point_source == "route"
    assert len(point_records) == 4


def test_bbox_graph_key_encodes_uploaded_extent():
    """Dynamic OSM graph keys should reflect the uploaded route extent."""
    coords = [(15.1450, 120.5880), (15.1490, 120.5920)]

    graph_key = _build_bbox_graph_key(coords)

    assert graph_key.startswith("bbox:")


def test_coords_within_bounds_detects_out_of_area_routes():
    """Bounds helper should reject coordinates outside the loaded graph extent."""
    angeles_bounds = (15.10, 120.55, 15.20, 120.65)

    assert _coords_within_bounds([(15.1450, 120.5880)], angeles_bounds)
    assert not _coords_within_bounds([(42.4600, -71.0900)], angeles_bounds)


def test_simplify_transit_path_reduces_dense_collinear_points():
    """Douglas-Peucker simplification should reduce redundant transit samples."""
    point_records = [
        {"coord": (15.1450 + index * 0.0001, 120.5880 + index * 0.0001), "timestamp": float(index)}
        for index in range(12)
    ]

    simplified = simplify_transit_path(point_records, [], tolerance_meters=15.0)

    assert len(simplified) < len(point_records)


def test_closed_tsp_tour_starts_and_ends_at_depot():
    """The TSP solver must return a closed tour covering each node once."""
    node_coords = (
        (15.1450, 120.5880),
        (15.1470, 120.5900),
        (15.1490, 120.5920),
        (15.1460, 120.5930),
    )

    matrix = build_distance_matrix_meters(node_coords)
    tour = solve_closed_tsp(matrix)

    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)
    assert tour[0] == 0
    assert tour[-1] == 0
    assert set(tour[:-1]) == {0, 1, 2, 3}

WAYPOINT_TSP_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.0" creator="Test">
  <wpt lat="15.1450" lon="120.5880"><name>1st</name></wpt>
  <wpt lat="15.1470" lon="120.5900"><name>2</name></wpt>
  <wpt lat="15.1490" lon="120.5920"><name>3</name></wpt>
  <trk>
    <trkseg>
      <trkpt lat="15.1450" lon="120.5880" />
      <trkpt lat="15.1460" lon="120.5890" />
      <trkpt lat="15.1470" lon="120.5900" />
      <trkpt lat="15.1480" lon="120.5910" />
      <trkpt lat="15.1490" lon="120.5920" />
    </trkseg>
  </trk>
</gpx>
"""


def test_parse_spatial_gpx_extracts_tracks_and_named_waypoints():
    """The upload parser should keep dense track geometry and named placemarks."""
    parsed = parse_gpx_spatial_upload_data(WAYPOINT_TSP_GPX.encode("utf-8"), "waypoints.gpx")

    assert len(parsed["original_path"]) == 5
    assert len(parsed["waypoints"]) == 3
    assert parsed["waypoints"][0]["label"] == "1st"
    assert parsed["point_source"] == "track"


def test_rdp_simplifier_reduces_collinear_track_points():
    """Epsilon in coordinate degrees should remove redundant track points."""
    points = tuple((15.1450 + idx * 0.0001, 120.5880 + idx * 0.0001) for idx in range(20))

    simplified = ramer_douglas_peucker(points, epsilon=0.00001)

    assert simplified == [points[0], points[-1]]


def test_waypoint_tsp_covers_each_placemark_once():
    """The waypoint TSP solver should return one visit per waypoint for an open path."""
    coords = (
        (15.1450, 120.5880),
        (15.1470, 120.5900),
        (15.1490, 120.5920),
    )

    matrix = build_waypoint_distance_matrix_meters(coords)
    order = solve_waypoint_tsp(tuple(tuple(row) for row in matrix), return_to_start=False, time_limit_seconds=1)

    assert order[0] == 0
    assert len(order) == 3
    assert set(order) == {0, 1, 2}
    assert calculate_order_distance_km(matrix, order) > 0

