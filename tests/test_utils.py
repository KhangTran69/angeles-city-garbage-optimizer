from src.config import MAP_AVG_SPEED_KMH, STOP_SERVICE_MINUTES
from src.utils import haversine, get_route_load, estimate_route_duration_minutes, route_distance_road

def test_haversine_same_point():
    """Distance to self should be 0."""
    point = (15.1450, 120.5887)
    assert haversine(point, point) == 0.0

def test_get_route_load():
    """Verify total load calculation."""
    route = ["Agapito del Rosario", "Amsic"]
    # 850 + 1250 = 2100
    assert get_route_load(route) == 2100

def test_estimate_duration():
    """Verify duration formula."""
    # Uses the configured app constants so the test stays aligned with the product.
    dist = 22.0
    stops = 2
    expected = (dist / MAP_AVG_SPEED_KMH) * 60 + (stops * STOP_SERVICE_MINUTES)
    assert estimate_route_duration_minutes(dist, stops) == expected

def test_haversine_known_dist():
    """Approx distance between Agapito and Amsic is ~2.3km."""
    d = haversine((15.1433, 120.5881), (15.1556, 120.5709))
    assert 2.0 < d < 2.5


def test_route_distance_road_falls_back_without_graph():
    """Distance calculation should still work when no road graph is available."""
    distance = route_distance_road(["Amsic", "Anunas"], None, None, None)

    assert distance > 0
