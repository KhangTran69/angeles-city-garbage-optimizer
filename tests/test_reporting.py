from src.reporting import build_export_payload


def test_build_export_payload_contains_route_summaries():
    """Export payload should include route summaries and metadata."""
    results = {
        "routes": [["Amsic", "Anunas"]],
        "dists": [10.0],
        "total": 10.0,
        "history": [12.0, 10.0],
        "time": 1.23,
        "params": {"random_seed": 42},
        "metadata": {"timestamp": "2026-04-02T12:00:00"},
        "waste_dict": {"Amsic": 1000, "Anunas": 900},
        "capacity": 5000,
    }

    payload = build_export_payload("Test", results)

    assert payload["title"] == "Test"
    assert payload["route_summaries"][0]["vehicle"] == 1
    assert payload["route_summaries"][0]["load_kg"] == 1900
