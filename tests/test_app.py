from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
from streamlit.testing.v1 import AppTest

import src.experiment_store as experiment_store
import src.map_utils as map_utils
import src.optimization as optimization_module


def test_under_ceiling_infeasible_run_stays_diagnostic_only(monkeypatch, tmp_path):
    matrix_size = len(optimization_module.BARANGAYS) + 1
    distance_matrix = np.zeros((matrix_size, matrix_size))
    point_index = optimization_module._build_point_index()
    monkeypatch.setattr(
        optimization_module,
        "load_cached_distance_matrix",
        lambda: (distance_matrix, point_index),
    )
    monkeypatch.setattr(experiment_store, "append_experiment_record", lambda *args, **kwargs: None)
    map_html = []
    cache_dir = tmp_path / "map-cache"
    cache_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(map_utils, "_map_cache_dir", lambda: str(cache_dir))
    monkeypatch.setattr(components, "html", lambda value, **kwargs: map_html.append(value))

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.checkbox[1].set_value(False).run()
    app.multiselect[0].set_value(["Balibago"]).run()
    app.button[3].click().run()

    result = app.session_state["results"]
    assert not app.exception
    assert result["params"]["selected"] == ["Balibago"]
    assert result["feasible"] is False
    assert result["feasibility"]["input_reason_codes"] == ["stop_exceeds_vehicle_capacity"]
    assert "Balibago (6470 kg)" in result["feasibility_message"]
    assert any("not a feasible dispatch" in warning.value for warning in app.warning)
    route_cards = [element.value for element in app.markdown if "route-card" in element.value]
    assert route_cards
    assert not any("DYNAMIC DISPATCH" in card for card in route_cards)
    assert not any("DISPATCH:" in card for card in route_cards)

    dispatch_zip = next(
        button for button in app.get("download_button") if button.label == "📦 Dispatch ZIP"
    )
    assert dispatch_zip.disabled is True

    app.session_state["map_route_view"] = 0
    app.run()
    map_markdown = [element.value for element in app.markdown]
    assert any("TACTICAL ROUTE DIAGNOSTIC" in value for value in map_markdown)
    assert any("INFEASIBLE ROUTE CANDIDATE" in value for value in map_markdown)
    assert any("CANDIDATE STOP LIST" in value for value in map_markdown)
    assert not any("OPERATIONAL DISPATCH" in value for value in map_markdown)
    assert map_html
    assert "Vehicle 1 Diagnostic Candidate" in map_html[-1]
    assert "Vehicle 1 Dispatch" not in map_html[-1]

    analysis_text = "\n".join(map_markdown)
    assert "Estimated Savings" not in analysis_text
    assert "Fuel Conservation" not in analysis_text
    assert "Carbon Mitigation" not in analysis_text
    assert "Diagnostic Route Candidates" in analysis_text
    assert "Performance vs Sector-Based Baselines" not in analysis_text
    assert "Candidate Path" in analysis_text
    assert "Optimized Path" not in analysis_text
    assert "Reduction" not in analysis_text


def test_feasible_run_renders_active_dispatch_and_enables_zip(monkeypatch, tmp_path):
    matrix_size = len(optimization_module.BARANGAYS) + 1
    distance_matrix = np.ones((matrix_size, matrix_size))
    np.fill_diagonal(distance_matrix, 0)
    point_index = optimization_module._build_point_index()
    monkeypatch.setattr(
        optimization_module,
        "load_cached_distance_matrix",
        lambda: (distance_matrix, point_index),
    )

    def fail_if_routing_fallback_is_used(*args, **kwargs):
        raise AssertionError("The mocked local matrix should avoid routing fallbacks.")

    monkeypatch.setattr(
        optimization_module,
        "load_road_network",
        fail_if_routing_fallback_is_used,
    )
    monkeypatch.setattr(
        optimization_module,
        "get_distance_matrix",
        fail_if_routing_fallback_is_used,
    )
    monkeypatch.setattr(
        experiment_store,
        "append_experiment_record",
        lambda *args, **kwargs: None,
    )
    map_html = []
    cache_dir = tmp_path / "map-cache-feasible"
    cache_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(map_utils, "_map_cache_dir", lambda: str(cache_dir))
    monkeypatch.setattr(components, "html", lambda value, **kwargs: map_html.append(value))

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.checkbox[1].set_value(False).run()
    app.multiselect[0].set_value(["Capaya"]).run()
    app.button[3].click().run()

    result = app.session_state["results"]
    assert not app.exception
    assert result["feasible"] is True
    assert any("Capaya" in route for route in result["routes"])
    assert any("OPTIMIZED FLEET DISPATCH" in element.value for element in app.markdown)

    route_cards = [element.value for element in app.markdown if "route-card" in element.value]
    assert any(
        "ACTIVE DISPATCH" in card and '<span class="stop-barangay">Capaya</span>' in card
        for card in route_cards
    )

    dispatch_zip = next(
        button for button in app.get("download_button") if button.label == "📦 Dispatch ZIP"
    )
    assert dispatch_zip.disabled is False

    app.session_state["map_route_view"] = 0
    app.run()
    assert any("OPERATIONAL DISPATCH" in element.value for element in app.markdown)
    assert "Vehicle 1 Dispatch" in map_html[-1]

    analysis_text = "\n".join(element.value for element in app.markdown)
    assert "Estimated Savings" in analysis_text
    assert "Fuel Conservation" in analysis_text
    assert "Carbon Mitigation" in analysis_text
    assert "Optimized Path" in analysis_text
    assert "Reduction" in analysis_text


def test_empty_selection_with_five_core_trucks_skips_dynamic_input_validation(monkeypatch):
    matrix_size = len(optimization_module.BARANGAYS) + 1
    distance_matrix = np.zeros((matrix_size, matrix_size))
    point_index = optimization_module._build_point_index()
    monkeypatch.setattr(
        optimization_module,
        "load_cached_distance_matrix",
        lambda: (distance_matrix, point_index),
    )
    monkeypatch.setattr(experiment_store, "append_experiment_record", lambda *args, **kwargs: None)

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.checkbox[1].set_value(False).run()
    app.multiselect[0].set_value([]).run()
    app.slider[0].set_value(5).run()
    app.button[3].click().run()

    assert not app.exception
    assert app.session_state["results"]["feasible"] is True
    assert len(app.session_state["results"]["routes"]) == 5


def test_rejected_small_fleet_clears_previous_dispatch(monkeypatch, tmp_path):
    matrix_size = len(optimization_module.BARANGAYS) + 1
    distance_matrix = np.ones((matrix_size, matrix_size))
    np.fill_diagonal(distance_matrix, 0)
    point_index = optimization_module._build_point_index()
    monkeypatch.setattr(
        optimization_module,
        "load_cached_distance_matrix",
        lambda: (distance_matrix, point_index),
    )

    def fail_if_routing_fallback_is_used(*args, **kwargs):
        raise AssertionError("routing fallback was not expected")

    monkeypatch.setattr(optimization_module, "load_road_network", fail_if_routing_fallback_is_used)
    monkeypatch.setattr(optimization_module, "get_distance_matrix", fail_if_routing_fallback_is_used)
    monkeypatch.setattr(experiment_store, "append_experiment_record", lambda *args, **kwargs: None)
    cache_dir = tmp_path / "map-cache-small-fleet"
    cache_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(map_utils, "_map_cache_dir", lambda: str(cache_dir))
    monkeypatch.setattr(components, "html", lambda *args, **kwargs: None)

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.checkbox[1].set_value(False).run()
    app.multiselect[0].set_value(["Capaya"]).run()
    run_button = next(
        button for button in app.button if button.label == "🚀 EXECUTE OPTIMIZATION"
    )
    run_button.click().run()
    assert app.session_state["results"]["feasible"] is True

    fleet_size = next(
        slider for slider in app.slider if slider.label == "Fleet Size (Vehicles)"
    )
    fleet_size.set_value(5).run()
    run_button = next(
        button for button in app.button if button.label == "🚀 EXECUTE OPTIMIZATION"
    )
    run_button.click().run()

    assert any("need at least 6 vehicles" in item.value for item in app.error)
    assert app.session_state["results"] == {}

    fleet_size = next(
        slider for slider in app.slider if slider.label == "Fleet Size (Vehicles)"
    )
    fleet_size.set_value(6).run()
    assert app.session_state["results"] == {}
    assert not any("OPTIMIZED FLEET DISPATCH" in item.value for item in app.markdown)
