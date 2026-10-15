"""
Streamlit session state management for the Angeles City Garbage Route Optimizer.
Initializes default values and handles performance history.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, List


def init_session_state():
    """Ensure all required session state keys are initialized."""
    if "results" not in st.session_state:
        st.session_state.results = {}

    if "ga_params_history" not in st.session_state:
        st.session_state.ga_params_history = []

    if "map_route_view" not in st.session_state:
        st.session_state.map_route_view = "all"

    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = {}

    if "map_cache_signature" not in st.session_state:
        st.session_state.map_cache_signature = ""

    if "map_cache_warmed" not in st.session_state:
        st.session_state.map_cache_warmed = False

    if "barangay_coords_editable" not in st.session_state:
        from src.data import BARANGAY_COORDS

        st.session_state.barangay_coords_editable = dict(BARANGAY_COORDS)

    if "coords_reset_trigger" not in st.session_state:
        st.session_state.coords_reset_trigger = 0


def add_to_history(entry: Dict[str, Any]):
    """Add a new GA run to the recent history list (capped at 10)."""
    from src.config import MAX_HISTORY_SIZE

    history = st.session_state.ga_params_history
    history.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **entry})
    # Keep only the last N runs
    if len(history) > MAX_HISTORY_SIZE:
        st.session_state.ga_params_history = history[-MAX_HISTORY_SIZE:]
    else:
        st.session_state.ga_params_history = history
