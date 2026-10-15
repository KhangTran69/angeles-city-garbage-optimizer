"""
Angeles City Garbage Route Optimizer - Streamlit UI.
Premium Publication-Grade Dashboard (UI/UX Redesign).
Thesis AL-1: Graph Theory + Genetic Algorithms for CS Thesis.
"""

import streamlit as st
import pandas as pd
import time
import io
import json
import zipfile
import html
import numpy as np
import random
from datetime import datetime
from typing import Dict, Optional, Any, List

# ── Project Core ───────────────────────────────────────────────────────────────
from src.config import *
from src.data import *
from src.experiment_store import *
from src.reporting import *
from src.logger import *
from src.session_state import *
from src.utils import *
from src.research_validation import *
from src.gpx_utils import *
from src.map_utils import *
from src.tsp_feature import integrate_tsp_feature

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title=f"AL-1 | {APP_TITLE}", page_icon="🚛", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
init_session_state()


def render_metric(label: str, value: str, delta: Optional[str] = None, icon: str = "📈"):
    """Helper to render modern glassmorphism metric cards."""
    delta_cls = "positive" if delta and "↓" in str(delta) else ""
    delta_html = (
        f"<div class='metric-delta {delta_cls}'>{html.escape(str(delta))}</div>" if delta else ""
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 10px;">{icon}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-label">{html.escape(str(label))}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_export_section(results):
    """Modernized export section with better button grouping."""
    if not results:
        return
    export_obj = build_export_payload(APP_TITLE, results)
    route_summary_rows = export_obj["route_summaries"]
    st.markdown("<div class='section-title'>📥 DATA EXPORT CONTROLS</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.download_button(
            "💾 Export JSON",
            json.dumps(export_obj, indent=2),
            "ga_results.json",
            "application/json",
            use_container_width=True,
        )

        csv_rows = []
        for i, rt in enumerate(results["routes"]):
            for si, stop in enumerate(rt):
                csv_rows.append({"Vehicle": i + 1, "Stop": si + 1, "Barangay": stop})
        c2.download_button(
            "📊 Export CSV",
            pd.DataFrame(csv_rows).to_csv(index=False),
            "ga_routes.csv",
            "text/csv",
            use_container_width=True,
        )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("summary.json", json.dumps(export_obj, indent=2))
            zf.writestr("route_summary.csv", pd.DataFrame(route_summary_rows).to_csv(index=False))
            for row in route_summary_rows:
                zf.writestr(
                    f"vehicle_{row['vehicle']}_stops.csv",
                    pd.DataFrame(
                        {
                            "vehicle": row["vehicle"],
                            "stop_order": list(range(1, len(row["route"]) + 1)),
                            "barangay": row["route"],
                        }
                    ).to_csv(index=False),
                )
        c3.download_button(
            "📦 Dispatch ZIP",
            zip_buf.getvalue(),
            "dispatch.zip",
            "application/zip",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ── Hero Header ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h1>{APP_TITLE}</h1>
                <p>📍 {APP_SUBTITLE}</p>
                <div style="margin-top: 15px; display: flex; gap: 10px;">
                    <span class="badge badge-green">Thesis AL-1</span>
                    <span class="badge badge-blue">Angeles City, PH 🇵🇭</span>
                </div>
            </div>
            <div style="text-align: right; opacity: 0.8;">
                <div style="font-size: 0.8rem; font-weight: 600; color: #69f0ae;">RESEARCH VALIDATION SYSTEM</div>
                <div style="font-size: 0.7rem; color: #a5d6a7;">VER 2.4.0 (STABLE)</div>
            </div>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)


# ── Global Shared Context (Lazy-Loaded) ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_core_resources(require_graph: bool = False):
    """Load the cached matrix first and only deserialize the road graph when needed."""
    from src.optimization import load_cached_distance_matrix, load_road_network, get_distance_matrix
    
    cached_matrix, cached_point_index = load_cached_distance_matrix()
    if cached_matrix is not None and not require_graph:
        return None, cached_matrix, cached_point_index

    with st.spinner("Synchronizing Road Network Physics..."):
        road_graph = load_road_network()
        dist_matrix, point_to_idx = (cached_matrix, cached_point_index)
        if road_graph and dist_matrix is None:
            dist_matrix, point_to_idx = get_distance_matrix(road_graph)
        return road_graph, dist_matrix, point_to_idx


def normalize_results_shape(results: Dict[str, Any]) -> Dict[str, Any]:
    """Keep session-stored optimization results internally consistent across reruns."""
    if not results:
        return {}

    routes = list(results.get("routes", []))
    raw_dists = list(results.get("dists", []))

    if len(raw_dists) < len(routes):
        raw_dists.extend([0.0] * (len(routes) - len(raw_dists)))
    elif len(raw_dists) > len(routes):
        raw_dists = raw_dists[: len(routes)]

    normalized_dists = [float(distance) for distance in raw_dists]
    results["routes"] = routes
    results["dists"] = normalized_dists
    results["total"] = float(sum(normalized_dists))

    if "waste_dict" in results and "capacity" in results:
        results["route_summaries"] = build_route_summary_rows(
            routes,
            normalized_dists,
            results["waste_dict"],
            results["capacity"],
        )

    return results


def get_results() -> Dict[str, Any]:
    """Return optimization results from session state."""
    if not st.session_state.results:
        return {}

    st.session_state.results = normalize_results_shape(st.session_state.results)
    return st.session_state.results


def build_metrics_from_ga_selection(vehicle_indices: List[int]) -> Optional[RouteMetrics]:
    """Aggregate RouteMetrics from a selection of GA optimization results (fleet or subset)."""
    res = get_results()
    if not res or not vehicle_indices:
        return None

    all_routes = res.get("routes", [])
    all_dists = res.get("dists", [])

    total_dist_km = 0.0
    total_stops = 0
    total_fuel = 0.0
    total_co2 = 0.0
    total_duration = 0.0
    combined_polyline = []

    from src.data import DEPOT_COORD, BARANGAY_COORDS

    for idx in vehicle_indices:
        if idx >= len(all_routes):
            continue

        route = all_routes[idx]
        dist_km = all_dists[idx]

        total_dist_km += dist_km
        total_stops += len(route)

        # Estimate duration per vehicle
        travel_time_mins = (dist_km / (MAP_AVG_SPEED_KMH or 20)) * 60
        service_time_mins = len(route) * (STOP_SERVICE_MINUTES or 5)
        total_duration += travel_time_mins + service_time_mins

        fuel = dist_km * (FUEL_PER_KM or 0.15)
        total_fuel += fuel
        total_co2 += fuel * (CO2_PER_LITER_DIESEL or 2.68)

        # Polyline aggregation
        v_polyline = [DEPOT_COORD]
        for stop in route:
            v_polyline.append(BARANGAY_COORDS.get(stop, DEPOT_COORD))
        v_polyline.append(DEPOT_COORD)
        combined_polyline.extend(v_polyline)

    if not combined_polyline:
        return None

    return RouteMetrics(
        route_type="optimized",
        total_distance_km=total_dist_km,
        total_duration_minutes=total_duration,
        avg_speed_kmh=MAP_AVG_SPEED_KMH or 20,
        num_waypoints=len(combined_polyline),
        num_stops=total_stops,
        estimated_fuel_liters=total_fuel,
        estimated_co2_kg=total_co2,
        start_time=datetime.now(),
        end_time=datetime.now(),
        start_coord=DEPOT_COORD,
        end_coord=DEPOT_COORD,
        polyline=combined_polyline,
    )


with st.sidebar:
    st.markdown(
        "<div class='sidebar-badge'>Thesis Artifact &copy; 2026</div>", unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sidebar-section-header'>🧬 GENETIC ALGORITHM</div>", unsafe_allow_html=True
    )
    n_vehicles = st.slider("Fleet Size (Vehicles)", 1, 25, DEFAULT_N_VEHICLES)
    pop_size = st.select_slider(
        "Population Size", options=[50, 100, 200, 300, 500], value=DEFAULT_POP_SIZE
    )
    generations = st.select_slider(
        "Evolution Cycles", options=[50, 100, 200, 500, 1000], value=DEFAULT_GENERATIONS
    )

    with st.expander("🛠️ Advanced GA Tuning"):
        randomize_seed = st.checkbox("Randomize Seed per Run", value=False)
        active_seed = DEFAULT_RANDOM_SEED if not randomize_seed else random.randint(0, 999999)
        
        random_seed = st.number_input("Research Seed", 0, 999999, active_seed, step=1, disabled=randomize_seed)
        st.info("💡 Fixed seeds ensure reproducible results for thesis reporting.")
        
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.30, DEFAULT_MUTATION_RATE)
        crossover_rate = st.slider("Crossover Rate", 0.50, 1.00, DEFAULT_CROSSOVER_RATE)
        benchmark_runs = st.slider("Benchmark Runs", 3, 15, DEFAULT_BENCHMARK_RUNS)

    st.markdown(
        "<div class='sidebar-section-header'>🚛 FLEET & CAPACITY</div>", unsafe_allow_html=True
    )
    st.info("🚛 **Hybrid Fleet Note**: Truck 1-5 (Core) are fixed. GA settings only affect Truck 6+.")
    vehicle_capacity = st.number_input(
        "Payload Capacity (kg)", 1000, 20000, DEFAULT_VEHICLE_CAPACITY_KG, 500
    )

    with st.expander("📍 BARANGAY SELECTION", expanded=False):
        select_all = st.checkbox("Select Full City", value=True)
        if select_all:
            selected = st.multiselect("Active Zones", BARANGAYS, default=BARANGAYS)
        else:
            selected = st.multiselect("Active Zones", BARANGAYS, default=[])

    rainy = st.toggle("Weather (Rainy +20% Volume)")
    waste_dict = (
        {k: int(v * 1.2) for k, v in BARANGAY_WASTE_KG.items()} if rainy else BARANGAY_WASTE_KG
    )

    st.markdown(
        "<div class='sidebar-section-header'>💰 ECONOMIC PARAMETERS</div>", unsafe_allow_html=True
    )
    with st.expander("Research Cost Constants"):
        current_diesel_price = st.number_input(
            "Diesel Price (₱/L)", 20.0, 200.0, float(DIESEL_PRICE)
        )
        current_fuel_per_km = st.number_input("Fuel Usage (L/km)", 0.05, 1.00, float(FUEL_PER_KM))
        current_working_days = st.number_input(
            "Operational Days/Year", 200, 365, int(COLLECTION_DAYS)
        )
        current_baseline_overhead = st.slider(
            "GSO Baseline Overhead (%)", 10, 100, int(DEFAULT_BASELINE_OVERHEAD)
        )

    st.markdown(
        "<div class='sidebar-section-header'>🌐 INFRASTRUCTURE</div>", unsafe_allow_html=True
    )
    with st.expander("Road Network Physics"):
        if ORS_API_KEY:
            st.markdown(
                "<div style='font-size:0.75rem; color:#69f0ae; margin-bottom:10px;'>🟢 REAL-ROAD ROUTING ACTIVE (ORS)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:0.75rem; color:#ef9a9a; margin-bottom:10px;'>🔴 ORS KEY NOT FOUND (.env)</div>",
                unsafe_allow_html=True,
            )

        current_avg_speed = st.number_input(
            "Traffic Speed (km/h)", 5.0, 60.0, float(MAP_AVG_SPEED_KMH)
        )
        current_stop_service = st.number_input(
            "Load Time (min/stop)", 1.0, 60.0, float(STOP_SERVICE_MINUTES)
        )
        truck_h = st.number_input("Vehicle Height (m)", 1.0, 5.0, 3.5)
        # Weight/Width simplified for this UI version to keep it clean

    st.write("---")
    run_btn = st.button("🚀 EXECUTE OPTIMIZATION", use_container_width=True, type="primary")

# ── Navigation / Tabs ──────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs(
    [
        "🚛 ROUTES",
        "📊 ANALYSIS",
        "🎓 RESEARCH",
        "🗺️ MAP VIEW",
        "📍 LOCATIONS",
        "ℹ️ PROJECT",
        "📈 STABILITY",
        "📍 GPX COMPARISON",
        "🧭 TSP GPX",
    ]
)

# ── Tab 1: Optimized Routes ────────────────────────────────────────────────────
with t1:
    if run_btn:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        
        # 0. Capacity Validation Safeguard (User Requested: Block if > 130%)
        num_dynamic_calc = max(1, n_vehicles - 5)
        total_selected_waste = sum(waste_dict.get(b, 0) for b in selected)
        total_supplementary_capacity = num_dynamic_calc * vehicle_capacity
        
        if total_selected_waste > total_supplementary_capacity * 1.3:
            st.error(f"🚨 **Target Workload Exceeds Fleet Capacity (Limit: 130%)**")
            st.markdown(f"""
            The combined waste for selected active zones (**{total_selected_waste:,} kg**) exceeds the threshold 
            for the supplementary fleet (**{total_supplementary_capacity * 1.3:,.0f} kg**).
            
            **To proceed:**
            1. Increase **Fleet Size** (Trucks 6+) in the sidebar.
            2. Increase **Payload Capacity** (kg) per vehicle.
            3. Deselect some **Active Zones** to reduce the total workload.
            """)
            st.info("💡 Genetic Algorithms require a mathematically feasible volume-to-capacity ratio to converge effectively.")
            st.stop()

        # 1. Prepare CORE GSO FLEET (Fixed)
        core_routes = DETAILED_GSO_ROUTES
        core_dists = [12.16, 10.99, 17.08, 11.82, 13.46]
        core_total = sum(core_dists)
        
        dynamic_routes = []
        dynamic_dists = []
        dynamic_total = 0.0
        history = [core_total]
        run_time = 0.1
        
        # 2. Run DYNAMIC OPTIMIZATION (if barangays selected)
        if selected:
            # Recalibrate: n_vehicles (slider) is the TOTAL target
            # Dynamic fleet = Total target - 5 core trucks
            num_dynamic = max(1, n_vehicles - 5)
            
            road_graph, dist_matrix, pt_to_idx = load_core_resources()
            prog_bar = st.progress(0, text="Initializing supplementary fleet...")
            log_box = st.empty()
            
            with st.spinner("Metaheuristic algorithm evolving supplemental paths..."):
                from src.optimization import run_ga
                start_time = time.time()
                dynamic_routes, dynamic_dists, dynamic_total, history = run_ga(
                    selected,
                    num_dynamic,
                    pop_size,
                    generations,
                    mutation_rate,
                    crossover_rate,
                    prog_bar,
                    log_box,
                    vehicle_capacity,
                    waste_dict,
                    road_graph,
                    dist_matrix,
                    pt_to_idx,
                    int(random_seed),
                )
                end_time = time.time()
                run_time = end_time - start_time
            st.success(f"Dynamic Dispatch Active: {len(selected)} barangays assigned to Trucks 6-{5+len(dynamic_routes)}")
        else:
            st.info("Core Fleet Mode: No supplemental barangays selected. Rendering tactical baseline.")

        # 3. Combine Hybrid Results
        final_routes = core_routes + dynamic_routes
        final_dists = core_dists + dynamic_dists
        total_km = core_total + dynamic_total
        
        route_summaries = build_route_summary_rows(
            final_routes, final_dists, waste_dict, vehicle_capacity
        )
        
        st.session_state.results = {
            "routes": final_routes,
            "dists": final_dists,
            "total": float(total_km),
            "history": history,
            "time": run_time,
            "params": {
                "n_vehicles": len(final_routes), # Compatibility key (Total)
                "n_vehicles_additional": num_dynamic if selected else 0,
                "n_vehicles_total": len(final_routes),
                "generations": generations,
                "pop_size": pop_size,
                "selected": selected,
                "is_stochastic": randomize_seed,
                "random_seed": int(random_seed),
                "benchmark_runs": benchmark_runs,
                "baseline_overhead": current_baseline_overhead,
                "diesel_price": current_diesel_price,
                "fuel_per_km": current_fuel_per_km,
                "working_days": current_working_days,
                "avg_speed": current_avg_speed,
                "stop_service": current_stop_service,
            },
            "waste_dict": waste_dict,
            "capacity": vehicle_capacity,
            "route_summaries": route_summaries,
            "metadata": {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "random_seed": int(random_seed),
                "benchmark_runs": benchmark_runs,
                "data_provenance": DATA_PROVENANCE,
                "ors_enabled": bool(ORS_API_KEY),
            },
        }
        st.session_state.map_route_view = "all"
        add_to_history({"params": st.session_state.results["params"], "total_km": total_km})
        append_experiment_record(
            {
                "timestamp": st.session_state.results["metadata"]["timestamp"],
                "type": "hybrid_run",
                "total_km": round(float(total_km), 2),
                "runtime_seconds": round(run_time, 2),
                "params": st.session_state.results["params"],
                "route_summaries": route_summaries,
            }
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.rerun()

    if st.session_state.results:
        res = st.session_state.results
        st.markdown(
            "<div class='section-title'>🚛 OPTIMIZED FLEET DISPATCH</div>", unsafe_allow_html=True
        )

        # Overview glass card
        cols = st.columns(4)
        with cols[0]:
            render_metric("Total Distance", f"{res['total']:.2f} km", icon="🛣️")
        with cols[1]:
            render_metric("Fleet Size", f"{len(res['routes'])} Trucks", icon="🚛")
        with cols[2]:
            render_metric(
                "Avg Load",
                f"{np.mean([get_route_load(r, res['waste_dict']) for r in res['routes']]):.0f} kg",
                icon="⚖️",
            )
        with cols[3]:
            render_metric("Runtime", f"{res.get('time', 0):.2f}s", icon="⏱️")

        routes_list = res.get("routes", [])
        dists_list = res.get("dists", [])
        
        for i, (rt, dk) in enumerate(zip(routes_list, dists_list)):
            # Ensure rt is handled as a list
            rt_is_active = rt and len(rt) > 0
            
            util_pct = get_load_utilization_pct(rt, res["capacity"], res["waste_dict"])
            util_color = get_utilization_style(util_pct)
            
            # Pull Driver/Sector for the 5 primary trucks
            driver = TRUCK_DRIVERS.get(i + 1, "Staff Driver")
            sector = TRUCK_SECTORS.get(i + 1, "Optimization Area")
            
            # Fleet categorization
            fleet_type = "CORE GSO FLEET" if i < 5 else "DYNAMIC DISPATCH"
            status_tag = "ACTIVE DISPATCH" if rt_is_active else "STANDBY / OFF-DUTY"
            status_color = "#69f0ae" if rt_is_active else "#ef9a9a"

            st.markdown(
                f"""
            <div class='route-card' style='opacity: {1.0 if rt_is_active else 0.6}; border: 1px solid {"rgba(105,240,174,0.3)" if rt_is_active else "rgba(255,255,255,0.1)"};'>
                <div class='route-header'>
                    <div class='route-title'>
                        <span style='color:{VEHICLE_COLORS[i % len(VEHICLE_COLORS)]}; font-size:1.2rem;'>🚛</span>
                        TRUCK {i + 1} <span style='font-size:0.65rem; color:#81c784; margin-left:8px; border:1px solid #81c784; padding:1px 4px; border-radius:3px;'>{fleet_type}</span>
                    </div>
                    <div class='route-dist'>{dk:.2f} km</div>
                </div>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>
                    <div style='font-size:0.75rem; color:#81c784; font-family:"JetBrains Mono"; text-transform:uppercase;'>
                        {f"DISPATCH: {driver}" if i < 5 else f"SECTOR: {sector}"}
                    </div>
                    <div style='font-size:0.6rem; background:{status_color}; color:black; padding:2px 6px; border-radius:4px; font-weight:800;'>{status_tag}</div>
                </div>
                <div class='route-stops'>
                    <span class='stop-depot'>DEPOT</span>
                    <span class='stop-arrow'>→</span>
                    {' <span class="stop-arrow">→</span> '.join([f'<span class="stop-barangay">{s}</span>' for s in rt]) if rt_is_active else '<span style="color:#ef9a9a; font-style:italic;">No active collection assigned</span>'}
                    <span class='stop-arrow'>→</span>
                    <span class='stop-depot'>DEPOT</span>
                </div>
                <div style='margin-top:20px;'>
                    <div style='display:flex; justify-content:space-between; font-size:0.75rem; color:#a5d6a7; margin-bottom:5px; font-family:"JetBrains Mono";'>
                        <span>LOAD UTILIZATION</span>
                        <span>{get_route_load(rt, res["waste_dict"])} / {res["capacity"]} kg ({util_pct}%)</span>
                    </div>
                    <div class='utilization-container'>
                        <div class='utilization-bar' style='width: {util_pct}%; background: {util_color}; shadow: 0 0 10px {util_color}44;'></div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        render_export_section(res)
    else:
        st.info("Welcome to AL-1 Optimization. Please configure parameters and execute the run.")

# ── Tab 2: Analysis ────────────────────────────────────────────────────────────
with t2:
    res = get_results()
    if res:
        p = res["params"]
        total = res["total"]
        baseline_total = round(total * (1 + p["baseline_overhead"] / 100), 2)
        saved_km = round(baseline_total - total, 2)

        st.markdown(
            "<div class='section-title'>💎 ECONOMIC & IMPACT ANALYSIS</div>", unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric(
                "Estimated Savings",
                f"₱{saved_km * p['fuel_per_km'] * p['diesel_price'] * p['working_days']:,.2f}",
                f"↓ {p['baseline_overhead']:.2f}% vs GSO",
                icon="💰",
            )
        with c2:
            render_metric(
                "Fuel Conservation",
                f"{saved_km * p['fuel_per_km'] * p['working_days']:,.2f} L / Year",
                "Energy Efficiency",
                icon="⛽",
            )
        with c3:
            render_metric(
                "Carbon Mitigation",
                f"{saved_km * p['fuel_per_km'] * CO2_PER_LITER_DIESEL * p['working_days']:,.2f} kg CO2",
                "Environmental Impact",
                icon="🌱",
            )

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Performance vs Sector-Based Baselines")
        table_html = "<table class='comparison-table' style='width:100%'><thead><tr><th>Vehicle</th><th>Load (kg)</th><th>Optimized Path</th><th>Reduction</th></tr></thead><tbody>"
        for i, (rt, dk) in enumerate(zip(res["routes"], res["dists"])):
            saved = dk * (p["baseline_overhead"] / 100)
            table_html += f"<tr><td>Unit {i + 1}</td><td>{get_route_load(rt, res['waste_dict']):.2f} kg</td><td><span class='badge badge-green'>{dk:.2f} km</span></td><td><span style='color:#69f0ae'>↓ {saved:.2f} km</span></td></tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if res.get("history") and len(res["history"]) > 1:
            st.markdown(
                "<div class='section-title'>📉 GENETIC ALGORITHM CONVERGENCE</div>", unsafe_allow_html=True
            )
            with st.expander("🔬 How to interpret this chart?", expanded=False):
                st.markdown("""
                    This chart tracks the 'Best Fitness' (Distance) found by the ensemble across generations.
                    - **Steep drops**: Initial rapid convergence.
                    - **Plateaus**: The algorithm has likely found a local optimum. 
                    - **No changes**: If the line is flat from Gen 1, the **Greedy Seed** was already the best option, or you should try a different **Random Seed**.
                """)
            history_df = pd.DataFrame(
                {
                    "Generation": range(1, len(res["history"]) + 1),
                    "Best Distance (km)": res["history"],
                }
            )
            st.line_chart(history_df.set_index("Generation"), color="#69f0ae")

        st.markdown(
            "<div class='section-title'>📊 WORKLOAD DISTRIBUTION</div>", unsafe_allow_html=True
        )
        bal_df = pd.DataFrame(
            {"Vehicle": [f"V{i + 1}" for i in range(len(res["routes"]))], "Distance": res["dists"]}
        )
        st.bar_chart(bal_df.set_index("Vehicle"), color="#1b5e20")

        st.markdown(
            "<div class='section-title'>🚚 ROUTE SUMMARY TABLE</div>", unsafe_allow_html=True
        )
        st.dataframe(
            pd.DataFrame(res["route_summaries"]), hide_index=True, use_container_width=True
        )

        experiment_history = load_experiment_history()
        if experiment_history:
            st.markdown(
                "<div class='section-title'>🧾 EXPERIMENT HISTORY</div>", unsafe_allow_html=True
            )
            history_table = pd.DataFrame(experiment_history[-10:])
            history_table["vehicles"] = history_table["params"].apply(
                lambda value: value.get("n_vehicles_total", value.get("n_vehicles", 0))
            )
            history_table["generations"] = history_table["params"].apply(
                lambda value: value.get("generations", 0)
            )
            history_table["population"] = history_table["params"].apply(
                lambda value: value.get("pop_size", 0)
            )
            history_table["seed"] = history_table["params"].apply(
                lambda value: value.get("random_seed", "N/A")
            )
            st.dataframe(
                history_table[
                    [
                        "timestamp",
                        "type",
                        "total_km",
                        "vehicles",
                        "generations",
                        "population",
                        "seed",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.warning("Execution required for analysis generation.")

# ── Tab 3: Research ────────────────────────────────────────────────────────────
with t3:
    st.markdown(
        "<div class='section-title'>🎓 THESIS VALIDATION FRAMEWORK</div>", unsafe_allow_html=True
    )
    render_methodology_expander()
    render_research_assumptions(int(random_seed), benchmark_runs, has_official_data=False)
    res = get_results()
    if res:
        p = res["params"]
        road_graph, dist_matrix, pt_to_idx = load_core_resources()
        render_research_baseline_comparison(p["selected"], road_graph, dist_matrix, pt_to_idx, res)
        st.divider()
        if st.button("🧬 EXECUTE STATISTICAL T-TEST"):
            road_graph, dist_matrix, pt_to_idx = load_core_resources()
            run_statistical_verification(
                10,
                p["selected"],
                p["n_vehicles"],
                p["pop_size"],
                p["generations"],
                mutation_rate,
                crossover_rate,
                res["capacity"],
                road_graph,
                dist_matrix,
                pt_to_idx,
                res["waste_dict"],
                int(p["random_seed"]),
            )

# ── Tab 4: Tactical Map Analytics ──────────────────────────────────────────────
with t4:
    res = get_results()
    if res:
        road_graph, dist_matrix, pt_to_idx = load_core_resources()
        nv = len(res["routes"])
        view = st.session_state.map_route_view

        # 🛡️ Safety check: Reset view if index is now out of range (e.g. fewer vehicles in new run)
        if view != "all" and (not isinstance(view, int) or view >= nv):
            view = "all"
            st.session_state.map_route_view = "all"
        map_cache_signature = build_map_cache_signature(
            res["routes"],
            res["dists"],
            res["waste_dict"],
            ORS_API_KEY,
        )
        st.session_state.map_cache_signature = map_cache_signature

        st.markdown(
            "<div class='section-title'>🖥️ TACTICAL DISPATCH DASHBOARD</div>", unsafe_allow_html=True
        )

        # 📊 Tier 1: Global Status Bar (Full Width Metrics)
        m_col1, m_col2, m_col3 = st.columns(3)
        if view == "all":
            total_dist = sum(res["dists"])
            total_load = sum([get_route_load(r, res["waste_dict"]) for r in res["routes"]])
            with m_col1:
                render_metric("Fleet Distance", f"{total_dist:.2f} km", icon="🛣️")
            with m_col2:
                render_metric("Total Fleet Load", f"{total_load} kg", icon="⚖️")
            with m_col3:
                render_metric("Active Vehicles", f"{nv}", icon="🚛")
        else:
            rt = res["routes"][view]
            dk = res["dists"][view]
            with m_col1:
                render_metric("Dispatch Path", f"{dk:.2f} km", icon="🛣️")
            with m_col2:
                render_metric(
                    "Current Load", f"{get_route_load(rt, res['waste_dict'])} kg", icon="⚖️"
                )
            with m_col3:
                render_metric("Total Stops", f"{len(rt)}", icon="📍")

        st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)

        # 🗺️ Tier 2: Tactical Command Split
        l_panel, r_map = st.columns([1.1, 2.7])

        with l_panel:
            st.markdown("<div class='glass-card' style='padding:20px;'>", unsafe_allow_html=True)

            # 🛠️ Mission Control Group
            st.markdown(
                "<div style='font-size:0.75rem; font-weight:700; color:#69f0ae; letter-spacing:1px; margin-bottom:15px;'>🎮 MISSION CONTROL</div>",
                unsafe_allow_html=True,
            )

            # Routing Status Indicators
            if ORS_API_KEY:
                st.markdown(
                    "<div style='font-size:0.65rem; color:#69f0ae; margin-bottom:10px;'>🟢 REAL-ROAD ROUTING ACTIVE</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='font-size:0.65rem; color:#ffb74d; margin-bottom:10px;'>🟠 STRAIGHT-LINE FALLBACK (NO API KEY)</div>",
                    unsafe_allow_html=True,
                )

            sim_time = st.slider("Route Progress (Min)", 0, 300, 300, key="sim_timeline")

            # 🚛 Dispatcher Selectors
            st.markdown(
                "<div style='font-size:0.7rem; color:#81c784; margin-top:20px; margin-bottom:10px;'>VEHICLE SELECTOR</div>",
                unsafe_allow_html=True,
            )
            btn_cols = st.columns(3)
            for i in range(nv):
                with btn_cols[i % 3]:
                    if st.button(
                        f"T-{i + 1}",
                        key=f"t_btn_{i}",
                        use_container_width=True,
                        type="primary" if view == i else "secondary",
                    ):
                        st.session_state.map_route_view = i
                        st.rerun()
            if st.button(
                "🌎 FULL FLEET",
                use_container_width=True,
                type="primary" if view == "all" else "secondary",
            ):
                st.session_state.map_route_view = "all"
                st.rerun()

            if view != "all":
                # Detailed Dispatch Card (Driver & Sector)
                driver = res.get("route_summaries", [{}])[view].get("driver", TRUCK_DRIVERS.get(view+1, "Staff Driver"))
                sector = res.get("route_summaries", [{}])[view].get("sector", TRUCK_SECTORS.get(view+1, "Standard Sector"))
                
                st.markdown(
                    f"""
                    <div style='background:rgba(255,255,255,0.05); border-radius:12px; padding:20px; border:1px solid rgba(105,240,174,0.2); margin-top:20px;'>
                        <div style='color:#69f0ae; font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px;'>OPERATIONAL DISPATCH</div>
                        <div style='font-size:1.4rem; font-weight:800; color:white; margin:5px 0;'>TRUCK {view + 1}</div>
                        <div style='display:flex; gap:15px; margin-top:10px;'>
                            <div>
                                <div style='font-size:0.6rem; color:#81c784; text-transform:uppercase;'>Driver</div>
                                <div style='font-size:0.9rem; font-weight:600; color:#69f0ae;'>{driver}</div>
                            </div>
                            <div>
                                <div style='font-size:0.6rem; color:#81c784; text-transform:uppercase;'>Sector</div>
                                <div style='font-size:0.9rem; font-weight:600; color:#69f0ae;'>{sector}</div>
                            </div>
                        </div>
                        {f'''<div style='margin-top:15px; padding-top:15px; border-top:1px solid rgba(105,240,174,0.1);'>
                            <a href='{GOOGLE_MAPS_LINKS[view+1]}' target='_blank' style='text-decoration:none; color:#69f0ae; font-size:0.8rem; display:flex; align-items:center; gap:8px;'>
                                🗺️ View Official Route in Google Maps
                            </a>
                        </div>''' if GOOGLE_MAPS_LINKS.get(view+1) else ""}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 📍 Stop Registry (Scrollable)
            st.markdown(
                "<div style='margin-top:25px; font-size:0.75rem; font-weight:700; color:#69f0ae;'>📍 DISPATCH REGISTRY</div>",
                unsafe_allow_html=True,
            )
            if view != "all":
                st.markdown("<div class='stop-registry-scroll'>", unsafe_allow_html=True)
                sched = compute_stop_schedule(
                    res["routes"][view], res["dists"][view], road_graph, dist_matrix, pt_to_idx, vehicle_idx=view
                )
                for s in sched:
                    st.markdown(
                        f"""
                    <div style='border-left:3px solid {VEHICLE_COLORS[view % len(VEHICLE_COLORS)]}; padding:10px; margin-bottom:10px; background:rgba(105, 240, 174, 0.05); border-radius:0 8px 8px 0;'>
                        <div style='font-size:0.85rem; font-weight:600; color:#e8f5e9;'>{s["name"]}</div>
                        <div style='font-size:0.7rem; color:#69f0ae; font-family:"JetBrains Mono";'>ETA: {s["eta_label"]}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='stop-registry-scroll'>", unsafe_allow_html=True)
                for i in range(nv):
                    st.markdown(
                        f"""
                    <div style='display:flex; align-items:center; gap:10px; margin-bottom:12px;'>
                        <div style='width:12px; height:12px; border-radius:3px; background:{VEHICLE_COLORS[i % len(VEHICLE_COLORS)]};'></div>
                        <div style='font-size:0.9rem;'>Vehicle #{i + 1} ({res["dists"][i]:.2f} km)</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                "<div style='font-size:0.65rem; color:#a5d6a7; margin-top:30px; opacity:0.6;'>OSMnx Network + GA Optimization.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with r_map:
            import streamlit.components.v1 as components

            cache_targets = ["all", *range(nv)]
            missing_map_views = get_missing_map_views(map_cache_signature, cache_targets)

            if missing_map_views:
                with st.spinner("Saving tactical map cache files..."):
                    warm_cached_map_views(
                        routes=res["routes"],
                        dists=res["dists"],
                        view_modes=missing_map_views,
                        colors=VEHICLE_COLORS,
                        road_graph=road_graph,
                        distance_matrix=dist_matrix,
                        point_to_index=pt_to_idx,
                        waste_dict=res["waste_dict"],
                        ors_key=ORS_API_KEY,
                        signature=map_cache_signature,
                    )

            cached_map_html = get_cached_map_view_html(
                res["routes"],
                res["dists"],
                VEHICLE_COLORS,
                st.session_state.map_route_view,
                road_graph,
                dist_matrix,
                pt_to_idx,
                res["waste_dict"],
                ORS_API_KEY,
                map_cache_signature,
            )
            components.html(cached_map_html, height=650, scrolling=False)
    else:
        st.info(
            "Dispatcher standby. Execute optimization in Tab 1 to generate tactical routing maps."
        )

# ── Hidden Tabs (Settings, About, Sensitivity) ────────────────────────────────

with t5:
    st.markdown(
        "<div class='section-title'>📍 ZONE COORDINATE REGISTRY</div>", unsafe_allow_html=True
    )

    # Build dataframe with distance metrics for research validation
    coords_list = []
    for k, v in st.session_state.barangay_coords_editable.items():
        dist_to_depot = haversine(DEPOT_COORD, v)
        coords_list.append(
            {"Barangay": k, "Lat": v[0], "Lon": v[1], "Dist to Depot (km)": dist_to_depot}
        )

    df = pd.DataFrame(coords_list)

    # Use reset trigger as key to force-clear data_editor state on reset
    editor_key = f"coords_editor_{st.session_state.coords_reset_trigger}"
    edited_df = st.data_editor(
        df,
        key=editor_key,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Dist to Depot (km)": st.column_config.NumberColumn(
                format="%.3f km", help="Great-circle distance from the GSO Transfer Station."
            )
        },
    )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("💾 Persist Changes", use_container_width=True):
        updated_coords = {
            row["Barangay"]: (float(row["Lat"]), float(row["Lon"]))
            for _, row in edited_df.iterrows()
        }
        persisted_coords = save_barangay_coords(updated_coords)
        st.session_state.barangay_coords_editable = dict(persisted_coords)
        st.success("Metadata registry persisted to data/barangay_coords.json.")
        st.rerun()

    if c2.button("🔄 Factory Reset", use_container_width=True):
        restored_coords = save_barangay_coords(DEFAULT_BARANGAY_COORDS)
        st.session_state.barangay_coords_editable = dict(restored_coords)
        st.session_state.coords_reset_trigger += 1  # Force widget to recreate
        st.rerun()

    c3.download_button(
        "📥 Backup JSON",
        json.dumps(st.session_state.barangay_coords_editable),
        "coords.json",
        use_container_width=True,
    )

with t6:
    st.markdown(
        "<div class='main-header' style='padding:30px;'><h2>Project AL-1 Thesis Background</h2><p>This system uses a <b>single-objective Genetic Algorithm</b> with <b>capacity-aware route decoding</b> and heuristic local search (2-Opt) to approximate a <b>Capacitated Vehicle Routing Problem (CVRP)</b> solution for Angeles City.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-title'>Optimized Coverage Zones</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for i, b in enumerate(BARANGAYS):
        target = [col1, col2, col3][i % 3]
        target.markdown(f"🔹 {b}")

with t7:
    res = get_results()
    if res:
        road_graph, dist_matrix, pt_to_idx = load_core_resources()
        run_sensitivity_analysis_tab(
            selected,
            n_vehicles,
            pop_size,
            generations,
            mutation_rate,
            crossover_rate,
            vehicle_capacity,
            road_graph,
            dist_matrix,
            pt_to_idx,
            waste_dict,
            int(res["params"]["random_seed"]),
        )
        st.markdown("<div class='section-title'>📏 BENCHMARK MODE</div>", unsafe_allow_html=True)
        if st.button("📐 Run Benchmark Suite", type="primary"):
            from src.optimization import run_benchmark
            benchmark = run_benchmark(
                selected,
                n_vehicles,
                pop_size,
                generations,
                mutation_rate,
                crossover_rate,
                vehicle_capacity,
                waste_dict,
                road_graph,
                dist_matrix,
                pt_to_idx,
                int(res["params"]["random_seed"]),
                benchmark_runs,
            )
            st.session_state.benchmark_results = benchmark
            append_experiment_record(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "type": "benchmark",
                    "total_km": round(float(benchmark["mean_total"]), 2),
                    "runtime_seconds": None,
                    "params": res["params"],
                    "benchmark": {
                        "seeds": benchmark["seeds"],
                        "mean_total": benchmark["mean_total"],
                        "std_total": benchmark["std_total"],
                        "best_total": benchmark["best_total"],
                        "worst_total": benchmark["worst_total"],
                    },
                }
            )
        if st.session_state.get("benchmark_results"):
            benchmark = st.session_state.benchmark_results
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric("Benchmark Mean", f"{benchmark['mean_total']:.2f} km", icon="📊")
            with c2:
                render_metric("Benchmark Std Dev", f"{benchmark['std_total']:.2f} km", icon="📏")
            with c3:
                render_metric("Best Seed", f"{benchmark['best_total']:.2f} km", icon="🏁")
            with c4:
                render_metric("Worst Seed", f"{benchmark['worst_total']:.2f} km", icon="⚠️")
            benchmark_df = pd.DataFrame(
                {"Seed": benchmark["seeds"], "Total Distance (km)": benchmark["totals"]}
            )
            st.dataframe(benchmark_df, hide_index=True, use_container_width=True)
            st.line_chart(benchmark_df.set_index("Seed"))
    else:
        st.info("Stability analysis available after initial run.")

# ── Tab 8: GPX Route Comparison ────────────────────────────────────────────────────
with t8:
    st.markdown(
        """<div class='section-title'>📍 GPX ROUTE TRI-COMPARISON</div>""", unsafe_allow_html=True
    )
    st.markdown(
        """
    <div class='glass-card'>
        <p><b>Scientifically measure real-world driver behavior vs optimized routes.</b></p>
        <ul style='margin: 15px 0 15px 25px;'>
            <li><b style='color: #ef5350;'>Week 1 - Baseline:</b> Geo Tracker recordings of current driver routes</li>
            <li><b style='color: #42a5f5;'>Week 2+ - Optimized:</b> What drivers were supposed to follow</li>
            <li><b style='color: #66bb6a;'>Local Knowledge:</b> Experienced drivers' preferred routes</li>
        </ul>
        <p style='font-size: 0.85rem; color: #a5d6a7;'>📊 Upload GPX files from Geo Tracker, Strava, Garmin, or any GPS tracking app.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize session state for GPX comparison
    if "gpx_comparison" not in st.session_state:
        st.session_state.gpx_comparison = None
    if "gpx_project_name" not in st.session_state:
        st.session_state.gpx_project_name = ""

    # File upload section
    st.markdown("""<div style='margin: 25px 0;'></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        baseline_file = st.file_uploader(
            "🔴 Baseline (Manual Route)",
            type=["gpx"],
            key="baseline_gpx",
            help="Week 1: Driver's actual route via Geo Tracker",
        )

    with c2:
        # 🧪 GA Result Integration (Now Exclusive)
        res = get_results()
        ga_vehicle_indices = []
        use_active_ga = False

        if res:
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
            <div style='background:rgba(66, 165, 245, 0.1); padding:15px; border-radius:10px; border:1px solid rgba(66, 165, 245, 0.3);'>
                <div style='color:#42a5f5; font-size:0.85rem; font-weight:700; margin-bottom:10px;'>🔵 ACTIVE OPTIMIZATION RESULT</div>
                <p style='font-size:0.8rem; margin:0;'>Using the latest mathematical paths from the Genetic Algorithm.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            n_v = len(res.get("routes", []))
            st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

            # Select All Toggle
            select_all = st.checkbox("Select All Fleet Vehicles", value=True)

            if select_all:
                ga_vehicle_indices = list(range(n_v))
                st.info(f"Comparing against full fleet ({n_v} trucks, {res['total']:.2f} km)")
            else:
                ga_vehicle_indices = st.multiselect(
                    "Select Specific Vehicles",
                    range(n_v),
                    default=[0] if n_v > 0 else [],
                    format_func=lambda x: f"Vehicle {x + 1} ({res['dists'][x]:.2f} km)",
                )

            use_active_ga = len(ga_vehicle_indices) > 0
        else:
            st.info("🚀 Run the optimizer in Tab 1 first to load the optimized route here.")

    with c3:
        local_file = st.file_uploader(
            "🟢 Local Knowledge",
            type=["gpx"],
            key="local_gpx",
            help="Experienced driver's preferred route",
        )

    # Compare button
    st.markdown("""<div style='margin: 20px 0;'></div>""", unsafe_allow_html=True)
    compare_col1, compare_col2 = st.columns([2, 1])

    with compare_col1:
        project_name_input = st.text_input(
            "Project Name (Optional)",
            placeholder="e.g., Route Set - April 2026 - Area X",
            key="gpx_project_name_input",
        )

    with compare_col2:
        compare_btn = st.button("🔍 COMPARE ROUTES", type="primary", use_container_width=True)

    if compare_btn:
        if not (baseline_file or use_active_ga or local_file):
            st.warning("Upload at least one GPX file to compare.")
        else:
            with st.spinner("Parsing GPX files and computing metrics..."):
                # Parse uploaded files
                baseline_gpx = parse_gpx_content(baseline_file.read()) if baseline_file else None

                # Optimized is now handled via active GA result helper (multi-select supported)
                if use_active_ga:
                    optimized_gpx = build_metrics_from_ga_selection(ga_vehicle_indices)
                else:
                    optimized_gpx = None

                local_gpx = parse_gpx_content(local_file.read()) if local_file else None

                # Reset file pointers for potential re-read
                if baseline_file:
                    baseline_file.seek(0)
                if local_file:
                    local_file.seek(0)

                # Compare routes
                comparison = compare_routes(baseline_gpx, optimized_gpx, local_gpx)
                st.session_state.gpx_comparison = comparison
                st.session_state.gpx_project_name = (
                    project_name_input or f"Comparison_{datetime.now().strftime('%Y%m%d_%H%M')}"
                )

                log_user_action(
                    logger,
                    "gpx_comparison",
                    {
                        "project_name": st.session_state.gpx_project_name,
                        "has_baseline": baseline_file is not None,
                        "has_optimized": use_active_ga,
                        "has_local": local_file is not None,
                    },
                )

    # Display results
    if st.session_state.gpx_comparison:
        comp = st.session_state.gpx_comparison

        # Efficiency ranking card
        if comp.efficiency_ranking:
            st.markdown("""<div style='margin: 30px 0;'></div>""", unsafe_allow_html=True)
            st.markdown(
                """<div class='section-title'>🏆 EFFICIENCY RANKING</div>""", unsafe_allow_html=True
            )

            ranking_icons = {
                "baseline": "🔴 Manual",
                "optimized": "🔵 Optimized",
                "local_knowledge": "🟢 Local Knowledge",
            }

            ranking_cols = st.columns(len(comp.efficiency_ranking))
            for i, route_type in enumerate(comp.efficiency_ranking):
                with ranking_cols[i]:
                    rank_badge = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    label = ranking_icons.get(route_type, route_type)
                    st.markdown(
                        f"""
                    <div class='metric-card' style='text-align: center;'>
                        <div style='font-size: 2rem;'>{rank_badge}</div>
                        <div style='font-size: 1.1rem; font-weight: 600;'>{label}</div>
                        <div style='font-size: 0.85rem; color: #a5d6a7;'>#{i + 1} Most Efficient</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        # Metrics comparison table
        st.markdown("""<div style='margin: 30px 0;'></div>""", unsafe_allow_html=True)
        st.markdown(
            """<div class='section-title'>📊 SIDE-BY-SIDE METRICS</div>""", unsafe_allow_html=True
        )

        metrics_data = []
        if comp.baseline:
            metrics_data.append(
                {
                    "Route": "🔴 Baseline (Manual)",
                    "Distance": f"{comp.baseline.total_distance_km:.2f} km",
                    "Duration": f"{comp.baseline.total_duration_minutes:.1f} min",
                    "Avg Speed": f"{comp.baseline.avg_speed_kmh:.1f} km/h",
                    "Stops": comp.baseline.num_stops,
                    "Fuel": f"{comp.baseline.estimated_fuel_liters:.2f} L",
                    "CO₂": f"{comp.baseline.estimated_co2_kg:.2f} kg",
                }
            )
        if comp.optimized:
            metrics_data.append(
                {
                    "Route": "🔵 Optimized",
                    "Distance": f"{comp.optimized.total_distance_km:.2f} km",
                    "Duration": f"{comp.optimized.total_duration_minutes:.1f} min",
                    "Avg Speed": f"{comp.optimized.avg_speed_kmh:.1f} km/h",
                    "Stops": comp.optimized.num_stops,
                    "Fuel": f"{comp.optimized.estimated_fuel_liters:.2f} L",
                    "CO₂": f"{comp.optimized.estimated_co2_kg:.2f} kg",
                }
            )
        if comp.local_knowledge:
            metrics_data.append(
                {
                    "Route": "🟢 Local Knowledge",
                    "Distance": f"{comp.local_knowledge.total_distance_km:.2f} km",
                    "Duration": f"{comp.local_knowledge.total_duration_minutes:.1f} min",
                    "Avg Speed": f"{comp.local_knowledge.avg_speed_kmh:.1f} km/h",
                    "Stops": comp.local_knowledge.num_stops,
                    "Fuel": f"{comp.local_knowledge.estimated_fuel_liters:.2f} L",
                    "CO₂": f"{comp.local_knowledge.estimated_co2_kg:.2f} kg",
                }
            )

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        # Savings analysis
        st.markdown("""<div style='margin: 30px 0;'></div>""", unsafe_allow_html=True)
        st.markdown(
            """<div class='section-title'>💰 SAVINGS ANALYSIS</div>""", unsafe_allow_html=True
        )

        savings_cols = st.columns(2)

        with savings_cols[0]:
            if comp.optimized_savings_km is not None:
                savings_color = "#69f0ae" if comp.optimized_savings_km > 0 else "#ef5350"
                st.markdown(
                    f"""
                <div class='glass-card'>
                    <div style='font-size: 0.85rem; color: #a5d6a7; margin-bottom: 10px;'>🔵 OPTIMIZED VS BASELINE</div>
                    <div style='font-size: 1.8rem; font-weight: 700; color: {savings_color};'>
                        {"↓" if comp.optimized_savings_km > 0 else "↑"} {abs(comp.optimized_savings_km):.2f} km
                    </div>
                    <div style='font-size: 1.1rem; color: {savings_color}; margin-top: 5px;'>
                        {comp.optimized_savings_pct:.1f}% {"saved" if comp.optimized_savings_km > 0 else "longer"}
                    </div>
                    <div style='margin-top: 15px; font-size: 0.85rem; color: #81c784;'>
                        Est. Annual Savings: <b>₱{abs(comp.optimized_savings_km) * 0.15 * 100 * 250:,.0f}</b>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Upload both Baseline and Optimized routes to see savings.")

        with savings_cols[1]:
            if comp.local_knowledge_savings_km is not None:
                savings_color = "#69f0ae" if comp.local_knowledge_savings_km > 0 else "#ef5350"
                st.markdown(
                    f"""
                <div class='glass-card'>
                    <div style='font-size: 0.85rem; color: #a5d6a7; margin-bottom: 10px;'>🟢 LOCAL KNOWLEDGE VS BASELINE</div>
                    <div style='font-size: 1.8rem; font-weight: 700; color: {savings_color};'>
                        {"↓" if comp.local_knowledge_savings_km > 0 else "↑"} {abs(comp.local_knowledge_savings_km):.2f} km
                    </div>
                    <div style='font-size: 1.1rem; color: {savings_color}; margin-top: 5px;'>
                        {comp.local_knowledge_savings_pct:.1f}% {"saved" if comp.local_knowledge_savings_km > 0 else "longer"}
                    </div>
                    <div style='margin-top: 15px; font-size: 0.85rem; color: #81c784;'>
                        Est. Annual Savings: <b>₱{abs(comp.local_knowledge_savings_km) * 0.15 * 100 * 250:,.0f}</b>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Upload both Baseline and Local Knowledge routes to see savings.")

        # Deviation analysis
        if comp.deviation_score_optimized is not None or comp.deviation_score_local is not None:
            st.markdown("""<div style='margin: 30px 0;'></div>""", unsafe_allow_html=True)
            st.markdown(
                """<div class='section-title'>📏 ROUTE DEVIATION ANALYSIS</div>""",
                unsafe_allow_html=True,
            )

            dev_cols = st.columns(2)
            with dev_cols[0]:
                if comp.deviation_score_optimized is not None:
                    st.metric(
                        label="🔵 Optimized Route Deviation from Baseline",
                        value=f"{comp.deviation_score_optimized:.0f} m (avg)",
                        help="Average distance optimized route differs from baseline path",
                    )
            with dev_cols[1]:
                if comp.deviation_score_local is not None:
                    st.metric(
                        label="🟢 Local Knowledge Deviation from Baseline",
                        value=f"{comp.deviation_score_local:.0f} m (avg)",
                        help="Average distance local knowledge route differs from baseline",
                    )

        # Save project
        st.markdown("""<div style='margin: 30px 0;'></div>""", unsafe_allow_html=True)
        save_c1, save_c2, save_c3 = st.columns(3)

        with save_c1:
            if st.button("💾 Save Comparison Project", use_container_width=True):
                from pathlib import Path

                projects_dir = Path(__file__).parent / "gpx_projects"
                saved_path = save_comparison_project(
                    comp, st.session_state.gpx_project_name, projects_dir
                )
                st.success(f"Saved to: {saved_path}")

        with save_c2:
            # Export CSV
            if metrics_data:
                csv_data = pd.DataFrame(metrics_data).to_csv(index=False)
                st.download_button(
                    "📊 Export Metrics CSV",
                    csv_data,
                    f"{st.session_state.gpx_project_name}_metrics.csv",
                    "text/csv",
                    use_container_width=True,
                )

        with save_c3:
            # Export JSON
            import json

            json_data = json.dumps(comp.to_dict(), indent=2)
            st.download_button(
                "💾 Export Full JSON",
                json_data,
                f"{st.session_state.gpx_project_name}_full.json",
                "application/json",
                use_container_width=True,
            )

        # Map visualization
        st.markdown("""<div style='margin: 30px 0;'></div>""", unsafe_allow_html=True)
        st.markdown(
            """<div class='section-title'>🗺️ ROUTE OVERLAY MAP</div>""", unsafe_allow_html=True
        )
        
        from streamlit_folium import st_folium
        comparison_map = build_gpx_comparison_map(comp)
        st_folium(comparison_map, use_container_width=True, height=600)

        # Google Maps Verification Section
        st.markdown("""<div style='margin: 30px 0;'></div>""", unsafe_allow_html=True)
        st.markdown(
            """<div class='section-title'>🔍 GOOGLE MAPS API VERIFICATION</div>""", unsafe_allow_html=True
        )
        
        from src.config import ORS_API_KEY
        google_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
        
        if not google_api_key or "your_google_maps" in google_api_key:
            st.warning("⚠️ Google Maps API Key not found in .env. Please add `GOOGLE_MAPS_API_KEY` to enable verification.")
            st.info("This feature uses Google's proprietary routing engine to verify our optimized routes for truck feasibility and distance accuracy.")
        else:
            if comp.optimized:
                st.write("Verifying the **Optimized (GA)** route against Google Maps Directions API...")
                if st.button("🚀 Run Google Maps Verification"):
                    from src.google_utils import verify_optimized_route
                    with st.spinner("Calling Google Maps API..."):
                        # Get coordinates from optimized polyline
                        v_report = verify_optimized_route(
                            "Optimized GA Route",
                            comp.optimized.polyline,
                            comp.optimized.total_distance_km,
                            google_api_key
                        )
                    
                    if v_report["success"]:
                        st.success("✅ Google Maps verification complete!")
                        
                        v_col1, v_col2, v_col3 = st.columns(3)
                        v_col1.metric("GA Distance", f"{v_report['current_distance_km']:.2f} km")
                        v_col2.metric("Google Distance", f"{v_report['google_distance_km']:.2f} km")
                        v_col3.metric("Delta", f"{v_report['delta_km']:+.2f} km", delta=f"{v_report['delta_pct']:+.1f}%")
                        
                        st.markdown(f"""
                        <div class='glass-card'>
                            <b>Verification Summary:</b><br>
                            The Google Maps Directions API calculated a distance of <b>{v_report['google_distance_km']:.2f} km</b> 
                            for the optimized sequence. This is a <b>{abs(v_report['delta_pct']):.1f}%</b> 
                            {'difference' if abs(v_report['delta_pct']) > 0.1 else 'match'} compared to our internal OSMnx results.
                            <br><br>
                            <b>Google Estimated Duration:</b> {v_report['google_duration_mins']:.1f} mins
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"Google Maps Verification Failed: {v_report['error']}")



    else:
        st.markdown("""<div style='margin: 30px 0;'></div>""", unsafe_allow_html=True)
        st.info("👆 Upload GPX files and click 'Compare Routes' to see analysis.")

with t9:
    integrate_tsp_feature()

st.markdown(
    f"""
    <div class='footer-note'>
        Project Angeles City AL-1 &copy; 2026 &bull; Refined Publication-Grade Analytics Dashboard &bull; Philippines 🇵🇭
    </div>
""",
    unsafe_allow_html=True,
)
