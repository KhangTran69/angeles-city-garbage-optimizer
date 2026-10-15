"""
Research validation and methodology logic for Angeles City Garbage Route Optimizer.
Includes T-test, sensitivity analysis, and methodology documentation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import html
from scipy import stats
from typing import List, Dict, Optional, Any
from src.optimization import (
    build_seed_chromosomes,
    run_benchmark,
    run_ga,
    evaluate_manual_baseline,
    greedy_baseline,
)
from src.utils import get_route_load

def run_sensitivity_analysis_tab(
    selected_barangays: List[str], 
    n_vehicles: int, 
    pop_size: int, 
    generations: int, 
    mutation_rate: float, 
    crossover_rate: float,
    vehicle_capacity_kg: int,
    road_graph, 
    distance_matrix: np.ndarray, 
    point_to_index: Dict[str, int],
    waste_dict: Optional[Dict[str, int]] = None,
    base_seed: int = 0,
):
    """Run sensitivity analysis (N repetitions of the GA)."""
    st.markdown("<div class='section-title'>Sensitivity Analysis (5 Runs)</div>", unsafe_allow_html=True)
    if st.button("🔄 Run Analysis (5x)", type="primary", key="sensitivity_btn"):
        st.info("Running a reproducible 5-seed benchmark to check stability...")
        benchmark = run_benchmark(
            selected_barangays,
            n_vehicles,
            pop_size,
            generations,
            mutation_rate,
            crossover_rate,
            vehicle_capacity_kg,
            waste_dict,
            road_graph,
            distance_matrix,
            point_to_index,
            base_seed,
            5,
        )
        sa_results = benchmark["totals"]
        
        st.success("Analysis Complete!")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Mean Total Distance", f"{benchmark['mean_total']:.2f} km")
        with c2:
            st.metric("Standard Deviation", f"± {benchmark['std_total']:.2f} km")
            
        sa_df = pd.DataFrame({
            "Seed": benchmark["seeds"],
            "Total Distance (km)": sa_results
        })
        st.line_chart(sa_df.set_index("Seed"))


def render_methodology_expander():
    """Display the research data generation Methodology expander."""
    with st.expander("📊 Data Generation Methodology & Sources", expanded=True):
        st.markdown("""
        **Waste Volume Estimation Model**
        
        The daily waste estimates per barangay are derived using the standard methodology endorsed by the
        **Philippine Department of Environment and Natural Resources (DENR)** and the **World Bank's
        What a Waste 2.0** global report:
        
        > **Formula:** `Daily Waste (kg) = Barangay Population × Per-Capita Waste Rate (0.40 kg/person/day)`
        
        The per-capita rate of **0.40 kg/person/day** reflects a research assumption aligned with
        DENR-reported averages for secondary cities. Population figures should be replaced or verified
        against official **2020 Philippine Statistics Authority (PSA) Census of Population and Housing**
        data for any formal deployment study.
        
        **Note:** The bundled barangay waste values in this prototype are still illustrative unless the
        underlying population and collection records are revalidated for the target study period.

        **Citation:** DENR Ecological Solid Waste Management Act (RA 9003) Implementation Guidelines;
        World Bank (2018). *What a Waste 2.0: A Global Snapshot of Solid Waste Management to 2050.*
        """)


def render_research_assumptions(
    random_seed: int,
    benchmark_runs: int,
    has_official_data: bool = False,
):
    """Render explicit assumptions and reproducibility notes for thesis reporting."""
    official_data_note = "Validated dataset loaded." if has_official_data else "Prototype dataset still includes illustrative waste and baseline values."
    with st.expander("🧪 Assumptions, Reproducibility, and Limitations", expanded=True):
        st.markdown(
            f"""
            - **Random seed:** `{random_seed}` is used as the default reproducibility seed for single runs.
            - **Benchmark configuration:** stability analysis uses `{benchmark_runs}` sequential seeds starting from the base seed.
            - **Travel-time model:** ETAs use configured average speed and per-stop service time assumptions, not live traffic data.
            - **Baseline methodology:** Manual GSO and greedy baselines are compared on the same distance model and vehicle-capacity rules.
            - **Data provenance:** {official_data_note}
            - **Interpretation:** Results should be treated as decision-support estimates, not field-verified dispatch instructions.
            """
        )

def render_research_baseline_comparison(
    selected: List[str], 
    road_graph, 
    distance_matrix: np.ndarray, 
    point_to_index: Dict[str, int],
    ga_results: Dict[str, Any]
):
    """Render the 3-way comparison metrics (Manual GSO vs Greedy vs GA)."""
    st.markdown("<div class='section-title'>🛣️ Three-Way Baseline Comparison</div>", unsafe_allow_html=True)
    
    ga_total = ga_results["total"]
    waste_dict = ga_results["waste_dict"]
    vehicle_capacity = ga_results["capacity"]
    params = ga_results.get("params", {})
    n_vehicles = params.get("n_vehicles", 5)
    total_demand = sum(waste_dict.get(barangay, 0) for barangay in selected)
    fleet_capacity = vehicle_capacity * n_vehicles
    
    with st.spinner("Computing reference baselines..."):
        gso_total, gso_routes = evaluate_manual_baseline(
            selected,
            road_graph,
            distance_matrix,
            point_to_index,
            vehicle_capacity,
            waste_dict,
        )
        greedy_total, greedy_routes = greedy_baseline(
            selected,
            n_vehicles,
            waste_dict,
            road_graph,
            distance_matrix,
            point_to_index,
            vehicle_capacity,
        )
    
    ga_vs_greedy = round(((greedy_total - ga_total) / greedy_total) * 100, 1) if greedy_total > 0 else 0
    ga_vs_gso = round(((gso_total - ga_total) / gso_total) * 100, 1) if gso_total > 0 else 0

    if total_demand > fleet_capacity:
        st.warning(
            f"⚠️ **Comparison caution:** total demand is **{total_demand:,} kg**, but fleet capacity is only "
            f"**{fleet_capacity:,} kg** ({n_vehicles} vehicles × {vehicle_capacity:,} kg). "
            "This scenario is infeasible for a single dispatch wave, so the distance comparison is not a strict apples-to-apples vehicle comparison."
        )
    else:
        st.info(
            f"🔬 **Research Note:** All baselines in this view use the active capacity setting "
            f"(**{vehicle_capacity:,} kg per vehicle**) and the same distance model."
        )

    cols = st.columns(3)
    c_labels = [
        ("🛣️ Manual GSO", f"{gso_total:.2f} km", "Sector Baseline", "#ef9a9a"),
        ("🤖 Greedy Algorithm", f"{greedy_total:.2f} km", "Nearest Neighbor", "#ffb74d"),
        ("🧬 GA Optimized", f"{ga_total:.2f} km", f"↓ {ga_vs_gso:.2f}% Efficiency", "#69f0ae")
    ]
    
    for col, (lbl, val, delta, color) in zip(cols, c_labels):
        with col:
            st.markdown(
                f'''
                <div class="metric-card">
                    <div class="metric-value" style="color:{color}">{html.escape(val)}</div>
                    <div class="metric-label">{html.escape(lbl)}</div>
                    <div class="metric-delta">{"⚡ " if "↓" in delta else "⚪ "}{html.escape(delta)}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

    comparison_rows = [
        {
            "Method": "Manual GSO",
            "Distance (km)": round(gso_total, 2),
            "Trips/Routes": len(gso_routes),
            "Max Route Load (kg)": max((get_route_load(route, waste_dict) for route in gso_routes), default=0),
        },
        {
            "Method": "Greedy",
            "Distance (km)": round(greedy_total, 2),
            "Trips/Routes": len(greedy_routes),
            "Max Route Load (kg)": max((get_route_load(route, waste_dict) for route in greedy_routes), default=0),
        },
        {
            "Method": "GA",
            "Distance (km)": round(ga_total, 2),
            "Trips/Routes": len(ga_results["routes"]),
            "Max Route Load (kg)": max((get_route_load(route, waste_dict) for route in ga_results["routes"]), default=0),
        },
    ]
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)

def run_statistical_verification(
    n_runs: int, 
    selected_barangays: List[str], 
    n_vehicles: int, 
    pop_size: int, 
    generations: int, 
    mutation_rate: float, 
    crossover_rate: float,
    vehicle_capacity_kg: int,
    road_graph, 
    distance_matrix: np.ndarray, 
    point_to_index: Dict[str, int],
    waste_dict: Optional[Dict[str, int]] = None,
    base_seed: int = 0,
):
    """Perform a one-sample T-test comparing repeated GA runs to the GSO baseline."""
    ga_distances = []
    gso_ref, _ = evaluate_manual_baseline(
        selected_barangays,
        road_graph,
        distance_matrix,
        point_to_index,
        vehicle_capacity_kg,
        waste_dict,
    )
    seed_chromosomes = build_seed_chromosomes(
        selected_barangays,
        n_vehicles,
        vehicle_capacity_kg,
        waste_dict,
        road_graph,
        distance_matrix,
        point_to_index,
    )

    prog = st.progress(0, text="Running GA trials...")
    for i in range(n_runs):
        _, _, trial_total, _ = run_ga(
            selected_barangays, n_vehicles, pop_size, generations, mutation_rate, crossover_rate,
            None, None, vehicle_capacity_kg, waste_dict, road_graph, distance_matrix, point_to_index,
            base_seed + i, seed_chromosomes,
        )
        ga_distances.append(trial_total)
        prog.progress((i + 1) / n_runs)

    t_stat, p_value = stats.ttest_1samp(ga_distances, popmean=gso_ref)
    ga_mean = round(np.mean(ga_distances), 2)
    mean_saving = round(gso_ref - ga_mean, 2)
    pct_saving = round(mean_saving / gso_ref * 100, 1) if gso_ref > 0 else 0
    is_sig = p_value < 0.05

    if is_sig:
        st.success(f"✅ **Significant** (p = {p_value:.2f} < 0.05). GA saved **{mean_saving:.2f} km ({pct_saving:.2f}%)** on average.")
    else:
        st.warning(f"⚠️ Not significant (p = {p_value:.2f}).")

    st.dataframe(pd.DataFrame({"Seed": [base_seed + i for i in range(n_runs)], "GA Distance": ga_distances, "Baseline": gso_ref}), hide_index=True)
    st.info(f"📄 **Results Statement:** GA Mean: {ga_mean:.2f}km, Baseline: {gso_ref:.2f}km, Improvement: {pct_saving:.2f}%. p-value={p_value:.2f}")
