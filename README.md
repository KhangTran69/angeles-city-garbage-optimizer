# Angeles City Garbage Route Optimizer

## About

The Angeles City Garbage Route Optimizer is an interactive Streamlit decision-support tool for exploring more efficient garbage collection routes across Angeles City, Philippines.

It models the collection problem as an approximate **Capacitated Vehicle Routing Problem (CVRP)** and combines graph-based distance data, a single-objective **Genetic Algorithm**, capacity-aware route assignment, and **2-opt local search**. The dashboard lets users compare routes, inspect vehicle loads, view estimated schedules and operating metrics, and export dispatch-ready summaries.

This project was developed as a **Thesis AL-1 research prototype** for route-planning analysis. It is intended for experimentation and operational planning support; waste volumes, baseline routes, travel times, and costs should be validated against official local records before real-world deployment.

**Built with:** Python, Streamlit, NetworkX, OSMnx, Folium, NumPy, SciPy, and OR-Tools.

## Features

- **Approximate CVRP workflow** — uses capacity-aware route assignment plus overload penalties
- **Road-network distances** via OpenStreetMap / OSMnx (falls back to Haversine)
- **Real-time GA progress** with best-distance trend chart
- **Fixed random seed controls** for reproducible runs
- **Interactive Folium map** with per-truck views and numbered stop markers
- **ETA schedule** with stop arrival times
- **Fuel, CO2, and cost metrics**
- **Editable barangay coordinates** with JSON persistence
- **Export package** — JSON, CSV, and ZIP outputs
- **Benchmark mode** for repeatable multi-seed comparisons
- **Experiment history** persisted to JSON for auditability
- **Dark professional UI** with custom CSS

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
cd angeles-garbage-optimizer

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
# Optional dev tooling
pip install -r requirements-dev.txt
```

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### First-Time Setup

On first run, the app may download the Angeles City road network from OpenStreetMap and cache it in `data/`. If the road graph is unavailable, the optimizer falls back to Haversine distances so the app remains usable.

## Project Structure

```
angeles-garbage-optimizer/
├── app.py                              # Streamlit UI entry point
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── .gitignore
├── pyrightconfig.json                  # Type checker config
├── .streamlit/config.toml              # Streamlit theme (optional)
│
├── src/                                # Core logic package
│   ├── __init__.py
│   ├── config.py                       # Constants, colours, defaults
│   ├── data.py                         # Barangays, coords, waste kg, depot
│   ├── utils.py                        # haversine, ETA, metrics, 2-opt
│   ├── optimization.py                 # GA engine
│   ├── map_utils.py                    # Folium map, road tracing, schedule
│   ├── session_state.py                # Streamlit session init
│   └── research_validation.py          # Baseline comparison and stability tools
│
├── data/                               # Static data files
│   ├── barangay_coords.json
│   └── angeles_city_network.graphml
│
└── tests/                              # Pytest coverage for utility and optimizer logic
    ├── test_data.py
    ├── test_optimization.py
    └── test_utils.py
```

## Usage

1. **Configure** — set vehicles, population, generations, rates in the sidebar.
2. **Select barangays** — all 33 or a subset.
3. **Run** — click "Run Optimization" and watch the GA evolve.
4. **Analyse** — route cards, optimization trend, baseline comparison, savings.
5. **Map** — per-truck or all-routes interactive map.
6. **Export** — ZIP dispatch package with JSON, route summaries, and per-vehicle CSVs.
7. **Benchmark** — run repeatable multi-seed comparisons for stability checks.

## GA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Vehicles | 3 | Number of garbage trucks |
| Population | 100 | Solutions per generation |
| Generations | 200 | Evolution cycles |
| Mutation Rate | 0.05 | Probability of random swap |
| Crossover Rate | 0.85 | Probability of route recombination |
| Vehicle Capacity | 5000 kg | Max load per truck |

## Methodology Notes

- The optimizer is **not** an exact solver and does not guarantee a globally optimal CVRP solution.
- Vehicle capacity is modeled through a capacity-aware decoder, balance-aware assignment heuristics, and overload penalties, which is more realistic than equal splitting but still heuristic.
- ETA values are estimated from configured average speed and service-time assumptions, not live traffic feeds.
- OpenRouteService routing is optional for map visualization; optimization primarily uses the cached OSMnx graph or Haversine fallback.
- Single runs can be made reproducible by reusing the same random seed; benchmark mode compares sequential seeds starting from a fixed base seed.

## Data Sources

- **Road network**: OpenStreetMap via OSMnx
- **Barangay coordinates**: Bundled JSON file editable from the app
- **Waste estimates**: Illustrative daily collection volumes for research/demo use
- **Diesel price**: DOE Region 3 reference
- **Fuel consumption**: DOTr standard for medium-duty trucks

## Limitations

- Waste volume and baseline route data are partly illustrative and should be validated against official operational records before deployment.
- The GA is stochastic; repeated runs can yield different route assignments.
- The app is best viewed as a thesis-grade decision-support prototype rather than a production dispatch system.

## Verification

```bash
python -m pytest -q
python -m py_compile app.py src/*.py tests/*.py
```

Optional static checks:

```bash
pyright
ruff check .
```

## License

AL-1 Research Project — Angeles City, Pampanga, Philippines
