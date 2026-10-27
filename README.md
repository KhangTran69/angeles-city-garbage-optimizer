# Angeles City Garbage Route Optimizer

## About

An interactive Streamlit tool for optimizing garbage collection routes across Angeles City, Philippines. It approximates a **Capacitated Vehicle Routing Problem (CVRP)** using graph distances, a **Genetic Algorithm**, capacity-aware assignment, and **2-opt local search**. The dashboard compares routes, loads, schedules, operating metrics, and exports.

Built as a **Thesis AL-1 research prototype** for planning and experimentation. Validate operational data before deployment.

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

The tactical map works without an API key using the bundled coordinates and local route data. To enable optional ORS road-curved routing, copy `.env.example` to `.env` and add an `ORS_API_KEY`. For Streamlit hosting, add the same value as a secret named `ORS_API_KEY`.

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

GitHub Actions runs these on every push and pull request:

```bash
python -m pytest -q
python -m ruff check src tests --select E9,F
```

Optional static checks, advisory until the existing backlog is cleared:

```bash
python -m basedpyright src
```

`ruff.toml` still selects `E`, `F`, `I`, and `UP` for local use, but the CI gate
is narrower on purpose: the repository has not yet been cleaned up to that
standard. Tighten `--select` in `.github/workflows/ci.yml` as the backlog shrinks.

## License

AL-1 Research Project — Angeles City, Pampanga, Philippines
