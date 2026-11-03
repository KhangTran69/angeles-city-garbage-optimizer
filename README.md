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

### Reproduce the optimizer sensitivity report

Run `python -m src.benchmark` from the repository root to compare the capacity-aware greedy baseline with GA seeds 42–44 for three demand cases: the bundled base estimates and uniform -10% / +10% changes. The same six barangays (Amsic, Anunas, Balibago, Capaya, Claro M. Recto, and Cuayan), three vehicles, 10,000 kg capacity per vehicle, built-in default coordinates, Haversine distance matrix, population 20, 30 generations, mutation rate 0.05, and crossover rate 0.85 are used in every case. Positive kilogram values are rounded to the nearest integer, with half-kilogram ties rounded up. Scenario construction copies the source estimates and does not alter them.

The schema-versioned JSON includes exact demand values, a canonical SHA-256 input fingerprint for each scenario, source Git revision / dirty state / source-tree SHA-256, Python and NumPy versions, greedy distance and feasibility, and each seed's GA distance, feasibility, runtime, and timeout status. The source-tree hash covers tracked and nonignored untracked project files, including documentation; it excludes common caches/build output, local `.env` files (except `.env.example`), and the exact `--json-out` file. Outside a usable Git worktree, it falls back to hashing project files with those exclusions. Summaries include feasibility rate, distance spread, mean runtime, and timeout rate. Runtime varies by machine and is reported for context, not as a performance guarantee. Use `--json-out report.json` to save the report.

An optional `--timeout-seconds 5` sets a positive finite cooperative budget independently for each scenario/seed GA run. The per-run deadline starts at invocation; setup before the first generation check and an active generation cannot be interrupted, so runtime may exceed the budget in either case. The best-so-far route is retained and reported with its feasibility and timeout status. The timeout setting and early-stopping patience/minimum improvement are included in each scenario fingerprint.

Waste estimates are illustrative, not validated municipal measurements. Haversine distances are straight-line estimates, not road distances. This report measures algorithm sensitivity on the specified inputs; it is not operational or field validation.

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
- Benchmark results include per-seed runtime, route feasibility, and timeout status, plus mean runtime, feasibility rate, and timeout rate. Feasibility means every selected barangay appears exactly once and each route stays within vehicle capacity. The optional `timeout_seconds` deadline begins at invocation, but setup before the first generation check and an active generation cannot be interrupted; runtime can therefore exceed the budget in either case. The solver retains its best-so-far route, and timed-out routes still contribute to distance and feasibility summaries. The report also records the GA's early-stopping patience and minimum-improvement settings.

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
