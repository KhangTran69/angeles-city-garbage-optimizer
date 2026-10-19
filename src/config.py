"""
Configuration constants for Angeles City Garbage Route Optimizer.
All tunable parameters, colors, and default values live here.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path=None):
    """Bespoke pure-Python .env loader to eliminate external dependencies."""
    path = path or os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    if v.strip():
                        os.environ.setdefault(k.strip(), v.strip())


def get_config_value(name: str) -> str:
    """Read a configuration value from the environment or Streamlit secrets."""
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        from streamlit.runtime import exists as streamlit_runtime_exists

        if not streamlit_runtime_exists():
            return ""
        import streamlit as st

        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""

# Initialize environment for AL-1 Optimization
load_env()

# ── Path Configuration ────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GRAPH_PATH = os.path.join(DATA_DIR, "angeles_city_network.graphml")
DISTANCE_MATRIX_PATH = os.path.join(DATA_DIR, "distance_matrix.npy")
COORDS_PATH = os.path.join(DATA_DIR, "barangay_coords.json")
EXPERIMENTS_PATH = os.path.join(DATA_DIR, "experiment_history.json")

# ── Cost Parameters ────────────────────────────────────────────────────────────
DIESEL_PRICE = 110.00         
FUEL_PER_KM = 0.15           
COLLECTION_DAYS = 312        
CO2_PER_LITER_DIESEL = 2.68

# ── Routing Parameters (ORS) ───────────────────────────────────────────────────
ORS_API_KEY = get_config_value("ORS_API_KEY")
GOOGLE_MAPS_API_KEY = get_config_value("GOOGLE_MAPS_API_KEY")
ORS_ROUTING_PROFILE = "driving-hgv"
ORS_BASE_URL = f"https://api.openrouteservice.org/v2/directions/{ORS_ROUTING_PROFILE}"


# ── CVRP Capacity ──────────────────────────────────────────────────────────────
DEFAULT_VEHICLE_CAPACITY_KG = 5000
CVRP_OVERLOAD_PENALTY_PER_KG = 1.5

# ── Truck Routing Constraints ──────────────────────────────────────────────────
DEFAULT_TRUCK_HEIGHT = 3.5
DEFAULT_TRUCK_WIDTH = 2.5
DEFAULT_TRUCK_LENGTH = 10.0
DEFAULT_TRUCK_WEIGHT = 12.0

# ── ETA / Speed ────────────────────────────────────────────────────────────────
MAP_AVG_SPEED_KMH = 25.0
STOP_SERVICE_MINUTES = 4.0

# ── OSMnx / Map Settings ──────────────────────────────────────────────────────
OSM_BBOX = (15.10, 15.20, 120.50, 120.64)
MAP_DEFAULT_LOCATION = [15.14141069995556, 120.62967669596279]

# ── Vehicle Palette ────────────────────────────────────────────────────────────
VEHICLE_COLORS = [
    "#ff6b6b", "#ffb74d", "#64b5f6", "#69f0ae", "#ce93d8",
    "#ff8a65", "#80cbc4", "#b0bec5", "#ffe082", "#b39ddb",
    # 15 additional colors
    "#f06292", "#ba68c8", "#9575cd", "#7986cb", "#4fc3f7",
    "#4dd0e1", "#81c784", "#dce775", "#ffd54f", "#e57373", 
    "#f0bac0", "#c5e1a5", "#fff59d", "#bcaaa4", "#eeeeee"
]

# ── Route Segment Palette ──────────────────────────────────────────────────────
SEGMENT_COLORS = [
    "#e74c3c", "#3498db", "#f1c40f", "#2ecc71", "#9b59b6",
]

# ── Default GA Parameters ─────────────────────────────────────────────────────
DEFAULT_N_VEHICLES = 25
DEFAULT_POP_SIZE = 100
DEFAULT_GENERATIONS = 200
DEFAULT_MUTATION_RATE = 0.05
DEFAULT_CROSSOVER_RATE = 0.85
DEFAULT_BASELINE_OVERHEAD = 25
DEFAULT_RANDOM_SEED = 42
DEFAULT_BENCHMARK_RUNS = 5

# ── App metadata ───────────────────────────────────────────────────────────────
APP_TITLE = "Angeles City Garbage Route Optimizer"
APP_SUBTITLE = "Optimizing Route Efficiency using Graph Theory and Genetic Algorithms (Thesis AL-1)"

# ── UI Constants ───────────────────────────────────────────────────────────────
MAX_HISTORY_SIZE = 10

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base Styles & Typography */
html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
}
h1, h2, h3, .main-header h1, .section-title { 
    font-family: 'Outfit', sans-serif; 
}
[data-testid="stMetricValue"], .metric-value, .route-dist, .gen-log, .comparison-table { 
    font-family: 'JetBrains Mono', monospace; 
}

.stApp { 
    background: radial-gradient(circle at top right, #1b2a1d, #0a0f0d);
    color: #e8f5e9; 
}

/* Glassmorphism Design System */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(105, 240, 174, 0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}

/* Hero Header & Branding */
.main-header { 
    background: linear-gradient(90deg, rgba(27, 94, 32, 0.8) 0%, rgba(10, 15, 13, 0.9) 100%);
    border: 1px solid rgba(105, 240, 174, 0.2);
    border-radius: 20px; 
    padding: 40px; 
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}
.main-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(105, 240, 174, 0.1) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 { 
    font-size: 2.8rem; 
    font-weight: 800; 
    color: #69f0ae; 
    margin: 0; 
    letter-spacing: -1.5px;
    text-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
.main-header p { 
    color: #a5d6a7; 
    font-size: 1.1rem; 
    margin: 12px 0 0; 
    opacity: 0.9;
    font-weight: 400;
}

/* Sidebar Specific */
div[data-testid="stSidebar"] { 
    background: #070a08; 
    border-right: 1px solid rgba(105, 240, 174, 0.1); 
}
.sidebar-badge {
    background: rgba(105, 240, 174, 0.1);
    color: #69f0ae;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 12px;
    border: 1px solid rgba(105, 240, 174, 0.3);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 20px;
    display: inline-block;
}
.sidebar-section-header {
    font-size: 0.8rem;
    font-weight: 700;
    color: #69f0ae;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 25px 0 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Advanced Metric Cards */
.metric-card { 
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(105, 240, 174, 0.1);
    border-radius: 16px; 
    padding: 24px; 
    text-align: center; 
    transition: transform 0.3s ease, border-color 0.3s ease;
}
.metric-card:hover { 
    transform: translateY(-5px);
    border-color: rgba(105, 240, 174, 0.4);
}
.metric-value { 
    font-size: 2.2rem; 
    font-weight: 700; 
    color: #69f0ae; 
    line-height: 1; 
}
.metric-label { 
    font-size: 0.8rem; 
    color: #81c784; 
    margin-top: 10px; 
    text-transform: uppercase; 
    letter-spacing: 1px;
    opacity: 0.8;
}

/* Route & Utilization Design */
.route-card { 
    background: rgba(27, 94, 32, 0.1);
    border: 1px solid rgba(105, 240, 174, 0.15);
    border-radius: 16px; 
    padding: 24px; 
    margin-bottom: 18px; 
    transition: all 0.3s ease;
}
.route-card:hover { 
    background: rgba(27, 94, 32, 0.15);
    border-color: rgba(105, 240, 174, 0.4);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
}
.utilization-container {
    margin-top: 15px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    position: relative;
}
.utilization-bar {
    height: 100%;
    border-radius: 8px;
    transition: width 1s ease-in-out;
}

/* Tab & Table Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(105, 240, 174, 0.1);
    border-radius: 10px 10px 0 0;
    padding: 10px 20px;
    color: #a5d6a7;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: rgba(27, 94, 32, 0.3) !important;
    border-color: #69f0ae !important;
    color: #69f0ae !important;
}

.section-title { 
    font-size: 1.3rem; 
    font-weight: 700; 
    color: #69f0ae; 
    margin: 35px 0 20px; 
    display: flex;
    align-items: center;
    gap: 12px;
}
.section-title::before {
    content: '';
    width: 4px;
    height: 24px;
    background: #69f0ae;
    border-radius: 4px;
}

/* Global Component Overrides */
.stButton > button { 
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    color: white; 
    border: none;
    border-radius: 12px; 
    padding: 12px 24px;
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(27, 94, 32, 0.3);
}
.stButton > button:hover { 
    transform: scale(1.02);
    box-shadow: 0 6px 20px rgba(105, 240, 174, 0.4);
}
.gen-log { 
    background: #050806; 
    border: 1px solid rgba(105, 240, 174, 0.2); 
    border-radius: 12px; 
    padding: 20px; 
    font-size: 0.8rem; 
    color: #69f0ae; 
    max-height: 250px; 
    overflow-y: auto; 
}

/* Tooltip & Legend Styling */
.map-legend-item {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    color: #e8f5e9;
}
.legend-color-box {
    width: 14px;
    height: 14px;
    border-radius: 3px;
}
.stop-registry-scroll {
    max-height: 450px;
    overflow-y: auto;
    padding-right: 10px;
    margin-top: 15px;
}
/* Scrollbar Styling for Command Center */
.stop-registry-scroll::-webkit-scrollbar { width: 4px; }
.stop-registry-scroll::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
.stop-registry-scroll::-webkit-scrollbar-thumb { background: rgba(105, 240, 174, 0.2); border-radius: 10px; }
</style>
"""
