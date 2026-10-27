"""
Data definitions for Angeles City Garbage Route Optimizer.
Barangays, coordinates, waste estimates, and depot location.
"""

import json
import math
import os
from typing import Dict, Tuple

from src.config import COORDS_PATH, OSM_BBOX

# ── Official 33 Barangays of Angeles City ──────────────────────────────────────
BARANGAYS = [
    "Agapito del Rosario",
    "Amsic",
    "Anunas",
    "Balibago",
    "Capaya",
    "Claro M. Recto",
    "Cuayan",
    "Cutcut",
    "Cutud",
    "Lourdes North West",
    "Lourdes Sur (Talimundoc)",
    "Lourdes Sur East",
    "Malabañas",
    "Margot",
    "Marisol (Ninoy Aquino)",
    "Mining",
    "Pampang (Santo Niño)",
    "Pandan",
    "Pulung Bulu",
    "Pulung Cacutud",
    "Pulung Maragul",
    "Salapungan",
    "San José",
    "San Nicolas",
    "Santa Teresita",
    "Santa Trinidad",
    "Santo Cristo",
    "Santo Domingo",
    "Santo Rosario (Población)",
    "Sapalibutad",
    "Sapangbato",
    "Tabun",
    "Virgen Delos Remedios",
]

# ── Angeles City Transfer Station ──────────────────────────────────────────────────────────────────────
DEPOT = "Angeles City Transfer Station"
DEPOT_COORD = (15.14141069995556, 120.62967669596279)

DATA_PROVENANCE = {
    "barangay_coordinates": "Editable JSON dataset bundled with the app; verify against official GIS records before field use.",
    "waste_estimates": "Illustrative research inputs derived from simplified per-capita assumptions, not fully validated operational logs.",
    "manual_gso_baseline": "Prototype baseline sectors for comparison; replace with documented dispatch records for formal evaluation.",
}


# ── Barangay Coordinates (latitude, longitude) ─────────────────────────────────
DEFAULT_BARANGAY_COORDS = {
    "Agapito del Rosario": (15.144314032643162, 120.5895143627856),
    "Amsic": (15.15585270907212, 120.5711102229492),
    "Anunas": (15.155761613285806, 120.55345285759914),
    "Balibago": (15.166482078071924, 120.59090982479879),
    "Capaya": (15.141262643924124, 120.62671956712708),
    "Claro M. Recto": (15.147679223436391, 120.59265552097031),
    "Cuayan": (15.147075171205898, 120.54849706905532),
    "Cutcut": (15.137117887469142, 120.58257735840519),
    "Cutud": (15.175334358903227, 120.6266527959632),
    "Lourdes North West": (15.145000554193194, 120.5847417935733),
    "Lourdes Sur (Talimundoc)": (15.14202410517985, 120.59076958062035),
    "Lourdes Sur East": (15.143355960747465, 120.59328582479841),
    "Malabañas": (15.15159321178637, 120.58985722479844),
    "Margot": (15.170715179813422, 120.53470730945637),
    "Marisol (Ninoy Aquino)": (15.153297441445833, 120.59226482474382),
    "Mining": (15.143081767186729, 120.60459490258394),
    "Pampang (Santo Niño)": (15.146521465983724, 120.57595372479837),
    "Pandan": (15.146829518328397, 120.60554576159188),
    "Pulung Bulu": (15.1361170674414, 120.59514444491188),
    "Pulung Cacutud": (15.165786792363994, 120.6166618959631),
    "Pulung Maragul": (15.161947362194583, 120.60777982664803),
    "Salapungan": (15.146804310125386, 120.59732979528673),
    "San José": (15.133645478424059, 120.5944536707712),
    "San Nicolas": (15.138893947253278, 120.58506654848571),
    "Santa Teresita": (15.15036154296399, 120.58564878724228),
    "Santa Trinidad": (15.14243853473967, 120.58192569167949),
    "Santo Cristo": (15.14296728284809, 120.59854924524177),
    "Santo Domingo": (15.127845552005104, 120.60069477483226),
    "Santo Rosario (Población)": (15.135378225671289, 120.59095889236859),
    "Sapalibutad": (15.158317707501068, 120.63204402479857),
    "Sapangbato": (15.17102616981832, 120.51500915363448),
    "Tabun": (15.149155489334131, 120.61410752479838),
    "Virgen Delos Remedios": (15.149756312956685, 120.59274586712732),
}


def load_barangay_coords(path: str = COORDS_PATH) -> Dict[str, Tuple[float, float]]:
    """Load persisted barangay coordinates, falling back to bundled defaults."""
    if not os.path.exists(path):
        return dict(DEFAULT_BARANGAY_COORDS)

    try:
        with open(path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return dict(DEFAULT_BARANGAY_COORDS)

    coords: Dict[str, Tuple[float, float]] = {}
    south, north, west, east = OSM_BBOX
    for barangay in BARANGAYS:
        raw_value = raw_data.get(barangay)
        if (
            isinstance(raw_value, (list, tuple))
            and len(raw_value) == 2
            and all(isinstance(component, (int, float)) for component in raw_value)
        ):
            lat, lon = float(raw_value[0]), float(raw_value[1])
            if (
                math.isfinite(lat) and math.isfinite(lon)
                and south <= lat <= north and west <= lon <= east
            ):
                coords[barangay] = (lat, lon)
            else:
                coords[barangay] = DEFAULT_BARANGAY_COORDS[barangay]
        else:
            coords[barangay] = DEFAULT_BARANGAY_COORDS[barangay]
    return coords


def save_barangay_coords(
    coords: Dict[str, Tuple[float, float]],
    path: str = COORDS_PATH,
) -> Dict[str, Tuple[float, float]]:
    """Persist validated coordinates to disk and refresh the in-memory mapping."""
    cleaned_coords: Dict[str, Tuple[float, float]] = {}
    south, north, west, east = OSM_BBOX
    for barangay in BARANGAYS:
        lat, lon = coords.get(barangay, DEFAULT_BARANGAY_COORDS[barangay])
        lat, lon = float(lat), float(lon)
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise ValueError(f"Coordinates for {barangay} must be finite numbers.")
        if not south <= lat <= north or not west <= lon <= east:
            raise ValueError(f"Coordinates for {barangay} fall outside the configured city bounds.")
        cleaned_coords[barangay] = (lat, lon)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(cleaned_coords, file, indent=2, ensure_ascii=False)

    BARANGAY_COORDS.clear()
    BARANGAY_COORDS.update(cleaned_coords)
    # Any saved distance matrix is now stale; its coordinate signature no longer
    # matches, so load_cached_distance_matrix() rebuilds it on the next run.
    return dict(cleaned_coords)


BARANGAY_COORDS = load_barangay_coords()

# ── Manual Historical GSO Baseline (Simulated Geographic Sector Routing) ───────
MANUAL_GSO_BASELINE = [
    ["Capaya", "Tabun", "Sapalibutad", "Pulung Maragul", "Balibago", "Santa Teresita"],
    [
        "Pulung Cacutud",
        "Mining",
        "Salapungan",
        "Lourdes Sur East",
        "Agapito del Rosario",
        "Claro M. Recto",
        "Lourdes Sur (Talimundoc)",
        "San Nicolas",
        "Pulung Bulu",
        "Santo Rosario (Población)",
    ],
    [
        "Cutud",
        "Pandan",
        "Santo Cristo",
        "San José",
        "Virgen Delos Remedios",
        "Cutcut",
        "Pampang (Santo Niño)",
        "Cuayan",
    ],
    [
        "Santo Domingo",
        "Lourdes North West",
        "Marisol (Ninoy Aquino)",
        "Malabañas",
        "Amsic",
        "Anunas",
        "Margot",
        "Sapangbato",
        "Santa Trinidad",
    ],
]

# ── Estimated daily waste per barangay (kg) ────────────────────────────────────
# Note: Based mathematically on the DENR standard 0.4kg/capita daily generation scaling.
BARANGAY_WASTE_KG = {
    "Agapito del Rosario": 2240,
    "Amsic": 3300,
    "Anunas": 2590,
    "Balibago": 6470,
    "Capaya": 2060,
    "Claro M. Recto": 2430,
    "Cuayan": 1770,
    "Cutcut": 2900,
    "Cutud": 3560,
    "Lourdes North West": 2350,
    "Lourdes Sur (Talimundoc)": 2010,
    "Lourdes Sur East": 2160,
    "Malabañas": 4440,
    "Margot": 1430,
    "Marisol (Ninoy Aquino)": 2770,
    "Mining": 1660,
    "Pampang (Santo Niño)": 3750,
    "Pandan": 2590,
    "Pulung Bulu": 1900,
    "Pulung Cacutud": 2350,
    "Pulung Maragul": 3040,
    "Salapungan": 1800,
    "San José": 2480,
    "San Nicolas": 2140,
    "Santa Teresita": 1900,
    "Santa Trinidad": 1720,
    "Santo Cristo": 2320,
    "Santo Domingo": 1500,
    "Santo Rosario (Población)": 2770,
    "Sapalibutad": 3830,
    "Sapangbato": 1140,
    "Tabun": 3480,
    "Virgen Delos Remedios": 1820,
}

# ── Service Time Windows (minutes from midnight) ───────────────────────────────
# Morning: 360-600 (06:00 - 10:00)
# Midday: 600-840 (10:00 - 14:00)
# Afternoon: 840-1080 (14:00 - 18:00)
BARANGAY_TIME_WINDOWS = {
    "Agapito del Rosario": (360, 600),
    "Amsic": (360, 600),
    "Anunas": (360, 600),
    "Balibago": (360, 600),
    "Capaya": (360, 600),
    "Claro M. Recto": (360, 600),
    "Cuayan": (360, 600),
    "Cutcut": (360, 600),
    "Cutud": (360, 600),
    "Lourdes North West": (360, 600),
    "Lourdes Sur (Talimundoc)": (360, 600),
    "Lourdes Sur East": (600, 840),
    "Malabañas": (600, 840),
    "Margot": (600, 840),
    "Marisol (Ninoy Aquino)": (600, 840),
    "Mining": (600, 840),
    "Pampang (Santo Niño)": (600, 840),
    "Pandan": (600, 840),
    "Pulung Bulu": (600, 840),
    "Pulung Cacutud": (600, 840),
    "Pulung Maragul": (600, 840),
    "Salapungan": (600, 840),
    "San José": (840, 1080),
    "San Nicolas": (840, 1080),
    "Santa Teresita": (840, 1080),
    "Santa Trinidad": (840, 1080),
    "Santo Cristo": (840, 1080),
    "Santo Domingo": (840, 1080),
    "Santo Rosario (Población)": (840, 1080),
    "Sapalibutad": (840, 1080),
    "Sapangbato": (840, 1080),
    "Tabun": (840, 1080),
    "Virgen Delos Remedios": (840, 1080),
}

# ── Ground-Truth Google Maps Routes (Trucks 1-5) ──────────────────────────────
GOOGLE_ROUTES = [
    [(15.1413667, 120.6296767), (15.1223439, 120.5940988), (15.1386843, 120.5875438), (15.136378, 120.5877386), (15.1378357, 120.5888598), (15.1361128, 120.5915305), (15.1386843, 120.5875438), (15.1413667, 120.6296767)],
    [(15.1413667, 120.6296767), (15.1547393, 120.6051794), (15.148715, 120.5849035), (15.1499453, 120.5842684), (15.1441721, 120.5841265), (15.1413667, 120.6296767)],
    [(15.1413667, 120.6296767), (15.181632, 120.628241), (15.1461902, 120.581271), (15.1347717, 120.5927292), (15.1406659, 120.5949253), (15.1413667, 120.6296767)],
    [(15.1413667, 120.6296767), (15.1313793, 120.5926982), (15.1341864, 120.5840197), (15.1511958, 120.5912123), (15.1449838, 120.5930707), (15.1413667, 120.6296767)],
    [(15.1413667, 120.6296767), (15.1223439, 120.5940988), (15.1520318, 120.5917372), (15.1450964, 120.5854603), (15.1413667, 120.6296767)]
]

GOOGLE_MAPS_LINKS = {
    1: "https://maps.app.goo.gl/hAzek7LbArg25xdQ6",
    2: "https://maps.app.goo.gl/k2fW283rMsxpSDRj7",
    3: "https://maps.app.goo.gl/8PXd6RnsYNJX2D418",
    4: "https://maps.app.goo.gl/154Qd5QJGMf1t9iz6",
    5: "https://maps.app.goo.gl/BaixwZPXBbtcvfe26"
}

TRUCK_DRIVERS = {
    1: "Antonio Cleope",
    2: "Willy Dela Cruz",
    3: "Edwin Benedicto",
    4: "Antonio Atienza",
    5: "Ricky Benedicto"
}

TRUCK_SECTORS = {
    1: "Sto. Rosario & Miranda Sector",
    2: "Abacan & Pampang Sector",
    3: "Magalang Road & Central Sector",
    4: "Schools & Heritage Sector",
    5: "Highway & Kalayaan Sector"
}

DETAILED_GSO_ROUTES = [
    ["Sto. Domingo Boundary", "Corner Rizal St. & Sto. Rosario St.", "Corner Plaridel St. & Sto. Rosario St.", "Corner Plaridel St. & Miranda St.", "Corner Miranda St. & Sto. Entierro St.", "Rotonda Jollibee San Nicolas"],
    ["New Abacan Bypass (Jaoville Ave, Pandan)", "Visitacion St & San Francisco St.", "Arayat Blvd & Arayat Rd", "Pampang Rd & Kalayaan Rd"],
    ["Magalang Road (Dieg’s to Cutud Boundary)", "Hospital ning Ona / 174th Precinct", "Lakandula St.", "Miranda St. (cor. Lakandula)", "Sto. Entierro St."],
    ["Villa Angelina Sports Center", "Holy Rosary Catholic Cemetery", "Jake Gonzales Blvd", "Angeles City Science High School"],
    ["MacArthur Hi-way (Sto. Domingo Boundary)", "MacArthur Hi-way (Marisol Rotonda)", "Kalayaan St."]
]
