"""
Angeles City Garbage Collection Route Data Generator
Converts scanned GSO schedule into CVRP-ready Python data structures.
Uses OSMnx for road network and Nominatim for geocoding.
"""

import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
import folium
import json
import logging
from typing import List, Dict, Tuple, Optional
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Angeles City bounding box (approximate)
ANGELES_BOUNDS = {"north": 15.20, "south": 15.10, "east": 120.65, "west": 120.55}

# Waste estimation rules
WASTE_RULES = {
    "school": {"elementary": 200, "high_school": 300, "college": 300},
    "hospital": 400,
    "street_segment": {"residential": 50, "main_road": 80},
    "landmark": 100,
    "rotonda": 60,
}

# Scanned schedule data
RAW_ROUTES = {
    "TRUCK_1": {
        "driver": "Antonio Cleope",
        "helpers": ["Romeo Candelaria", "Ronnie Dungca", "Vic Salunga", "Mario Gonzales"],
        "route": [
            "Sto Rosario St. (corner Rizal St., up to Sto. Domingo boundary)",
            "Plaridel St. (corner Sto. Rosario St. up to corner Miranda St.)",
            "Miranda St. (corner Sto. Entierro St., up to Rotonda San Nicolas)",
        ],
    },
    "TRUCK_2": {
        "driver": "Willy dela Cruz",
        "helpers": ["Emmanuel Dacanay", "Rommel Valdez", "Randy Infante", "Enrico Tanhueco"],
        "route": [
            "New Abacan Bypass Road (Jaoville and Pulung Maragul both side)",
            "Arayat Blvd. (corner San Francisco St., up to corner Fil-Am Road)",
            "Pampang Road (corner Arayat Blvd. up to corner Kalayaan St.)",
            "San Francisco St. (corner Visitacion St., up to corner Arayat Blvd.)",
        ],
    },
    "TRUCK_3": {
        "driver": "Edwin Benedicto",
        "helpers": ["Angelo Calma", "Eduardo Dizon", "Dranreb Boneo"],
        "route": [
            "Angeles Magalang Road (Dieg's up along Cutud boundary)",
            "Sto Entierro",
            "Lakandula St.",
            "Miranda St. (corner Lakandula up to corner Sto. Entierro St.)",
            "Hospital Ning Ona",
            "174th (Precint)",
        ],
    },
    "TRUCK_4": {
        "driver": "Antonio Atienza",
        "helpers": ["Ruben Calaquian", "Richard Maliwat", "Jayvee Cortez", "John Kerzee Atienza"],
        "route": [
            "Jake Gonzales St., Henson St., Rizal Ext., along cut-cut cemetery",
            "G. Valdez St.",
            "Angeles Science High School, Dr. Clemente Elem. School",
            "Villa Ngelina",
        ],
    },
    "TRUCK_5": {
        "driver": "Ricky Benedicto",
        "helpers": ["Erwin Capulong", "Enrico Capulong", "Rodrigo Valdez", "Cresencio Bina-Loac"],
        "route": [
            "Mac-Arthur Hi-way (boundary Sto. Domingo up to Marisol Rotonda)",
            "Kalayaan St.",
        ],
    },
    "TRUCK_6": {
        "driver": "Michael Muriel",
        "helpers": ["Jose Mari de Leon", "Marionne Garcia", "Homer Isidro", "Melvin Jay Villaflor"],
        "route": [
            "City High Pampang (Mon, Wed, Fri)",
            "Gueco Elem. (Tues, Thur)",
            "Rafael Main (Mon, Wed, Fri)",
            "City College (Mon, Wed, Sat)",
            "Trade School (Tues, Thurs, Sat)",
            "City Hygene (Mon, Thur)",
            "City High Pampang (Wed, Fri)",
        ],
    },
}

# Depot location (Angeles City Transfer Station)
DEPOT_COORD = (15.14141069995556, 120.62967669596279)


def setup_geocoder():
    """Initialize Nominatim geocoder."""
    return Nominatim(user_agent="AL1_garbage_optimizer")


def geocode_location(geocoder, location_name: str) -> Optional[Tuple[float, float]]:
    """Geocode a location within Angeles City."""
    try:
        query = f"{location_name}, Angeles City, Pampanga, Philippines"
        location = geocoder.geocode(query, timeout=10)
        if location:
            lat, lon = location.latitude, location.longitude
            # Check if within bounds
            if (
                ANGELES_BOUNDS["south"] <= lat <= ANGELES_BOUNDS["north"]
                and ANGELES_BOUNDS["west"] <= lon <= ANGELES_BOUNDS["east"]
            ):
                return (lat, lon)
            else:
                logger.warning(
                    f"Location {location_name} is outside Angeles City bounds: {lat}, {lon}"
                )
                return None
        else:
            logger.warning(f"Could not geocode: {location_name}")
            return None
    except Exception as e:
        logger.error(f"Geocoding error for {location_name}: {e}")
        return None


def load_angeles_graph():
    """Load Angeles City road network."""
    logger.info("Loading Angeles City road network...")
    bbox = (
        ANGELES_BOUNDS["south"],
        ANGELES_BOUNDS["north"],
        ANGELES_BOUNDS["east"],
        ANGELES_BOUNDS["west"],
    )
    graph = ox.graph_from_bbox(bbox, network_type="drive")
    return graph


def create_street_segments(
    graph, start_point, end_point, street_name: str, segment_type="residential"
) -> List[Dict]:
    """Create pickup points along a street segment every 150m."""
    try:
        start_node = ox.distance.nearest_nodes(graph, start_point[1], start_point[0])
        end_node = ox.distance.nearest_nodes(graph, end_point[1], end_point[0])

        # Get shortest path
        route = nx.shortest_path(graph, start_node, end_node, weight="length")

        points = []
        total_distance = 0
        segment_counter = 1

        for i in range(len(route) - 1):
            node1 = route[i]
            node2 = route[i + 1]
            edge_data = graph.get_edge_data(node1, node2, 0) or {}

            if "length" in edge_data:
                segment_length = edge_data["length"] / 1000  # Convert to km
                total_distance += segment_length

                # Place points every 150m (0.15 km)
                while total_distance >= 0.15:
                    # Interpolate position
                    fraction = 0.15 / segment_length
                    lat1, lon1 = graph.nodes[node1]["y"], graph.nodes[node1]["x"]
                    lat2, lon2 = graph.nodes[node2]["y"], graph.nodes[node2]["x"]

                    lat = lat1 + fraction * (lat2 - lat1)
                    lon = lon1 + fraction * (lon2 - lon1)

                    waste_kg = WASTE_RULES["street_segment"][segment_type]

                    point = {
                        "id": f"{street_name.replace(' ', '_').upper()}_SEG{segment_counter}",
                        "name": f"{street_name} Segment {segment_counter}",
                        "type": "street_segment",
                        "lat": lat,
                        "lon": lon,
                        "waste_kg": waste_kg,
                        "service_days": ["All"],
                        "time_window": (None, None),
                        "notes": f"Auto-generated from {street_name}",
                    }
                    points.append(point)
                    segment_counter += 1
                    total_distance -= 0.15

        return points
    except Exception as e:
        logger.error(f"Error creating segments for {street_name}: {e}")
        return []


def parse_route_to_points(geocoder, graph, truck_id: str, route_items: List[str]) -> List[Dict]:
    """Convert route descriptions to pickup points."""
    points = []
    failed_geocodes = []

    for item in route_items:
        # Parse different types of route items
        if "School" in item or "College" in item or "Elem." in item:
            # Schools
            school_name = item.split(",")[0].strip()
            coord = geocode_location(geocoder, school_name)
            if coord:
                school_type = (
                    "elementary"
                    if "Elem." in item
                    else ("college" if "College" in item else "high_school")
                )
                waste_kg = WASTE_RULES["school"][school_type]
                point = {
                    "id": f"{school_name.replace(' ', '_').upper()}",
                    "name": school_name,
                    "type": "school",
                    "lat": coord[0],
                    "lon": coord[1],
                    "waste_kg": waste_kg,
                    "service_days": ["All"],  # Parse days if needed
                    "time_window": (None, None),
                    "notes": f"Truck {truck_id}",
                }
                points.append(point)
            else:
                failed_geocodes.append((school_name, "school"))

        elif "Hospital" in item:
            # Hospitals
            coord = geocode_location(geocoder, item)
            if coord:
                point = {
                    "id": f"{item.replace(' ', '_').upper()}",
                    "name": item,
                    "type": "hospital",
                    "lat": coord[0],
                    "lon": coord[1],
                    "waste_kg": WASTE_RULES["hospital"],
                    "service_days": ["All"],
                    "time_window": (None, None),
                    "notes": f"Truck {truck_id}",
                }
                points.append(point)
            else:
                failed_geocodes.append((item, "hospital"))

        elif "St." in item or "Street" in item or "Road" in item or "Blvd" in item:
            # Street segments
            street_name = item.split("(")[0].strip()
            coord = geocode_location(geocoder, street_name)
            if coord:
                # For simplicity, create a single point; in full implementation, use create_street_segments
                waste_kg = WASTE_RULES["street_segment"]["residential"]
                point = {
                    "id": f"{street_name.replace(' ', '_').upper()}",
                    "name": street_name,
                    "type": "street_segment",
                    "lat": coord[0],
                    "lon": coord[1],
                    "waste_kg": waste_kg,
                    "service_days": ["All"],
                    "time_window": (None, None),
                    "notes": f"Truck {truck_id}",
                }
                points.append(point)
            else:
                failed_geocodes.append((street_name, "street_segment"))

        else:
            # Other landmarks
            coord = geocode_location(geocoder, item)
            if coord:
                waste_kg = WASTE_RULES["landmark"]
                point = {
                    "id": f"{item.replace(' ', '_').upper()}",
                    "name": item,
                    "type": "landmark",
                    "lat": coord[0],
                    "lon": coord[1],
                    "waste_kg": waste_kg,
                    "service_days": ["All"],
                    "time_window": (None, None),
                    "notes": f"Truck {truck_id}",
                }
                points.append(point)
            else:
                failed_geocodes.append((item, "landmark"))

    return points, failed_geocodes


def create_truck_fleet() -> List[Dict]:
    """Create truck fleet configuration."""
    fleet = []
    for truck_id, data in RAW_ROUTES.items():
        truck = {
            "id": truck_id,
            "driver": data["driver"],
            "capacity_kg": 3000,  # Mini dump truck assumption
            "depot_location": DEPOT_COORD,
            "operating_hours": (8, 17),  # 08:00 to 17:00
        }
        fleet.append(truck)
    return fleet


def validate_coordinates(points: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Validate that all coordinates are within Angeles City bounds."""
    valid = []
    invalid = []
    for point in points:
        lat, lon = point["lat"], point["lon"]
        if (
            ANGELES_BOUNDS["south"] <= lat <= ANGELES_BOUNDS["north"]
            and ANGELES_BOUNDS["west"] <= lon <= ANGELES_BOUNDS["east"]
        ):
            valid.append(point)
        else:
            invalid.append(point)
    return valid, invalid


def generate_summary_report(points: List[Dict], failed_geocodes: List[Tuple]) -> str:
    """Generate summary report."""
    total_points = len(points)
    total_waste = sum(p["waste_kg"] for p in points)
    types_count = {}
    for p in points:
        types_count[p["type"]] = types_count.get(p["type"], 0) + 1

    report = f"""
Angeles City Garbage Collection Route Data Summary
================================================

Total Pickup Points: {total_points}
Total Estimated Daily Waste: {total_waste} kg

Points by Type:
"""
    for t, count in types_count.items():
        report += f"- {t}: {count}\n"

    report += f"\nFailed Geocoding: {len(failed_geocodes)} locations\n"
    for name, typ in failed_geocodes:
        report += f"- {name} ({typ})\n"

    return report


def create_folium_map(points: List[Dict], depot: Tuple[float, float]) -> folium.Map:
    """Create Folium map for visualization."""
    m = folium.Map(location=[15.15, 120.60], zoom_start=13)

    # Add depot
    folium.Marker(
        location=depot,
        popup="Angeles City Transfer Station",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)

    # Add pickup points
    for point in points:
        folium.Marker(
            location=[point["lat"], point["lon"]],
            popup=f"{point['name']} ({point['waste_kg']}kg)",
            icon=folium.Icon(color="blue", icon="trash"),
        ).add_to(m)

    return m


def main():
    """Main function to generate route data."""
    logger.info("Starting Angeles City route data generation...")

    # Setup
    geocoder = setup_geocoder()
    graph = load_angeles_graph()

    # Process routes
    all_points = []
    all_failed = []

    for truck_id, truck_data in RAW_ROUTES.items():
        logger.info(f"Processing {truck_id}...")
        points, failed = parse_route_to_points(geocoder, graph, truck_id, truck_data["route"])
        all_points.extend(points)
        all_failed.extend(failed)

        # Add delay to avoid geocoding rate limits
        time.sleep(1)

    # Validate coordinates
    valid_points, invalid_points = validate_coordinates(all_points)

    # Create truck fleet
    truck_fleet = create_truck_fleet()

    # Generate summary
    summary = generate_summary_report(valid_points, all_failed)

    # Save to file
    with open("angeles_city_routes.py", "w") as f:
        f.write(
            '"""\nAngeles City Garbage Collection Route Data\nGenerated from scanned GSO schedule\n"""\n\n'
        )
        f.write("PICKUP_POINTS = ")
        f.write(json.dumps(valid_points, indent=2))
        f.write("\n\nTRUCK_FLEET = ")
        f.write(json.dumps(truck_fleet, indent=2))
        f.write("\n\nDEPOT_COORD = ")
        f.write(str(DEPOT_COORD))
        f.write("\n")

    # Save failed geocodes
    if all_failed:
        df_failed = pd.DataFrame(all_failed, columns=["Location", "Type"])
        df_failed.to_csv("failed_geocodes.csv", index=False)

    # Create map
    m = create_folium_map(valid_points, DEPOT_COORD)
    m.save("angeles_city_routes_map.html")

    # Print summary
    print(summary)
    print(f"\nSaved to: angeles_city_routes.py")
    print(f"Map saved to: angeles_city_routes_map.html")
    if all_failed:
        print(f"Failed geocodes saved to: failed_geocodes.csv")


if __name__ == "__main__":
    main()
