import os
import csv
import json
import logging
import math
import pprint
import pandas as pd
import folium
import osmnx as ox
import networkx as nx
import pyproj
from typing import List, Dict, Any, Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from shapely.geometry import Point, LineString
from shapely.ops import transform

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- CONFIGURATION ---
BOUNDS = {'south': 15.10, 'north': 15.20, 'west': 120.55, 'east': 120.65}

TRUCK_FLEET = [
    {'id': 'TRUCK_1', 'driver': 'Antonio Cleope', 'capacity_kg': 3000, 'depot_location': None, 'operating_hours': (8, 17)},
    {'id': 'TRUCK_2', 'driver': 'Willy dela Cruz', 'capacity_kg': 3000, 'depot_location': None, 'operating_hours': (8, 17)},
    {'id': 'TRUCK_3', 'driver': 'Edwin Benedicto', 'capacity_kg': 3000, 'depot_location': None, 'operating_hours': (8, 17)},
    {'id': 'TRUCK_4', 'driver': 'Antonio Atienza', 'capacity_kg': 3000, 'depot_location': None, 'operating_hours': (8, 17)},
    {'id': 'TRUCK_5', 'driver': 'Ricky Benedicto', 'capacity_kg': 3000, 'depot_location': None, 'operating_hours': (8, 17)},
    {'id': 'TRUCK_6', 'driver': 'Michael Muriel', 'capacity_kg': 3000, 'depot_location': None, 'operating_hours': (8, 17)},
]

RAW_ROUTES = [
    {
        'truck': 'TRUCK_1',
        'items': [
            {'type': 'path', 'name': 'Sto Rosario St.', 'start': 'Sto Rosario St. corner Rizal St.', 'end': 'Sto. Domingo border'},
            {'type': 'path', 'name': 'Plaridel St.', 'start': 'Plaridel St. corner Sto. Rosario St.', 'end': 'Plaridel St. corner Miranda St.'},
            {'type': 'path', 'name': 'Miranda St.', 'start': 'Miranda St. corner Sto. Entierro St.', 'end': 'Rotonda San Nicolas'}
        ]
    },
    {
        'truck': 'TRUCK_2',
        'items': [
            {'type': 'path', 'name': 'New Abacan Bypass Road', 'start': 'Jaoville', 'end': 'Pulung Maragul', 'both_sides': True},
            {'type': 'path', 'name': 'Arayat Blvd.', 'start': 'Arayat Blvd. corner San Francisco St.', 'end': 'Arayat Blvd. corner Fil-Am Road', 'is_main': True},
            {'type': 'path', 'name': 'Pampang Road', 'start': 'Pampang Road corner Arayat Blvd.', 'end': 'Pampang Road corner Kalayaan St.'},
            {'type': 'path', 'name': 'San Francisco St.', 'start': 'San Francisco St. corner Visitacion St.', 'end': 'San Francisco St. corner Arayat Blvd.'}
        ]
    },
    {
        'truck': 'TRUCK_3',
        'items': [
            {'type': 'path', 'name': 'Angeles Magalang Road', 'start': 'Diego\'s', 'end': 'Cutud', 'is_main': True},
            {'type': 'street', 'name': 'Sto Entierro'},
            {'type': 'street', 'name': 'Lakandula St.'},
            {'type': 'path', 'name': 'Miranda St.', 'start': 'Miranda St. corner Lakandula', 'end': 'Miranda St. corner Sto. Entierro St.'},
            {'type': 'hospital', 'name': 'Hospital Ning Ona'},
            {'type': 'landmark', 'name': '174th Precinct'}
        ]
    },
    {
        'truck': 'TRUCK_4',
        'items': [
            {'type': 'street', 'name': 'Jake Gonzales St.'},
            {'type': 'street', 'name': 'Henson St.'},
            {'type': 'street', 'name': 'Rizal Ext.'},
            {'type': 'landmark', 'name': 'Cut-cut cemetery'},
            {'type': 'street', 'name': 'G. Valdez St.'},
            {'type': 'school', 'name': 'Angeles Science High School', 'subtype': 'high school'},
            {'type': 'school', 'name': 'Dr. Clemente Elem. School', 'subtype': 'elementary'},
            {'type': 'landmark', 'name': 'Villa Ngelina'}
        ]
    },
    {
        'truck': 'TRUCK_5',
        'items': [
            {'type': 'path', 'name': 'Mac-Arthur Hi-way', 'start': 'Sto. Domingo border', 'end': 'Marisol Rotonda', 'is_main': True},
            {'type': 'street', 'name': 'Kalayaan St.'}
        ]
    },
    {
        'truck': 'TRUCK_6',
        'items': [
            {'type': 'school', 'name': 'City High Pampang', 'subtype': 'high school', 'days': ['Mon', 'Wed', 'Fri']},
            {'type': 'school', 'name': 'Gueco Elem.', 'subtype': 'elementary', 'days': ['Tue', 'Thu']},
            {'type': 'school', 'name': 'Rafael Main', 'subtype': 'high school', 'days': ['Mon', 'Wed', 'Fri']},
            {'type': 'school', 'name': 'City College', 'subtype': 'college', 'days': ['Mon', 'Wed', 'Sat']},
            {'type': 'school', 'name': 'Trade School', 'subtype': 'high school', 'days': ['Tue', 'Thu', 'Sat']},
            {'type': 'landmark', 'name': 'City Hygene', 'days': ['Mon', 'Thu']},
        ]
    }
]

# --- SETUP AND HELPERS ---
geocoder = Nominatim(user_agent="AL1_garbage_optimizer_angeles")
failed_geocodes = []
point_counter = 1

def is_within_bounds(lat: float, lon: float) -> bool:
    return (BOUNDS['south'] <= lat <= BOUNDS['north']) and (BOUNDS['west'] <= lon <= BOUNDS['east'])

def geocode_loc(name: str) -> Optional[Tuple[float, float]]:
    query = f"{name}, Angeles City, Pampanga, Philippines"
    try:
        loc = geocoder.geocode(query, timeout=10)
        if loc:
            if is_within_bounds(loc.latitude, loc.longitude):
                return loc.latitude, loc.longitude
            else:
                logging.warning(f"Out of bounds: {name} at {loc.latitude}, {loc.longitude}")
                failed_geocodes.append({'query': query, 'name': name, 'reason': 'Out of bounds'})
        else:
            logging.warning(f"Failed to geocode: {name}")
            failed_geocodes.append({'query': query, 'name': name, 'reason': 'Not found'})
    except Exception as e:
        logging.error(f"Geocode error {name}: {e}")
        failed_geocodes.append({'query': query, 'name': name, 'reason': str(e)})
    return None

def get_waste_kg(item: Dict[str, Any]) -> int:
    itype = item.get('type')
    if itype == 'school':
        return 300 if item.get('subtype') in ['high school', 'college'] else 200
    elif itype == 'hospital':
        return 400
    elif itype == 'landmark':
        name_lower = item.get('name', '').lower()
        if 'rotonda' in name_lower or 'intersection' in name_lower:
            return 60
        return 100
    return 50 # Default for others / fallback

def get_street_waste_kg(item: Dict[str, Any], edge_data: Dict[str, Any]=None) -> int:
    if item.get('is_main'):
        return 80
    if edge_data and 'highway' in edge_data:
        h = edge_data['highway']
        if isinstance(h, list): h = h[0]
        if h in ['primary', 'secondary', 'trunk', 'tertiary']:
            return 80
    return 50

# --- MAIN GENERATOR FUNCTION ---
def run_generator():
    global point_counter
    logging.info("Downloading OSMnx Graph for Angeles City constraints...")
    # Get network within bounding box
    try:
        G = ox.graph_from_bbox(
            bbox=(BOUNDS['north'], BOUNDS['south'], BOUNDS['east'], BOUNDS['west']), 
            network_type='drive'
        )
        # Project to meters for distance calculations
        G_proj = ox.project_graph(G)
        project_to_wgs84 = pyproj.Transformer.from_crs(G_proj.graph['crs'], 'epsg:4326', always_xy=True).transform
    except Exception as e:
        logging.error(f"Failed to download or project OSM graph: {e}")
        return

    PICKUP_POINTS = []

    for truck_info in RAW_ROUTES:
        truck_id = truck_info['truck']
        logging.info(f"Processing routes for {truck_id}")
        
        for item in truck_info['items']:
            item_type = item['type']
            name = item['name']
            days = item.get('days', ['All'])
            notes = f"Original route: {truck_id}"
            
            # Point/Single Locations
            if item_type in ['school', 'hospital', 'landmark']:
                coords = geocode_loc(name)
                if coords:
                    kg = get_waste_kg(item)
                    PICKUP_POINTS.append({
                        'id': f'P{point_counter:03d}',
                        'name': name,
                        'type': item_type,
                        'lat': coords[0],
                        'lon': coords[1],
                        'waste_kg': kg,
                        'service_days': days,
                        'time_window': (None, None),
                        'notes': notes
                    })
                    point_counter += 1
            
            # Sub-segment / Shortest Path between two endpoints
            elif item_type == 'path':
                start_coords = geocode_loc(item['start'])
                end_coords = geocode_loc(item['end'])
                
                if not start_coords or not end_coords:
                    logging.warning(f"Could not geocode path endpoints for {name}. Skipping sub-route.")
                    continue
                
                # Find nearest nodes
                # project coords to proj_crs
                to_proj = pyproj.Transformer.from_crs('epsg:4326', G_proj.graph['crs'], always_xy=True).transform
                sx, sy = to_proj(start_coords[1], start_coords[0])
                ex, ey = to_proj(end_coords[1], end_coords[0])
                
                orig = ox.distance.nearest_nodes(G_proj, X=sx, Y=sy)
                dest = ox.distance.nearest_nodes(G_proj, X=ex, Y=ey)
                
                try:
                    path = nx.shortest_path(G_proj, orig, dest, weight='length')
                except nx.NetworkXNoPath:
                    logging.warning(f"No path found in OSMnx for {name} between {item['start']} and {item['end']}.")
                    continue
                
                # Iterate path edges and extract linestrings
                segment_id = 1
                for u, v in zip(path[:-1], path[1:]):
                    edge_data = min(G_proj.get_edge_data(u, v).values(), key=lambda d: d.get('length', 1))
                    geom = edge_data.get('geometry')
                    if not geom:
                        # Fallback to straight line between nodes
                        u_node = G_proj.nodes[u]
                        v_node = G_proj.nodes[v]
                        geom = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
                    
                    kg_per_pt = get_street_waste_kg(item, edge_data)
                    length = geom.length
                    num_points = max(1, int(length // 150))
                    
                    for i in range(num_points):
                        dist = 75 + (i * 150) # Start halfway
                        if dist > length: dist = length / 2
                        pt = geom.interpolate(dist)
                        lon, lat = project_to_wgs84(pt.x, pt.y)
                        
                        multiplier = 2 if item.get('both_sides') else 1
                        
                        PICKUP_POINTS.append({
                            'id': f'P{point_counter:03d}',
                            'name': f"{name} Segment {segment_id}",
                            'type': 'street_segment',
                            'lat': lat,
                            'lon': lon,
                            'waste_kg': kg_per_pt * multiplier,
                            'service_days': days,
                            'time_window': (None, None),
                            'notes': notes
                        })
                        point_counter += 1
                        segment_id += 1

            # Full street (find all elements with name)
            elif item_type == 'street':
                edges = []
                for u, v, k, d in G_proj.edges(keys=True, data=True):
                    if 'name' in d:
                        names = d['name'] if isinstance(d['name'], list) else [d['name']]
                        if any(name.lower() in n.lower() for n in names):
                            edges.append(d)
                
                if not edges:
                    logging.warning(f"Could not find any edges for street: {name}")
                    failed_geocodes.append({'query': f"OSM name tag: {name}", 'name': name, 'reason': 'No edges found'})
                    continue
                
                segment_id = 1
                for d in edges:
                    geom = d.get('geometry')
                    if not geom:
                        # we would need u and v to construct, but we didn't save them. Simplified skip
                        continue
                        
                    kg_per_pt = get_street_waste_kg(item, d)
                    length = geom.length
                    if length < 5: continue # skip microscopic edges
                    
                    num_points = max(1, int(length // 150))
                    
                    for i in range(num_points):
                        dist = 75 + (i * 150)
                        if dist > length: dist = length / 2
                        pt = geom.interpolate(dist)
                        lon, lat = project_to_wgs84(pt.x, pt.y)
                        
                        PICKUP_POINTS.append({
                            'id': f'P{point_counter:03d}',
                            'name': f"{name} Segment {segment_id}",
                            'type': 'street_segment',
                            'lat': lat,
                            'lon': lon,
                            'waste_kg': kg_per_pt,
                            'service_days': days,
                            'time_window': (None, None),
                            'notes': notes
                        })
                        point_counter += 1
                        segment_id += 1

    logging.info("Generation complete.")
    
    # Validation check
    total_waste = 0
    valid_points = []
    for p in PICKUP_POINTS:
        if is_within_bounds(p['lat'], p['lon']):
            valid_points.append(p)
            total_waste += p['waste_kg']
        else:
            logging.error(f"Generated point out of bounds: {p['name']} ({p['lat']}, {p['lon']})")
    
    # Save Data locally
    with open('angeles_city_routes_data.py', 'w') as f:
        f.write("# Auto-generated Route Data\n")
        f.write("TRUCK_FLEET = " + pprint.pformat(TRUCK_FLEET) + "\n\n")
        f.write("PICKUP_POINTS = " + pprint.pformat(valid_points, sort_dicts=False) + "\n")
        
    # Save Failed log
    if failed_geocodes:
        pd.DataFrame(failed_geocodes).to_csv('failed_geocoding.csv', index=False)
        
    # Folium map
    m = folium.Map(location=[15.15, 120.60], zoom_start=13)
    truck_colors = {'TRUCK_1': 'red', 'TRUCK_2': 'blue', 'TRUCK_3': 'green', 
                    'TRUCK_4': 'purple', 'TRUCK_5': 'orange', 'TRUCK_6': 'darkred'}
    for p in valid_points:
        truck_id = p['notes'].split(': ')[1]
        folium.CircleMarker(
            location=[p['lat'], p['lon']],
            radius=4,
            popup=f"{p['name']}<br>Waste: {p['waste_kg']}kg",
            color=truck_colors.get(truck_id, 'black'),
            fill=True
        ).add_to(m)
    m.save('angeles_city_coverage.html')
    
    # Summary report
    print("\n" + "="*40)
    print("SUMMARY REPORT")
    print("="*40)
    print(f"Total Points Created: {len(valid_points)}")
    print(f"Total Daily Waste (kg): {total_waste:,}")
    print(f"Total Failed Locations: {len(failed_geocodes)}")
    print("Files Generated:")
    print("  - angeles_city_routes_data.py (Python data structure)")
    print("  - failed_geocoding.csv (Locations requiring manual review)")
    print("  - angeles_city_coverage.html (Folium map visualization)")
    print("="*40)


if __name__ == "__main__":
    run_generator()
