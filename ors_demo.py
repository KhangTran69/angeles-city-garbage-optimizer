import os
import requests
import openrouteservice
from dotenv import load_dotenv

# 1. Setup: Load the API key from the .env file
load_dotenv()
api_key = os.getenv("ORS_API_KEY")

if not api_key:
    print("Error: ORS_API_KEY not found. Check your .env file.")
    exit()

# Example coordinates (Longitude, Latitude) for Angeles City
# Note: ORS expects [Longitude, Latitude] format
coords = [
    [120.5887, 15.1450], # Depot
    [120.5881, 15.1433], # Agapito del Rosario
    [120.5709, 15.1556], # Amsic
    [120.553, 15.1551],  # Anunas
    [120.5888, 15.1701], # Balibago
    [120.612, 15.163],   # Capaya
    [120.598, 15.153],   # Claro M. Recto
    [120.542, 15.161],   # Cuayan
    [120.575, 15.138],   # Cutcut
    [120.619, 15.158],   # Cutud
    [120.585, 15.145],   # Lourdes North West
    [120.59, 15.137],    # Lourdes Sur (Talimundoc)
    [120.595, 15.14],    # Lourdes Sur East
    [120.575, 15.171],   # Malabañas
    [120.521, 15.152],   # Margot
    [120.596, 15.157],   # Marisol (Ninoy Aquino)
    [120.6, 15.15],      # Mining
    [120.5743, 15.1482], # Pampang (Santo Niño)
    [120.604, 15.166],   # Pandan
    [120.6053, 15.1286], # Pulung Bulu
    [120.615, 15.175],   # Pulung Cacutud
    [120.602, 15.174],   # Pulung Maragul
    [120.593, 15.147],   # Salapungan
    [120.584, 15.136],   # San José
    [120.593, 15.135],   # San Nicolas
    [120.584, 15.151],   # Santa Teresita
    [120.587, 15.137],   # Santa Trinidad
    [120.598, 15.146],   # Santo Cristo
    [120.585, 15.133],   # Santo Domingo
    [120.589, 15.135],   # Santo Rosario (Población)
    [120.628, 15.166],   # Sapalibutad
    [120.505, 15.161],   # Sapangbato
    [120.608, 15.174],   # Tabun
    [120.588, 15.14]     # Virgen Delos Remedios
]

def get_directions_with_library():
    """Version 1: Using the official openrouteservice library."""
    print("\n--- Fetching via Official Library ---")
    client = openrouteservice.Client(key=api_key)
    
    try:
        # Using 'driving-hgv' for truck-specific routing
        routes = client.directions(
            coordinates=coords,
            profile='driving-hgv',
            format='geojson'
        )
        
        distance_km = routes['features'][0]['properties']['summary']['distance'] / 1000
        duration_min = routes['features'][0]['properties']['summary']['duration'] / 60
        
        print(f"Distance: {distance_km:.2f} km")
        print(f"Duration: {duration_min:.1f} mins")
        return routes
    except Exception as e:
        print(f"Library Error: {e}")

def get_directions_with_requests():
    """Version 2: Using standard requests (no extra dependency)."""
    print("\n--- Fetching via Requests ---")
    url = "https://api.openrouteservice.org/v2/directions/driving-hgv/geojson"
    
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': api_key,
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    body = {"coordinates": coords}
    
    try:
        response = requests.post(url, json=body, headers=headers)
        response.raise_for_status() # Check for HTTP errors
        
        data = response.json()
        summary = data['features'][0]['properties']['summary']
        
        print(f"Distance: {summary['distance']/1000:.2f} km")
        print(f"Duration: {summary['duration']/60:.1f} mins")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")

if __name__ == "__main__":
    # Security Warning:
    # Never print your 'api_key' to the console or logs in a production environment.
    
    get_directions_with_library()
    get_directions_with_requests()

    print("\nBest Practices Applied:")
    print("1. Keys are loaded from environment variables.")
    print("2. .env is added to .gitignore to prevent accidental commits.")
    print("3. Truck-specific 'driving-hgv' profile is utilized.")
    print("4. Error handling is included for network or API failures.")