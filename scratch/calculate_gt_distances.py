
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.data import GOOGLE_ROUTES
from src.utils import haversine

def calculate_route_total(coords):
    total = 0.0
    for i in range(len(coords) - 1):
        total += haversine(coords[i], coords[i+1])
    return round(total, 2)

if __name__ == "__main__":
    print("Calculating Ground-Truth Route Distances (Trucks 1-5):")
    calculated_dists = []
    for i, route in enumerate(GOOGLE_ROUTES):
        dist = calculate_route_total(route)
        calculated_dists.append(dist)
        print(f"Truck {i+1}: {dist} km")
    
    print("\nProposed Update for app.py:")
    print(f"core_dists = {calculated_dists}")
