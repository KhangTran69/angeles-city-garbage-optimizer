import re
import requests

links = [
    "https://maps.app.goo.gl/FbpGKc8CuiXyPu1s7",
    "https://maps.app.goo.gl/k2fW283rMsxpSDRj7",
    "https://maps.app.goo.gl/yiEhn9hu5NtdC3R99",
    "https://maps.app.goo.gl/QtQewNvKGwKk7MbW7",
    "https://maps.app.goo.gl/EeBwi6efmy3bLhrw6"
]

all_coords = []
for i, link in enumerate(links):
    r = requests.get(link)
    url = r.url
    # find all !1d... !2d... sequences
    matches = re.findall(r'!1d([0-9\.\-]+)!2d([0-9\.\-]+)', url)
    # The URL may have dupes, just list them
    parsed = [(float(lat), float(lon)) for lon, lat in matches]
    all_coords.append(parsed)
    print(f"Truck {i+1} Waypoints ({len(parsed)}): {parsed}")

