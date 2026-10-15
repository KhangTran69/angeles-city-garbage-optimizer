import requests

links = [
    "https://maps.app.goo.gl/FbpGKc8CuiXyPu1s7",
    "https://maps.app.goo.gl/k2fW283rMsxpSDRj7",
    "https://maps.app.goo.gl/yiEhn9hu5NtdC3R99",
    "https://maps.app.goo.gl/QtQewNvKGwKk7MbW7",
    "https://maps.app.goo.gl/EeBwi6efmy3bLhrw6"
]

for i, link in enumerate(links):
    r = requests.get(link)
    print(f"Truck {i+1}: {r.url}")

