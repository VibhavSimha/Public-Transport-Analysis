# overpass_fetcher.py
import requests, time
from bbox_helper import get_bengaluru_bbox

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def _overpass_query(ql: str, max_tries=3):
    for attempt in range(1, max_tries+1):
        resp = requests.post(OVERPASS_URL, data={"data": ql})
        text = resp.text.lstrip()
        if resp.status_code == 200 and not text.startswith('<'):
            try:
                return resp.json()
            except ValueError:
                print(f"[Attempt {attempt}] JSON parse failed. Snippet:\n{text[:300]}")
        else:
            print(f"[Attempt {attempt}] HTTP {resp.status_code} or HTML. Snippet:\n{text[:300]}")
        time.sleep(2**attempt)
    raise RuntimeError("Overpass query failed.")

def fetch_stops(snapshot_date: str):
    south, west, north, east = get_bengaluru_bbox()
    ql = f"""
    [out:json][timeout:180];
    (
      node({south},{west},{north},{east})["highway"="bus_stop"];
      node({south},{west},{north},{east})["public_transport"="stop_position"];
      node({south},{west},{north},{east})["public_transport"="platform"];
      node({south},{west},{north},{east})["amenity"="bus_station"];
      node({south},{west},{north},{east})["railway"="station"];
    );
    out tags center;
    """
    return _overpass_query(ql)

def fetch_routes(snapshot_date: str, route_type: str):
    south, west, north, east = get_bengaluru_bbox()
    ql = f"""
    [out:json][timeout:180];
    relation({south},{west},{north},{east})["type"="route"]["route"="{route_type}"];
    way(r);
    out geom;
    """
    return _overpass_query(ql)

def fetch_osm_snapshot(snapshot_date: str, route_type: str):
    stops = fetch_stops(snapshot_date)
    print(f"  • stops → {len(stops.get('elements', []))} elements")
    routes = fetch_routes(snapshot_date, route_type)
    print(f"  • routes → {len(routes.get('elements', []))} elements")
    return {"elements": stops.get("elements", []) + routes.get("elements", [])}
