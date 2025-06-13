import requests
import time
import geopandas as gpd
from shapely.geometry import Point, LineString
import json
import os

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MODES = {
    'bus': ('public_transport', 'platform', 'route', 'bus'),
    'metro': ('railway', 'station', 'route', 'subway'),
    'rail': ('railway', 'station', 'route', 'train'),
}
SNAPSHOTS = [f"{year}-01-01T00:00:00Z" for year in range(2015, 2025)]
OUTPUT_DIR = os.path.join('data', 'osm_snapshots')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_query(snapshot_date: str, key_node: str, val_node: str, key_route: str, val_route: str):
    return f"""
[date:"{snapshot_date}"][timeout:120];
area["name"="Bengaluru"]["boundary"="administrative"]->.city;
(
  node(area.city)["{key_node}"="{val_node}"];
);
out tags center;
(
  rel(area.city)["{key_route}"="{val_route}"];
  way(r);
);
out geom;
"""


def fetch_and_save(mode: str, tags: tuple):
    key_node, val_node, key_route, val_route = tags
    for date in SNAPSHOTS:
        print(f"Fetching {mode} snapshot {date}...")
        ql = build_query(date, key_node, val_node, key_route, val_route)
        resp = requests.post(OVERPASS_URL, data={'data': ql})
        resp.raise_for_status()
        data = resp.json()
        # convert to GeoDataFrame
        nodes = []
        ways = []
        for el in data['elements']:
            if el['type']=='node':
                nodes.append({
                    'id': el['id'], 'mode': mode,
                    **el.get('tags', {}),
                    'geometry': Point(el['center']['lon'], el['center']['lat'])
                })
            elif el['type']=='way':
                coords = [(c['lon'], c['lat']) for c in el['geometry']]
                ways.append({
                    'id': el['id'], 'mode': mode,
                    **el.get('tags', {}),
                    'geometry': LineString(coords)
                })
        gdf = gpd.GeoDataFrame(nodes + ways, crs='EPSG:4326')
        outpath = os.path.join(OUTPUT_DIR, f"{date[:4]}_{mode}.geojson")
        gdf.to_file(outpath, driver='GeoJSON')
        time.sleep(5)

if __name__ == '__main__':
    for mode, tags in MODES.items():
        fetch_and_save(mode, tags)