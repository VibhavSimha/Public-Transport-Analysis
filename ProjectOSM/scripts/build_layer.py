import geopandas as gpd
import networkx as nx
import sys, os
from shapely.geometry import LineString

INPUT_DIR = os.path.join('data', 'osm_snapshots')
OUTPUT_DIR = os.path.join('data', 'graphs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_mode_graph(year: str, mode: str):
    path = os.path.join(INPUT_DIR, f"{year}_{mode}.geojson")
    gdf = gpd.read_file(path)
    G = nx.MultiDiGraph()
    # Add point nodes
    pts = gdf[gdf.geom_type=='Point']
    for _, row in pts.iterrows():
        G.add_node(row['id'], mode=mode, x=row.geometry.x, y=row.geometry.y)
    # Add edges from LineString geometries
    lines = gdf[gdf.geom_type=='LineString']
    for _, row in lines.iterrows():
        coords = list(row.geometry.coords)
        for u, v in zip(coords, coords[1:]):
            G.add_edge(u, v, mode=mode, weight=LineString([u, v]).length)
    # save graphml
    outpath = os.path.join(OUTPUT_DIR, f"{mode}_{year}.graphml")
    nx.write_graphml(G, outpath)
    print(f"Saved {mode} graph for {year} to {outpath}")

if __name__ == '__main__':
    year, mode = sys.argv[1], sys.argv[2]
    build_mode_graph(year, mode)