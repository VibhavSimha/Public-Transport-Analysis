# main_pipeline.py
from dates import SNAPSHOT_DATES
from overpass_fetcher import fetch_osm_snapshot
from osm_to_gdf import osm_json_to_gdf
from graph_builder import build_graph_from_gdf
from transfers import add_transfer_links
from bbox_helper import get_bengaluru_bbox
print("Bengaluru bbox:", get_bengaluru_bbox())

import networkx as nx
import os
import pickle
from tqdm import tqdm

modes = {
    "bus": "bus",
    "metro": "subway",
    "rail": "train"
}

OUTPUT_DIR = "graphs_snapshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for snapshot in SNAPSHOT_DATES:
    full_graph = nx.MultiDiGraph(snapshot=snapshot)
    sub_graphs = {}
    
    for mode, osm_tag in modes.items():
        print(f"\nFetching {mode} at {snapshot}…")
        json_data = fetch_osm_snapshot(snapshot, osm_tag)
        # now you’ll see exactly how many stops vs routes were returned
        nodes_gdf, ways_gdf = osm_json_to_gdf(json_data)
        print(f"  → GeoDataFrames: {len(nodes_gdf)} nodes, {len(ways_gdf)} ways")
        G = build_graph_from_gdf(nodes_gdf, ways_gdf, mode)
        sub_graphs[mode] = G
        full_graph = nx.compose(full_graph, G)

        
    
    print("Adding transfer edges...")
    transfers = add_transfer_links(sub_graphs)
    full_graph.add_edges_from(transfers)
    
    with open(f"{OUTPUT_DIR}/graph_{snapshot[:4]}.pkl", "wb") as f:
        pickle.dump(full_graph, f)

    print(f"Saved snapshot {snapshot}")
