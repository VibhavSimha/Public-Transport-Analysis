# graph_builder.py
import networkx as nx
from shapely.geometry import LineString

def build_graph_from_gdf(nodes_gdf, ways_gdf, mode):
    G = nx.MultiDiGraph(mode=mode)
    
    for _, row in nodes_gdf.iterrows():
        G.add_node(row['id'], pos=(row.geometry.x, row.geometry.y), mode=mode)
    
    for _, row in ways_gdf.iterrows():
        coords = list(row.geometry.coords)
        for i in range(len(coords)-1):
            u = coords[i]
            v = coords[i+1]
            dist = LineString([u, v]).length
            G.add_edge(u, v, weight=dist, mode=mode)
    
    return G
