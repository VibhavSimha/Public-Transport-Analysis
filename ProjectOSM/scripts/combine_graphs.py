import networkx as nx
import os
import osmnx as ox
import sys

GRAPH_DIR = os.path.join('data', 'graphs')
OUTPUT_DIR = GRAPH_DIR


def load_layer(year, mode):
    path = os.path.join(GRAPH_DIR, f"{mode}_{year}.graphml")
    return nx.read_graphml(path)


def add_transfers(G, G_walk, modes):
    for m1 in modes:
        for m2 in modes:
            if m1>=m2: continue
            G1 = G[m1]; G2 = G[m2]
            for u, udata in G1.nodes(data=True):
                # find nearest walk node
                pt = (float(udata['y']), float(udata['x']))
                nearest = ox.distance.nearest_nodes(G_walk, pt[1], pt[0])
                # link to nodes of G2 within 100m
                for v, vdata in G2.nodes(data=True):
                    dist = ox.distance.euclidean_dist_vec(
                        float(udata['y']), float(udata['x']),
                        float(vdata['y']), float(vdata['x'])
                    )
                    if dist < 100:
                        G_comb.add_edge(u, v, mode='transfer', weight=120)

if __name__=='__main__':
    year = sys.argv[1]
    modes = ['bus','metro','rail']
    # load layers
    layer_graphs = {m: load_layer(year, m) for m in modes}
    G_comb = nx.compose_all(layer_graphs.values())
    print(f"Loaded and composed layers for {year}")
    # download walk graph once
    G_walk = ox.graph_from_place("Bengaluru, Karnataka, India", network_type='walk')
    add_transfers(layer_graphs, G_walk, modes)
    # save combined
    outpath = os.path.join(OUTPUT_DIR, f"combined_{year}.graphml")
    nx.write_graphml(G_comb, outpath)
    print(f"Saved combined multiplex graph for {year} to {outpath}")