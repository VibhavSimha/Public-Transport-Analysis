import os
import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
from sklearn.linear_model import LogisticRegression
import folium

def build_road_graph(path):
    roads = gpd.read_file(path)
    G = nx.Graph()
    for r in roads.itertuples():
        geom = getattr(r, 'geometry', None)
        if geom and geom.geom_type == 'LineString':
            prev = None
            for lon, lat in geom.coords:
                node = (round(lat, 6), round(lon, 6))
                G.add_node(node)
                if prev:
                    dist = geodesic(prev, node).meters
                    G.add_edge(prev, node, weight=dist)
                prev = node
    return G

def load_historical(path, years):
    snaps = {}
    for y in years:
        f = os.path.join(path, f"data_{y}.geojson")
        coords = []
        if os.path.exists(f):
            gdf = gpd.read_file(f)
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == 'Point':
                    # Always include if it's Namma Metro or subway station
                    if row.get('network') == 'Namma Metro' or row.get('station') == 'subway':
                        coords.append((geom.y, geom.x))
        snaps[y] = coords
    return snaps

def make_tree(nodes):
    arr = np.array(nodes)
    tree = BallTree(np.deg2rad(arr), metric='haversine')
    return tree, arr

def build_time_series(G, snaps, years):
    nodes = list(G.nodes)
    idx = pd.MultiIndex.from_product([nodes, years], names=('node','year'))
    ts = pd.DataFrame(0, index=idx, columns=['station'])
    tree, arr = make_tree(nodes)
    for y in years:
        coords = snaps[y]
        if coords:
            _, ix = tree.query(np.deg2rad(np.array(coords)), k=1)
            for i in ix.flatten():
                ts.at[(tuple(arr[i]), y), 'station'] = 1
    return ts

def train_model(ts, years):
    rows = []
    for node in ts.index.levels[0]:
        s = ts.xs(node, level=0)['station']
        cum = 0
        for y1, y2 in zip(years, years[1:]):
            cum += s.get(y1, 0)
            rows.append({'past': cum, 'year': y1, 'target': s.get(y2, 0)})
    df = pd.DataFrame(rows)
    if df['target'].nunique() < 2:
        return None
    X = df[['past','year']].values
    y = df['target'].values
    return LogisticRegression(solver='liblinear').fit(X, y)

def predict_lines(G, model, ts, year, start_year, count, min_dist, global_nodes):
    cand = []
    for node in G.nodes:
        if node in global_nodes:
            continue
        past = ts.xs(node, level=0)['station'].loc[start_year:year-1].sum()
        if past > 0:
            continue
        p = model.predict_proba([[past, year]])[0,1] if model else 0
        cand.append((node, p))
    cand.sort(key=lambda x: -x[1])
    sel = []
    rads = [np.deg2rad(n) for n in global_nodes]
    tree = BallTree(np.array(rads), metric='haversine') if rads else None
    for node, p in cand:
        rad = np.deg2rad(node)
        if tree:
            d = tree.query([rad], k=1)[0][0][0] * 6371000
            if d < min_dist:
                continue
        sel.append((node, p))
        rads.append(rad)
        tree = BallTree(np.array(rads), metric='haversine')
        if len(sel) >= count:
            break
    lines = []
    prev = global_nodes[-1] if global_nodes else None
    for node, _ in sel:
        if prev and nx.has_path(G, prev, node):
            lines.append(nx.shortest_path(G, prev, node, weight='weight'))
            prev = node
        else:
            lines.append([])
    return sel, lines

def visualize(G, ts, preds, lines, history, future, out):
    m = folium.Map(location=[12.9716,77.5946],zoom_start=10)
    gdf = gpd.read_file(f"data_2025.geojson")
    hgroup = folium.FeatureGroup(name='Historic', show=True)
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type=='Point' and (row.get('network')=='Namma Metro' or row.get('station')=='subway'):
            folium.CircleMarker([geom.y,geom.x],radius=4,color='blue',fill=True).add_to(hgroup)
    m.add_child(hgroup)
    colors = ['red','green','purple','orange']
    for i, (sel, line) in enumerate(zip(preds, lines)):
        layer = folium.FeatureGroup(name=str(future[i]), show=(i==0))
        for node, _ in sel:
            folium.CircleMarker(location=node, radius=5, color=colors[i%4], fill=True).add_to(layer)
        for path in line:
            if path:
                folium.PolyLine(locations=path, color=colors[i%4]).add_to(layer)
        m.add_child(layer)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out)

def main():
    history = list(range(2017, 2026))
    future = list(range(2026, 2036))
    G = build_road_graph('road_network.geojson')
    snaps = load_historical('.', history + [2025])
    ts = build_time_series(G, snaps, history)
    model = train_model(ts, history)
    preds, lines = [], []
    global_nodes = [n for n,_ in ts.xs(2025, level=1)['station'].items() if _==1]
    total_needed = max(1, int(len(G.nodes) * 0.02))
    per_year = total_needed // len(future)
    remaining = total_needed % len(future)
    for i, y in enumerate(future):
        count = per_year + (1 if i < remaining else 0)
        sel, ln = predict_lines(G, model, ts, y, history[0], count, 1000, global_nodes)
        preds.append(sel)
        lines.append(ln)
        global_nodes.extend([node for node,_ in sel])
        idx = pd.MultiIndex.from_product([[n for n,_ in sel], [y]], names=('node','year'))
        ts = pd.concat([ts, pd.DataFrame({'station':1}, index=idx)])
        history.append(y)
        model = train_model(ts, history) or model  # Keep previous model if training fails
    visualize(G, ts, preds, lines, history, future, 'predicted_map.html')

if __name__=='__main__':
    main()