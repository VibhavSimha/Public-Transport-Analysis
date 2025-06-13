import folium
from shapely.geometry import LineString, Point

# 1. Base map centered on Bengaluru
m = folium.Map(location=[12.97, 77.59], zoom_start=11)

# 2. Add edges
for u, v, data in G.edges(data=True):
    if 'pos' in G.nodes[u] and 'pos' in G.nodes[v]:
        lat1, lon1 = G.nodes[u]['pos'][1], G.nodes[u]['pos'][0]
        lat2, lon2 = G.nodes[v]['pos'][1], G.nodes[v]['pos'][0]
        folium.PolyLine(locations=[(lat1, lon1), (lat2, lon2)],
                        color='gray', weight=1, opacity=0.4).add_to(m)

# 3. Add nodes
for n, data in G.nodes(data=True):
    if 'pos' in data:
        lat, lon = data['pos'][1], data['pos'][0]
        folium.CircleMarker((lat, lon), radius=1, color='black', fill=True).add_to(m)

# 4. Display
m  # in Jupyter, this will render the interactive map
