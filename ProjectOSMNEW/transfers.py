# transfers.py
import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great‐circle distance (in meters) between two points.
    """
    R = 6371000  # Earth radius in m
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def add_transfer_links(G_all, max_distance=150):
    """
    For each pair of mode-graphs in G_all (dict mode→Graph),
    link nodes that both have 'pos' attributes and are within max_distance (meters).
    Returns a list of (u, v, attrs) edges.
    """
    transfer_edges = []
    modes = list(G_all.keys())
    for i, m1 in enumerate(modes):
        for m2 in modes[i+1:]:
            # gather only real stops
            nodes1 = [(u, data['pos']) for u, data in G_all[m1].nodes(data=True) if 'pos' in data]
            nodes2 = [(v, data['pos']) for v, data in G_all[m2].nodes(data=True) if 'pos' in data]

            for u, (lon1, lat1) in nodes1:
                for v, (lon2, lat2) in nodes2:
                    dist = haversine(lat1, lon1, lat2, lon2)
                    if dist <= max_distance:
                        # walking speed ~1.3 m/s → time = dist/1.3
                        walking_time = dist / 1.3
                        transfer_edges.append((u, v, {'mode': 'transfer', 'weight': walking_time}))
    return transfer_edges
