# bbox_helper.py
import osmnx as ox

def get_bengaluru_bbox():
    """
    Returns (south, west, north, east) in latitudes/longitudes
    suitable for Overpass.
    """
    # minx = lon_min, miny = lat_min, maxx = lon_max, maxy = lat_max
    minx, miny, maxx, maxy = ox.geocode_to_gdf("Bengaluru, Karnataka, India").unary_union.bounds
    south, west, north, east = miny, minx, maxy, maxx
    return south, west, north, east
