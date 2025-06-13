# osm_to_gdf.py
import geopandas as gpd
from shapely.geometry import Point, LineString

def osm_json_to_gdf(osm_data):
    elems = osm_data.get("elements", [])
    if not elems:
        # empty GeoDataFrames with the right schema
        nodes = gpd.GeoDataFrame(columns=["id","tags","geometry"], geometry="geometry", crs="EPSG:4326")
        ways  = gpd.GeoDataFrame(columns=["id","tags","geometry"], geometry="geometry", crs="EPSG:4326")
        return nodes, ways

    nodes, ways = [], []
    for el in elems:
        if el["type"]=="node" and "lat" in el:
            nodes.append({
                "id": el["id"],
                "tags": el.get("tags", {}),
                "geometry": Point(el["lon"], el["lat"])
            })
        elif el["type"]=="way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            ways.append({
                "id": el["id"],
                "tags": el.get("tags", {}),
                "geometry": LineString(coords)
            })
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs="EPSG:4326")
    ways_gdf  = gpd.GeoDataFrame(ways,  geometry="geometry", crs="EPSG:4326")
    return nodes_gdf, ways_gdf
