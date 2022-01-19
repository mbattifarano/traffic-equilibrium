import osmnx as ox
from shapely.geometry import Polygon


def geofence_from_place(place: str, buffer: float = 0.0) -> Polygon:
    return ox.geocode_to_gdf(place, buffer_dist=buffer).geometry.iloc[0]

