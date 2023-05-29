import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two points on the earth (specified in decimal degrees)

    Parameters:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: Haversine distance in kilometers.
    """
    R = 6371  # Earth's radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return R * c

def get_catchment_area_cities(routes: pd.DataFrame, warehouse_lat: float, warehouse_lon: float, distance_threshold: float) -> set:
    """
    Get the catchment area cities for a warehouse.

    Args:
        routes (pd.DataFrame): DataFrame containing the routes with columns ['Start Coordinates', 'Destination City'].
        warehouse_lat (float): Latitude of the warehouse.
        warehouse_lon (float): Longitude of the warehouse.
        distance_threshold (float): Distance threshold for the catchment area in kilometers.

    Returns:
        set: A set of cities within the catchment area.
    """
    catchment_area_cities = set()

    for index, row in routes.iterrows():
        destination_lat, destination_lon = map(float, row['End Coordinates'].split(','))  # Split coordinates and convert them to float
        destination_city = row['Destination City']
        destination_state = row['Destination State']

        if haversine_distance(warehouse_lat, warehouse_lon, destination_lat, destination_lon) <= distance_threshold:
            catchment_area_cities.add((destination_city, destination_state))  # Save as tuple

    return catchment_area_cities