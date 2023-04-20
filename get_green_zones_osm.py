"""This module is used to get objects that we consider green from OSM"""
from typing import Tuple

import overpass
import geopandas as gpd
import pandas as pd


def get_green_objects(bbox: Tuple[float, float, float, float]) -> list[dict]:
    """
    Define the query to retrieve all parks and forests within the bounding box.
    """
    # TODO: make list of green zone types an argument for this function
    query = """
    (
      way["leisure"="park"]({0});
      way["landuse"="forest"]({0});
    );
    out center;
    """.format(",".join(str(x) for x in bbox))

    # execute the query using the overpass API
    api = overpass.API()
    response = api.Get(query)

    # print the names of all parks and forests returned by the query
    objects_ids = []
    for feature in response['features']:
        if 'id' in feature:
            objects_ids.append(feature['id'])
    return objects_ids


def get_object_osm(object_id: int) -> gpd.GeoDataFrame:
    """Get geometry of object by its ID from OSM."""
    api = overpass.API()
    query = f"""
        way({object_id});
        (._;>;);
        out body;
    """
    response = api.Get(query)['features']
    df = pd.DataFrame(response)
    df['Longitude'] = df['geometry'].apply(lambda x: x['coordinates'][0] if x['coordinates'] else None)
    df['Latitude'] = df['geometry'].apply(lambda x: x['coordinates'][1] if x['coordinates'] else None)
    df.dropna(inplace=True)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(x=df.Longitude, y=df.Latitude), crs=4326
    )
    return gdf
