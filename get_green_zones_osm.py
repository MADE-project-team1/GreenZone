"""This module is used to get objects that we consider green from OSM"""
from typing import Tuple

import numpy
import numpy as np
import overpass
import geopandas as gpd
import pandas as pd
import osmnx as ox
from shapely.geometry import box
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    SHConfig,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)
import cv2


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


def get_green_space_areas(north, south, east, west):
    """
    Define the bounding box coordinates and get area of green area inside this bbox and gdf with green zones.
    # Example: north, south, east, west = 43.677542, 43.634644, 39.704246, 39.635672
    :param north:
    :param south:
    :param east:
    :param west:
    :return:
    """
    tags = {"leisure": ["park", "nature_reserve", "playground", "garden"],
            "landuse": ["recreation_ground", "meadow", "forest", "grass"],
            "natural": ["wood"]}

    gdf = ox.geometries_from_bbox(north, south, east, west, tags=tags)
    gdf = gdf[(gdf["geometry"].geom_type == "MultiPolygon") | (gdf["geometry"].geom_type == "Polygon")]
    bbox_polygon = gpd.GeoDataFrame({'geometry': [box(west, south, east, north)]}, crs=gdf.crs)
    gdf = gpd.overlay(bbox_polygon, gdf, how='intersection')
    gdf = gdf.to_crs({'init': 'epsg:3857'})
    green_rate = sum(gdf.geometry.area)

    return green_rate, gdf


def create_santinel_config():
    config = SHConfig()

    config.sh_client_id = '83bdb8ff-bcd4-4d56-a681-61b173acfada'
    config.sh_client_secret = 'p6m!6/IKpF|cD:t>/xm9I}dr5!QjlyRw>#DeNiya'

    config.save("my-profile")


def green_date_heatmap(img: numpy.ndarray):
    img_raw = (img * 255).astype(int)

    # Threshold the hue and saturation channels# Set the lower and upper green color thresholds
    lower_green = np.array([60, 50, 50])
    upper_green = np.array([150, 255, 255])

    heat_map = np.ones((img_raw.shape[0], img_raw.shape[1]))
    for x in range(img_raw.shape[0]):
        for y in range(img_raw.shape[1]):
            if all(img_raw[x, y, :] >= lower_green) and all(img_raw[x, y, :] <= upper_green):
                heat_map[x, y] = 0

    shape = heat_map.shape[1] * heat_map.shape[0]
    total_area = sum(sum(heat_map)) / shape
    return total_area


def get_santinelhub_image(north: float, south: float, east: float, west: float):
    """Get green rate from S"""
    def convert_image(img, factor):
        img = factor / 255 * img
        img = np.where(img > 1, 1, img)
        img = np.where(img < 0, 0, img)
        return img

    config = SHConfig("my-profile")
    resolution = 10
    betsiboka_coords_wgs84 = (north, south, east, west)
    betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
    betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)

    print(f"Image shape at {resolution} m resolution: {betsiboka_size} pixels")

    evalscript_true_color = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
    """

    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=("2022-06-01", "2022-06-30"),
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=betsiboka_bbox,
        size=betsiboka_size,
        config=config,
    )

    true_color_imgs = request_true_color.get_data()
    img = convert_image(true_color_imgs[0], 3.5)
    return green_date_heatmap(img)
