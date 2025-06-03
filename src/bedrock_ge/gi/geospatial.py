import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
from pyproj.crs import CompoundCRS

from bedrock_ge.gi.schemas import BedrockGIDatabase, BedrockGIGeospatialDatabase


def location_gis_geometry(brgi_db: BedrockGIDatabase) -> gpd.GeoDataFrame:
    location_gdf = brgi_db.Location
    return location_gdf


def create_lon_lat_height_table(brgi_db: BedrockGIDatabase) -> gpd.GeoDataFrame:
    wgs84_egm2008_crs = CRS("EPSG:9518")
    crs_lookup = brgi_db.Project.set_index("project_uid")
    dfs = []
    for project_uid, location_df in brgi_db.Location.groupby("project_uid"):
        horizontal_crs = CRS.from_wkt(crs_lookup.at[project_uid, "horizontal_crs_wkt"])
        vertical_crs = CRS.from_wkt(crs_lookup.at[project_uid, "vertical_crs_wkt"])
        compound_crs = CompoundCRS(
            name=f"{horizontal_crs.name} + {vertical_crs.name}",
            components=[horizontal_crs, vertical_crs],
        )
        transformer = Transformer.from_crs(
            compound_crs, wgs84_egm2008_crs, always_xy=True
        )
        lon, lat, egm2008_height = transformer.transform(
            location_df["easting"],
            location_df["northing"],
            location_df["ground_level_elevation"],
        )
        dfs.append(
            pd.DataFrame(
                {
                    "project_uid": project_uid,
                    "location_uid": location_df["location_uid"],
                    "longitude": lon,
                    "latitude": lat,
                    "egm2008_ground_level_height": egm2008_height,
                }
            )
        )

    lon_lat_height_df = pd.concat(dfs, ignore_index=True)
    return gpd.GeoDataFrame(
        lon_lat_height_df,
        crs=wgs84_egm2008_crs,
        geometry=gpd.points_from_xy(
            lon_lat_height_df["longitude"],
            lon_lat_height_df["latitude"],
            lon_lat_height_df["egm2008_ground_level_height"],
        ),
    )
