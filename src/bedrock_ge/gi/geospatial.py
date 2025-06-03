import geopandas as gpd
import pandas as pd
from pyproj import CRS, Transformer
from pyproj.crs import CompoundCRS
from shapely.geometry import LineString

from bedrock_ge.gi.schemas import BedrockGIDatabase, BedrockGIGeospatialDatabase


def location_gis_geometry(brgi_db: BedrockGIDatabase) -> gpd.GeoDataFrame:
    # TODO: Implement logic to handle multiple CRS'es in the input GI data:
    #       1. Create WKT geometry for each location in original CRS
    #       2. Convert to WGS84 + EGM2008 orthometric height EPSG:9518
    #       3. Interpolate InSituTest and Sample geospatial vector geometry from active geometry column
    hor_crs_series = brgi_db.Project["horizontal_crs_wkt"]
    vert_crs_series = brgi_db.Project["vertical_crs_wkt"]
    if hor_crs_series.nunique() > 1 or vert_crs_series.nunique() > 1:
        raise ValueError(
            "All projects must have the same horizontal and vertical CRS (Coordinate Reference System).\n"
            "Raise an issue on GitHub in case you need to be able to combine GI data that was acquired in multiple different CRSes."
        )

    horizontal_crs = CRS.from_wkt(hor_crs_series.iat[0])
    vertical_crs = CRS.from_wkt(vert_crs_series.iat[0])
    compound_crs = CompoundCRS(
        name=f"{horizontal_crs.name} + {vertical_crs.name}",
        components=[horizontal_crs, vertical_crs],
    )

    # TODO: Implement logic such that inclined borholes are handled correctly.
    #       All boreholes are now assumed to be vertical.
    location_df = brgi_db.Location.copy()
    location_df["elevation_at_base"] = (
        location_df["ground_level_elevation"] - location_df["depth_to_base"]
    )
    return gpd.GeoDataFrame(
        brgi_db.Location.copy(),
        geometry=location_df.apply(
            lambda row: LineString(
                [
                    (row["easting"], row["northing"], row["ground_level_elevation"]),
                    (row["easting"], row["northing"], row["elevation_at_base"]),
                ]
            ),
            axis=1,
        ),
        crs=compound_crs,
    )


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
        geometry=gpd.points_from_xy(
            lon_lat_height_df["longitude"],
            lon_lat_height_df["latitude"],
            lon_lat_height_df["egm2008_ground_level_height"],
        ),
        crs=wgs84_egm2008_crs,
    )
