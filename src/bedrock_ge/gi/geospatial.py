from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from pyproj import CRS, Transformer
from pyproj.crs import CompoundCRS
from shapely.geometry import LineString, Point
from shapely.ops import substring

from bedrock_ge.gi.schemas import (
    BedrockGIDatabase,
    BedrockGIGeospatialDatabase,
    InSituTestSchema,
)


def create_brgi_geospatial_database(
    brgi_db: BedrockGIDatabase,
) -> BedrockGIGeospatialDatabase:
    location_gdf = create_location_gdf(brgi_db)
    lon_lat_height_gdf = create_lon_lat_height_gdf(brgi_db)
    # for insitu_test_name, insitu_test_data in brgi_db.InSituTest.items():

    return BedrockGIGeospatialDatabase(
        Project=brgi_db.Project,
        Location=location_gdf,
        LonLatHeight=lon_lat_height_gdf,
        # InSituTests=calculate_in_situ_gis_geometry(
        #     brgi_db.InSituTest, brgi_db.Location
        # ),
        # Sample=calculate_sample_gis_geometry(brgi_db.Sample, brgi_db.Location),
        LabTests=brgi_db.LabTest,
        Other=brgi_db.Other,
    )


def create_location_gdf(brgi_db: BedrockGIDatabase) -> gpd.GeoDataFrame:
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


def create_lon_lat_height_gdf(brgi_db: BedrockGIDatabase) -> gpd.GeoDataFrame:
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


def interpolate_gi_geospatial_geometry(
    insitu_test_df: DataFrame[InSituTestSchema], location_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # TODO: implement a warning when interpolating GI geospatial geometry when
    # TODO: a single GI location has waaay too many rows in a certain In-Situ test.

    gdf = location_gdf[["location_uid", "geometry"]].merge(
        insitu_test_df,
        how="right",
        on="location_uid",
    )
    has_top_depth = "depth_to_top" in insitu_test_df.columns
    has_base_depth = "depth_to_base" in insitu_test_df.columns
    if has_top_depth and has_base_depth:
        return gpd.GeoDataFrame(
            insitu_test_df.copy(),
            geometry=gdf.apply(
                lambda row: substring_3d(
                    row["geometry"],
                    start_dist=row["depth_to_top"],
                    end_dist=row["depth_to_base"],
                ),
                axis=1,
            ),
            crs=gdf.crs,
        )
    elif has_top_depth or has_base_depth:
        depth_key = "depth_to_top" if has_top_depth else "depth_to_base"
        return gpd.GeoDataFrame(
            insitu_test_df.copy(),
            geometry=gdf.apply(
                lambda row: interpolate_3d(
                    row["geometry"],
                    distance=row[depth_key],
                ),
                axis=1,
            ),
            crs=gdf.crs,
        )
    else:
        raise ValueError(
            "An InSituTest dataframe must have either a 'depth_to_top' or a 'depth_to_base' column, or both."
        )


def calc_distances_along_3d_linestring(linestring: LineString) -> np.ndarray:
    """Calculate cumulative distances along a 3D LineString."""
    coords = np.array(linestring.coords)
    if coords.shape[1] < 3:
        raise ValueError("Coordinates must be 3D (x, y, z)")

    # Calculate 3D distances between consecutive points
    diffs = np.diff(coords, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))

    # Return cumulative distances (starting with 0)
    return np.concatenate([[0], np.cumsum(distances)])


def interpolate_3d(linestring: LineString, distance: float) -> Point:
    """Interpolate a point along a 3D LineString using true 3D distance.

    Return the first point if the distance is less than 0 or the last point if
    the distance is greater than the total length. This behavior is different than
    the shapely.LineString.interpolate method.

    Args:
        linestring: A 3D LineString geometry
        distance: Distance along the line in 3D space

    Returns:
        Point: The interpolated 3D point
    """
    if distance <= 0:
        return Point(linestring.coords[0])

    cumulative_distances = calc_distances_along_3d_linestring(linestring)
    total_length = cumulative_distances[-1]

    if distance >= total_length:
        return Point(linestring.coords[-1])

    # Find the segment where the distance falls
    segment_end_idx = int(np.searchsorted(cumulative_distances, distance))
    segment_end_dist = cumulative_distances[segment_end_idx]
    segment_start_idx = max(0, segment_end_idx - 1)  # Ensure non-negative
    segment_start_dist = cumulative_distances[segment_start_idx]

    # Get the coordinates of the point at the start of the segment
    p1 = np.array(linestring.coords[segment_start_idx])
    segment_length = segment_end_dist - segment_start_dist
    if segment_length == 0:
        return Point(p1)
    p2 = np.array(linestring.coords[segment_end_idx])

    # Calculate the ratio of how far along the segment the distance of interest falls
    ratio = (distance - segment_start_dist) / segment_length

    return Point(p1 + ratio * (p2 - p1))


def substring_3d(
    linestring: LineString, start_dist: float, end_dist: float
) -> LineString | Point:
    """Extract a substring of a 3D LineString using true 3D distances.

    Args:
        linestring: A 3D LineString geometry
        start_dist: Start distance along the line in 3D space
        end_dist: End distance along the line in 3D space

    Returns:
        LineString: The extracted 3D LineString segment
    """
    # Ensure start_dist <= end_dist
    if start_dist > end_dist:
        start_dist, end_dist = end_dist, start_dist

    # Calculate cumulative 3D distances
    cumulative_distances = calc_distances_along_3d_linestring(linestring)
    total_length = cumulative_distances[-1]

    # Handle edge cases
    start_dist = max(0, min(start_dist, total_length))
    end_dist = max(0, min(end_dist, total_length))

    if start_dist == end_dist:
        return interpolate_3d(linestring, start_dist)

    # Find segments that intersect with our range
    result_coords = []

    # Add start point if it's not at a linestring vertex
    start_point = interpolate_3d(linestring, start_dist)
    result_coords.append(start_point.coords[0])

    # Add all vertices that fall within the range
    for i, dist in enumerate(cumulative_distances):
        if start_dist < dist < end_dist:
            result_coords.append(linestring.coords[i])

    # Add end point if it's not at a vertex
    end_point = interpolate_3d(linestring, end_dist)
    if end_point.coords[0] != result_coords[-1]:  # Avoid duplicate points
        result_coords.append(end_point.coords[0])

    return LineString(result_coords)


def interpolate_3d_series(geometry_series, distance_series):
    """Apply interpolate_3d to a pandas Series of geometries and distances."""
    return geometry_series.apply(distance_series, interpolate_3d)


def substring_3d_series(geometry_series, start_series, end_series):
    """Apply substring_3d to a pandas Series of geometries and distance pairs."""

    def apply_substring(geom, start, end):
        return substring_3d(geom, start, end)

    return geometry_series.combine(
        start_series,
        lambda g, s: substring_3d(
            g,
            s,
            end_series[
                geometry_series.index.get_loc(
                    geometry_series[geometry_series == g].index[0]
                )
            ],
        ),
    )
