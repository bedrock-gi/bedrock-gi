from __future__ import annotations

from typing import Dict, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyproj.crs import CRS
from shapely.geometry import LineString, Point


def calculate_gis_geometry(
    no_gis_brgi_db: Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]],
    verbose: bool = True,
) -> Dict[str, gpd.GeoDataFrame]:
    """Calculates GIS geometry for tables in a Bedrock Ground Investigation database.

    This function processes a dictionary of DataFrames containing Ground Investigation (GI) data,
    adding appropriate GIS geometry to each table. It handles both 2D and 3D geometries,
    including vertical boreholes and sampling locations.

    Args:
        no_gis_brgi_db (Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]]): Dictionary containing
            the Bedrock GI database tables without GIS geometry. Keys are table names,
            values are either pandas DataFrames or GeoDataFrames.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        Dict[str, gpd.GeoDataFrame]: Dictionary containing the Bedrock GI database tables
            with added GIS geometry. All tables are converted to GeoDataFrames with
            appropriate CRS and geometry columns.

    Raises:
        ValueError: If the projects in the database use different Coordinate Reference Systems (CRS).

    Note:
        The function performs the following operations:

        1. Verifies all projects use the same CRS
        2. Calculates GIS geometry for the 'Location' table
        3. Creates a 'LonLatHeight' table for 2D visualization
        4. Processes 'Sample' table if present
        5. Processes all tables starting with "InSitu_"
    """
    # Make sure that the Bedrock database is not changed outside this function.
    brgi_db = no_gis_brgi_db.copy()

    if verbose:
        print("Calculating GIS geometry for the Bedrock GI database tables...")

    # Check if all projects have the same CRS
    if not brgi_db["Project"]["horizontal_crs_wkt"].nunique() == 1:
        raise ValueError(
            "All projects must have the same CRS (Coordinate Reference System).\n"
            "Raise an issue on GitHub in case you need to be able to combine GI data that was acquired in multiple different CRSes."
        )

    crs = CRS.from_wkt(brgi_db["Project"]["horizontal_crs_wkt"].iloc[0])

    # Calculate GIS geometry for the 'Location' table
    if verbose:
        print("Calculating GIS geometry for the Bedrock GI 'Location' table...")
    brgi_db["Location"] = calculate_location_gis_geometry(brgi_db["Location"], crs)

    # Create the 'LonLatHeight' table.
    # The 'LonLatHeight' table makes it easier to visualize the GIS geometry on 2D maps,
    # because vertical lines are often very small or completely hidden in 2D.
    # This table only contains the 3D of the GI locations at ground level,
    # in WGS84 (Longitude, Latitude, Height) coordinates.
    if verbose:
        print(
            "Creating 'LonLatHeight' table with GI locations in WGS84 geodetic coordinates...",
            "    WGS84 geodetic coordinates: (Longitude, Latitude, Ground Level Ellipsoidal Height)",
            sep="\n",
        )
    brgi_db["LonLatHeight"] = create_lon_lat_height_table(brgi_db["Location"], crs)

    # Create GIS geometry for tables that have In-Situ GIS geometry.
    # These are the 'Sample' table and 'InSitu_...' tables.
    # These tables are children of the Location table,
    # i.e. have the 'Location' table as the parent table.
    if "Sample" in brgi_db.keys():
        if verbose:
            print("Calculating GIS geometry for the Bedrock GI 'Sample' table...")
        brgi_db["Sample"] = calculate_in_situ_gis_geometry(
            brgi_db["Sample"], brgi_db["Location"], crs
        )

    for table_name, table in brgi_db.items():
        if table_name.startswith("InSitu_"):
            if verbose:
                print(
                    f"Calculating GIS geometry for the Bedrock GI '{table_name}' table..."
                )
            brgi_db[table_name] = calculate_in_situ_gis_geometry(
                table, brgi_db["Location"], crs
            )

    return brgi_db


def calculate_location_gis_geometry(
    brgi_location: Union[pd.DataFrame, gpd.GeoDataFrame], crs: CRS
) -> gpd.GeoDataFrame:
    """Calculates GIS geometry for a set of Ground Investigation locations.

    Args:
        brgi_location (Union[pd.DataFrame, gpd.GeoDataFrame]): The GI locations to calculate GIS geometry for.
        crs (pyproj.CRS): The Coordinate Reference System (CRS) to use for the GIS geometry.

    Returns:
        gpd.GeoDataFrame: The GIS geometry for the given GI locations, with additional columns:
            - longitude: The longitude of the location in the WGS84 CRS.
            - latitude: The latitude of the location in the WGS84 CRS.
            - wgs84_ground_level_height: The height of the ground level of the location in the WGS84 CRS.
            - elevation_at_base: The elevation at the base of the location.
            - geometry: The GIS geometry of the location.
    """
    # Calculate Elevation at base of GI location
    brgi_location["elevation_at_base"] = (
        brgi_location["ground_level_elevation"] - brgi_location["depth_to_base"]
    )

    # Make a gpd.GeoDataFrame from the pd.DataFrame by creating GIS geometry
    brgi_location = gpd.GeoDataFrame(
        brgi_location,
        geometry=brgi_location.apply(
            lambda row: LineString(
                [
                    (row["easting"], row["northing"], row["ground_level_elevation"]),
                    (row["easting"], row["northing"], row["elevation_at_base"]),
                ]
            ),
            axis=1,
        ),
        crs=crs,
    )

    # Calculate WGS84 geodetic coordinates
    brgi_location[["longitude", "latitude", "wgs84_ground_level_height"]] = (
        brgi_location.apply(
            lambda row: calculate_wgs84_coordinates(
                from_crs=crs,
                easting=row["easting"],
                northing=row["northing"],
                elevation=row["ground_level_elevation"],
            ),
            axis=1,
            result_type="expand",
        )
    )

    return brgi_location


def calculate_wgs84_coordinates(
    from_crs: CRS, easting: float, northing: float, elevation: Union[float, None] = None
) -> Tuple[float, float, (float | None)]:
    """Transforms coordinates from an arbitrary Coordinate Reference System (CRS) to the WGS84 CRS, which is the standard for geodetic coordinates.

    Args:
        from_crs (pyproj.CRS): The pyproj.CRS object of the CRS to transform from.
        easting (float): The easting coordinate of the point to transform.
        northing (float): The northing coordinate of the point to transform.
        elevation (float or None, optional): The elevation of the point to
            transform. Defaults to None.

    Returns:
        Tuple[float, float, (float | None)]: A tuple containing the longitude, latitude
            and WGS84 height of the transformed point, in that order.
            The height is None if no elevation was given, or if the provided CRS doesn't
            have a proper datum defined.
    """
    transformer = Transformer.from_crs(from_crs, 4326, always_xy=True)
    if elevation:
        lon, lat, wgs84_height = transformer.transform(easting, northing, elevation)
    else:
        lon, lat = transformer.transform(easting, northing)
        wgs84_height = None

    return (lon, lat, wgs84_height)


def create_lon_lat_height_table(
    brgi_location: gpd.GeoDataFrame, crs: CRS
) -> gpd.GeoDataFrame:
    """Creates a GeoDataFrame with GI locations in WGS84 (lon, lat, height) coordinates.

    The 'LonLatHeight' table makes it easier to visualize the GIS geometry on 2D maps,
    because vertical lines are often very small or completely hidden in 2D. This table
    only contains the 3D point of the GI locations at ground level, in WGS84 (Longitude,
    Latitude, Height) coordinates. Other attributes, such as the location type, sample
    type, geology description, etc., can be attached to this table by joining, i.e.
    merging those tables on the location_uid key.

    Args:
        brgi_location (GeoDataFrame): The GeoDataFrame with the GI locations.
        crs (CRS): The Coordinate Reference System of the GI locations.

    Returns:
        gpd.GeoDataFrame: The 'LonLatHeight' GeoDataFrame.
    """
    lon_lat_height = gpd.GeoDataFrame(
        brgi_location[
            [
                "project_uid",
                "location_uid",
            ]
        ],
        geometry=brgi_location.apply(
            lambda row: Point(
                row["longitude"], row["latitude"], row["wgs84_ground_level_height"]
            ),
            axis=1,
        ),
        crs=4326,
    )
    return lon_lat_height


def calculate_in_situ_gis_geometry(
    brgi_in_situ: Union[pd.DataFrame, gpd.GeoDataFrame],
    brgi_location: Union[pd.DataFrame, gpd.GeoDataFrame],
    crs: CRS,
) -> gpd.GeoDataFrame:
    """Calculates GIS geometry for a set of Ground Investigation in-situ data.

    Args:
        brgi_in_situ (Union[pd.DataFrame, gpd.GeoDataFrame]): The in-situ data to calculate GIS geometry for.
        brgi_location (Union[pd.DataFrame, gpd.GeoDataFrame]): The location data to merge with the in-situ data.
        crs (CRS): The Coordinate Reference System of the in-situ data.

    Returns:
        gpd.GeoDataFrame: The GIS geometry for the given in-situ data, with additional columns:
            - elevation_at_top: The elevation at the top of the in-situ data.
            - elevation_at_base: The elevation at the base of the in-situ data.
            - geometry: The GIS geometry of the in-situ data.
    """
    location_child = brgi_in_situ.copy()

    # Merge the location data into the in-situ data to get the location coordinates
    location_child = pd.merge(
        location_child,
        brgi_location[
            ["location_uid", "easting", "northing", "ground_level_elevation"]
        ],
        on="location_uid",
        how="left",
    )

    # Calculate the elevation at the top of the Sample or in-situ test
    location_child["elevation_at_top"] = (
        location_child["ground_level_elevation"] - location_child["depth_to_top"]
    )
    brgi_in_situ["elevation_at_top"] = location_child["elevation_at_top"]

    # Calculate the elevation at the base of the Sample or in-situ test
    if "depth_to_base" in location_child.columns:
        location_child["elevation_at_base"] = (
            location_child["ground_level_elevation"] - location_child["depth_to_base"]
        )
        brgi_in_situ["elevation_at_base"] = location_child["elevation_at_base"]

    # Create the in-situ data as a GeoDataFrame with LineString GIS geometry for
    # Samples or in-situ tests that have an elevation at the base of the Sample or in-situ test.
    brgi_in_situ = gpd.GeoDataFrame(
        brgi_in_situ,
        geometry=location_child.apply(
            lambda row: LineString(
                [
                    (row["easting"], row["northing"], row["elevation_at_top"]),
                    (row["easting"], row["northing"], row["elevation_at_base"]),
                ]
            )
            if "elevation_at_base" in row and not np.isnan(row["elevation_at_base"])
            else Point((row["easting"], row["northing"], row["elevation_at_top"])),
            axis=1,
        ),
        crs=crs,
    )
    return brgi_in_situ
