from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import geopandas as gpd
import pandas as pd

from bedrock_ge.gi.io_utils import brgi_db_to_dfs, geodf_to_df
from bedrock_ge.gi.schemas import BedrockGIDatabase, BedrockGIGeospatialDatabase


# ? Should this function be made a to_file(s) method of BedrockGIDatabase?
def write_brgi_db_to_file(
    brgi_db: BedrockGIDatabase | BedrockGIGeospatialDatabase,
    path: str | Path,
    driver: str,
) -> None:
    """Writes a Bedrock GI (geospatial) database to a file.

    Writes a Bedrock GI (geospatial) database to a file. The file type is
    determined by the `driver` argument. Possible values are "GPKG" and "EXCEL".

    Args:
        brgi_db (BedrockGIDatabase | BedrockGIGeospatialDatabase): The Bedrock GI (geospatial) database.
        path (str | Path): The path of the output file.
        driver (str): The type of the output file. Possible values are "GPKG" and "EXCEL".

    Returns:
        None
    """
    dict_of_dfs = brgi_db_to_dfs(brgi_db)
    if driver.upper() == "GPKG":
        write_gi_db_to_gpkg(dict_of_dfs, path)
    elif driver.upper() == "EXCEL":
        write_gi_db_to_excel(dict_of_dfs, path)


def write_gi_db_to_gpkg(
    dict_of_dfs: dict[str, pd.DataFrame | gpd.GeoDataFrame],
    gpkg_path: str | Path,
) -> None:
    """Writes a database with Bedrock Ground Investigation data to a GeoPackage file.

    Writes a dictionary of DataFrames containing Bedrock Ground Investigation data to a
    [GeoPackage file](https://www.geopackage.org/). Each DataFrame will be saved in a
    separate table named by the keys of the dictionary.

    Args:
        dict_of_dfs (dict[str, pd.DataFrame | gpd.GeoDataFrame]): A dictionary where
            keys are brgi table names and values are pandas DataFrames or GeoDataFrames
            with brgi data.
        gpkg_path (str | Path): The name of the output GeoPackage file.

    Returns:
        None
    """
    # Create a GeoDataFrame from the dictionary of DataFrames
    for table_name, df in dict_of_dfs.items():
        sanitized_table_name = sanitize_table_name(table_name)
        if isinstance(df, pd.DataFrame):
            df = gpd.GeoDataFrame(df)

        df.to_file(gpkg_path, driver="GPKG", layer=sanitized_table_name, overwrite=True)

    print(f"Ground Investigation data has been written to '{gpkg_path}'.")


def write_gi_db_to_excel(
    dict_of_dfs: dict[str, pd.DataFrame | gpd.GeoDataFrame],
    excel_path: str | Path,
) -> None:
    """Writes a database with Ground Investigation data to an Excel file.

    Each DataFrame in the database dictionary will be saved in a separate Excel sheet named
    after the dictionary keys. This function can be used on any GI database, whether in
    AGS, Bedrock, or another format.

    Args:
        dict_of_dfs (dict[str, pd.DataFrame | gpd.GeoDataFrame]): A dictionary where
            keys are GI table names and values are DataFrames with GI data.
        excel_path (str | Path): Path to the output Excel file. Can be provided as a
            string or Path object.

    Returns:
        None
    """
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet_name, df in dict_of_dfs.items():
            sanitized_sheet_name = sanitize_table_name(sheet_name)[:31]
            if isinstance(df, gpd.GeoDataFrame):
                df = geodf_to_df(df)

            df.to_excel(writer, sheet_name=sanitized_sheet_name, index=False)

    print(f"Ground Investigation data has been written to '{excel_path}'.")


def sanitize_table_name(sheet_name):
    """Replaces invalid characters and spaces in GI table names with underscores.

    Makes table names consistent with SQL, GeoPackage and Excel naming conventions by
    replacing invalid characters and spaces with underscores.

    Args:
        sheet_name (str): The original sheet name.

    Returns:
        sanitized_name (str): A sanitized sheet name with invalid characters and spaces replaced.
    """
    invalid_chars = [":", "/", "\\", "?", "*", "[", "]"]
    sanitized_name = sheet_name.strip()
    for char in invalid_chars:
        sanitized_name = sanitized_name.replace(char, "_")

    # Replace spaces with underscores
    sanitized_name = sanitized_name.replace(" ", "_")

    # Collapse multiple underscores to one
    sanitized_name = "_".join(filter(None, sanitized_name.split("_")))

    if sheet_name != sanitized_name:
        print(
            f"Table names shouldn't contain {invalid_chars} or spaces and shouldn't be longer than 31 characters.\n",
            f"Replaced '{sheet_name}' with '{sanitized_name}'.",
        )

    return sanitized_name
