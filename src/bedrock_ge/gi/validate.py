import geopandas as gpd  # type: ignore
import pandas as pd

from bedrock_ge.gi.schemas import (
    BedrockGIDatabase,
    BedrockGIGeospatialDatabase,
)


def check_brgi_geospatial_database(
    brgi_geodb: BedrockGIGeospatialDatabase,
):
    """Validates the structure and relationships of a 'Bedrock Ground Investigation' (BrGI) geospatial database.

    This function checks that all tables in the BrGI geospatialdatabase conform to their respective schemas
    and that all foreign key relationships are properly maintained. It validates the following tables:
    - Project
    - Location
    - LonLatHeight
    - All In-Situ test tables
    - Sample
    - All Lab test tables

    Args:
        brgi_geodb (BedrockGIGeospatialDatabase): Bedrock GI geospatial database object.

    Returns:
        is_valid (bool): True if all tables are valid and relationships are properly maintained.

    Example:
        ```python
        brgi_geodb = BedrockGIGeospatialDatabase(
            Project=project_df,
            Location=location_geodf,
            LonLatHeight=lon_lat_height_geodf,
            InSituTests={"ISPT": ispt_geodf},
            Sample=sample_geodf,
            LabTests={"LLPL": llpl_df},
        )
        check_brgi_geospatial_database(brgi_db)
        ```
    """
    # TODO: implement this
    return True


def check_brgi_database(
    brgi_db: BedrockGIDatabase,
):
    """Validates the structure and relationships of a 'Bedrock Ground Investigation' (BrGI) database.

    This function performs the same validation as `check_brgi_geospatial_database`, but uses schemas
    that don't require geospatial geometry. It validates the following tables:
    - Project (never has geospatial geometry)
    - Location (without geospatial geometry)
    - All In-Situ test tables (without geospatial geometry)
    - Sample (without geospatial geometry)
    - All Lab test tables (never has geospatial geometry)

    Args:
        brgi_db (BedrockGIDatabase): A Bedrock GI database object.

    Returns:
        bool: True if all tables are valid and relationships are properly maintained.

    Example:
        ```python
        brgi_db = BedrockGIDatabase(
            Project=project_df,
            Location=location_df,
            InSituTests={"ISPT": ispt_df},
            Sample=sample_df,
            LabTests={"LLPL": llpl_df},
        )
        check_brgi_database(brgi_db)
        ```
    """
    # TODO: implement this
    return True


def check_foreign_key(
    foreign_key: str,
    parent_table: pd.DataFrame | gpd.GeoDataFrame,
    table_with_foreign_key: pd.DataFrame | gpd.GeoDataFrame,
) -> bool:
    """Validates referential integrity between two tables by checking foreign key relationships.

    This function ensures that all foreign key values in a child table exist in the corresponding
    parent table, maintaining data integrity in the GIS database.

    Args:
        foreign_key (str): The name of the column that serves as the foreign key.
        parent_table (pd.DataFrame| gpd.GeoDataFrame): The parent table containing the primary keys.
        table_with_foreign_key (pd.DataFrame| gpd.GeoDataFrame): The child table containing the foreign keys.

    Returns:
        bool: True if all foreign keys exist in the parent table.

    Raises:
        ValueError: If any foreign key values in the child table do not exist in the parent table.

    Example:
        ```python
        check_foreign_key("project_uid", projects_df, locations_df)
        ```
    """
    # Get the foreign keys that are missing in the parent group
    missing_foreign_keys = table_with_foreign_key[
        ~table_with_foreign_key[foreign_key].isin(parent_table[foreign_key])
    ]

    # Raise an error if there are missing foreign keys
    if len(missing_foreign_keys) > 0:
        raise ValueError(
            f"This table contains '{foreign_key}'s that don't occur in the parent table:\n{missing_foreign_keys}"
        )

    return True
