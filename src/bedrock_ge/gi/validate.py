from typing import Dict, Union

import geopandas as gpd  # type: ignore
import pandas as pd

from bedrock_ge.gi.schemas import (
    InSituTestSchema,
    LocationSchema,
    ProjectSchema,
    SampleSchema,
)


# TODO: rename to check_brgi_geodb
# TODO: make this check actually work...
def check_brgi_database(brgi_db: Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]]):
    """Validates the structure and relationships of a 'Bedrock Ground Investigation' (BRGI) database (which is a dictionary of DataFrames).

    This function checks that all tables in the BRGI database conform to their respective schemas
    and that all foreign key relationships are properly maintained. It validates the following tables:
    - Project
    - Location
    - Sample
    - InSitu_TESTX
    - Lab_TESTY (not yet implemented)

    Args:
        brgi_db (Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]]): A dictionary
            containing the BRGI database tables, where keys are table names and
            values are the corresponding data tables (DataFrame or GeoDataFrame).

    Returns:
        is_valid (bool): True if all tables are valid and relationships are properly maintained.

    Example:
        ```python
        brgi_db = {
            "Project": project_df,
            "Location": location_gdf,
            "Sample": sample_gdf,
            "InSitu_ISPT": in_situ_ispt_gdf,
        }
        check_brgi_database(brgi_db)
        ```
    """
    for table_name, table in brgi_db.items():
        if table_name == "Project":
            ProjectSchema.validate(table)
            print("'Project' table aligns with Bedrock's 'Project' table schema.")
        elif table_name == "Location":
            LocationSchema.validate(table)
            check_foreign_key("project_uid", brgi_db["Project"], table)
            print("'Location' table aligns with Bedrock's 'Location' table schema.")
        elif table_name == "Sample":
            SampleSchema.validate(table)
            check_foreign_key("project_uid", brgi_db["Project"], table)
            check_foreign_key("location_uid", brgi_db["Location"], table)
            print("'Sample' table aligns with Bedrock's 'Sample' table schema.")
        # ! JG is pretty sure that this doesn't work
        # ! The line below should be:
        # ! elif table_name.startswith("InSitu_"):
        elif table_name == "InSitu":
            InSituTestSchema.validate(table)
            check_foreign_key("project_uid", brgi_db["Project"], table)
            check_foreign_key("location_uid", brgi_db["Location"], table)
            print(
                f"'{table_name}' table aligns with Bedrock's table schema for In-Situ measurements."
            )
        elif table_name.startswith("Lab_"):
            print(
                "🚨 !NOT IMPLEMENTED! We haven't come across Lab data yet. !NOT IMPLEMENTED!"
            )

    return True


# TODO: rename to check_brgi_db
def check_no_gis_brgi_database(
    brgi_db: Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]],
):
    """Validates the structure and relationships of a 'Bedrock Ground Investigation' (BGI) database without GIS geometry.

    This function performs the same validation as `check_brgi_database` but uses schemas
    that don't require GIS geometry. It validates the following tables:
    - Project (never has GIS geometry)
    - Location (without GIS geometry)
    - Sample (without GIS geometry)
    - InSitu_TESTX (without GIS geometry)
    - Lab_TESTY (not yet implemented)

    Args:
        brgi_db (Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]]): A dictionary
            containing the Bedrock GI database tables, where keys are table names
            and values are the corresponding data tables (DataFrame or GeoDataFrame).

    Returns:
        bool: True if all tables are valid and relationships are properly maintained.

    Example:
        ```python
        brgi_db = {
            "Project": projects_df,
            "Location": locations_df,
            "Sample": samples_df,
            "InSitu_measurements": insitu_df,
        }
        check_no_gis_brgi_database(brgi_db)
        ```
    """
    for table_name, table in brgi_db.items():
        if table_name == "Project":
            ProjectSchema.validate(table)
            print("'Project' table aligns with Bedrock's 'Project' table schema.")
        elif table_name == "Location":
            LocationSchema.validate(table)
            check_foreign_key("project_uid", brgi_db["Project"], table)
            print(
                "'Location' table aligns with Bedrock's 'Location' table schema without GIS geometry."
            )
        elif table_name == "Sample":
            SampleSchema.validate(table)
            check_foreign_key("project_uid", brgi_db["Project"], table)
            check_foreign_key("location_uid", brgi_db["Location"], table)
            print(
                "'Sample' table aligns with Bedrock's 'Sample' table schema without GIS geometry."
            )
        elif table_name.startswith("InSitu_"):
            InSituTestSchema.validate(table)
            check_foreign_key("project_uid", brgi_db["Project"], table)
            check_foreign_key("location_uid", brgi_db["Location"], table)
            print(
                f"'{table_name}' table aligns with Bedrock's '{table_name}' table schema without GIS geometry."
            )
        elif table_name.startswith("Lab_"):
            print(
                "🚨 !NOT IMPLEMENTED! We haven't come across Lab data yet. !NOT IMPLEMENTED!"
            )

    return True


def check_foreign_key(
    foreign_key: str,
    parent_table: Union[pd.DataFrame, gpd.GeoDataFrame],
    table_with_foreign_key: Union[pd.DataFrame, gpd.GeoDataFrame],
) -> bool:
    """Validates referential integrity between two tables by checking foreign key relationships.

    This function ensures that all foreign key values in a child table exist in the corresponding
    parent table, maintaining data integrity in the GIS database.

    Args:
        foreign_key (str): The name of the column that serves as the foreign key.
        parent_table (Union[pd.DataFrame, gpd.GeoDataFrame]): The parent table containing the primary keys.
        table_with_foreign_key (Union[pd.DataFrame, gpd.GeoDataFrame]): The child table containing the foreign keys.

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
