from collections.abc import Iterable

import pandas as pd

from bedrock_ge.gi.io_utils import convert_dtypes_object_to_string
from bedrock_ge.gi.schemas import (
    BedrockGIDatabase,
    InSituTestSchema,
    LabTestSchema,
    LocationSchema,
    ProjectSchema,
    SampleSchema,
)
from bedrock_ge.gi.validate import check_foreign_key


def merge_databases(
    brgi_databases: Iterable[BedrockGIDatabase],
) -> BedrockGIDatabase:
    """Merges the incoming Bedrock GI database into the target Bedrock GI database.

    The function concatenates the pandas DataFrames of the second dict of
    DataFrames to the first dict of DataFrames for the keys they have in common.
    Keys that are unique to either dictionary will be included in the final
    concatenated dictionary.

    Args:
        brgi_databases (Iterable[BedrockGIDatabase]): The Bedrock GI databases containing the data to be merged.

    Returns:
        BedrockGIDatabase: Merged Bedrock GI database.
    """
    dbs = list(brgi_databases)

    if not dbs:
        raise ValueError("Cannot merge an empty list of Bedrock GI databases.")
    elif len(dbs) == 1 and isinstance(dbs[0], BedrockGIDatabase):
        return dbs[0]

    project_dataframes = _filter_dataframes([db.Project for db in dbs])
    merged_project = pd.concat(project_dataframes, ignore_index=True)
    merged_project = merged_project.drop_duplicates().reset_index(drop=True)
    merged_project = convert_dtypes_object_to_string(merged_project.convert_dtypes())
    ProjectSchema.validate(merged_project)

    location_dataframes = _filter_dataframes([db.Location for db in dbs])
    merged_location = pd.concat(location_dataframes, ignore_index=True)
    merged_location = merged_location.drop_duplicates().reset_index(drop=True)
    merged_location = convert_dtypes_object_to_string(merged_location.convert_dtypes())
    LocationSchema.validate(merged_location)
    check_foreign_key("project_uid", merged_project, merged_location)

    insitu_tables: set[str] = set()
    lab_tables: set[str] = set()
    other_tables: set[str] = set()
    for db in dbs:
        insitu_tables.update(db.InSituTests.keys())
        if db.LabTests:
            lab_tables.update(db.LabTests.keys())
        if db.Other:
            other_tables.update(db.Other.keys())

    merged_insitu: dict[str, pd.DataFrame] = {}
    for table_name in insitu_tables:
        insitu_dataframes = _filter_dataframes(
            [db.InSituTests.get(table_name) for db in dbs]
        )
        insitu_df = pd.concat(insitu_dataframes, ignore_index=True)
        insitu_df = insitu_df.drop_duplicates().reset_index(drop=True)
        insitu_df = convert_dtypes_object_to_string(insitu_df.convert_dtypes())
        InSituTestSchema.validate(insitu_df)
        check_foreign_key("project_uid", merged_project, insitu_df)
        check_foreign_key("location_uid", merged_location, insitu_df)
        merged_insitu[table_name] = insitu_df

    sample_dfs = _filter_dataframes([db.Sample for db in dbs])
    merged_sample = None
    if sample_dfs:
        merged_sample = pd.concat(sample_dfs, ignore_index=True)
        merged_sample = merged_sample.drop_duplicates().reset_index(drop=True)
        merged_sample = convert_dtypes_object_to_string(merged_sample.convert_dtypes())
        SampleSchema.validate(merged_sample)
        check_foreign_key("project_uid", merged_project, merged_sample)

    merged_lab: dict[str, pd.DataFrame] = {}
    for table_name in lab_tables:
        lab_dataframes = _filter_dataframes([db.LabTests.get(table_name) for db in dbs])
        lab_df = pd.concat(lab_dataframes, ignore_index=True)
        lab_df = lab_df.drop_duplicates().reset_index(drop=True)
        lab_df = convert_dtypes_object_to_string(lab_df.convert_dtypes())
        LabTestSchema.validate(lab_df)
        check_foreign_key("project_uid", merged_project, lab_df)
        check_foreign_key("sample_uid", merged_sample, lab_df)
        merged_lab[table_name] = lab_df

    merged_other: dict[str, pd.DataFrame] = {}
    for table_name in other_tables:
        other_dataframes = _filter_dataframes([db.Other.get(table_name) for db in dbs])
        other_df = pd.concat(other_dataframes, ignore_index=True)
        other_df = other_df.drop_duplicates().reset_index(drop=True)
        other_df = convert_dtypes_object_to_string(other_df.convert_dtypes())
        check_foreign_key("project_uid", merged_project, other_df)
        merged_other[table_name] = other_df

    return BedrockGIDatabase(
        Project=merged_project,
        Location=merged_location,
        InSituTests=merged_insitu,
        Sample=merged_sample,
        LabTests=merged_lab,
        Other=merged_other,
    )


def _filter_dataframes(dataframes: list[pd.DataFrame | None]) -> list[pd.DataFrame]:
    """Filter out empty or all-NA DataFrames to avoid FutureWarnings."""
    valid_dfs = []
    for df in dataframes:
        if df is not None and not df.empty and not df.isna().all().all():
            if df.columns.duplicated().any():
                raise ValueError(
                    f"Duplicate column names found in dataframe:\n{list(df.columns)}"
                )

            df.dropna(axis=1, how="all", inplace=True)

            valid_dfs.append(df)
    return valid_dfs
