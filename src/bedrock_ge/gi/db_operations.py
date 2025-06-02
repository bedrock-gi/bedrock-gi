from collections.abc import Iterable

import pandas as pd

from bedrock_ge.gi.schemas import (
    BedrockGIDatabase,
    InSituTestSchema,
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
        target_db (BedrockGIDatabase): The Bedrock GI database into which the incoming data will be merged.
        incoming_db (BedrockGIDatabase): The Bedrock GI database containing the data to be merged.

    Returns:
        BedrockGIDatabase: Merged Bedrock GI database.
    """
    # write the body of this function that merges the incoming_db (BedrockGIDatabase) into the target_db (BedrockGIDatabase).
    # duplicate rows in the incoming_db (BedrockGIDatabase) will be dropped.
    # After merging tables validate them with the schemas from bedrock_ge.gi.schemas and check that foreign keys are correct.
    # In case the incoming_db contains tables that are not in the target_db, add them to the target_db.
    # The function must return a BedrockGIDatabase object.

    # merged_project = pd.concat(
    #     [target_db.Project, incoming_db.Project], ignore_index=True
    # )
    # ProjectSchema.validate(merged_project)

    # merged_location = pd.concat(
    #     [target_db.Location, incoming_db.Location], ignore_index=True
    # )
    # LocationSchema.validate(merged_location)
    # check_foreign_key("project_uid", merged_project, merged_location)

    # merged_insitu = {}

    # Draw inspiration from polars.concat
    # https://github.com/pola-rs/polars/blob/py-1.30.0/py-polars/polars/functions/eager.py

    dbs = list(brgi_databases)

    if not dbs:
        msg = "Cannot merge an empty list of Bedrock GI databases."
        raise ValueError(msg)
    elif len(dbs) == 1 and isinstance(dbs[0], BedrockGIDatabase):
        return dbs[0]

    merged_project = pd.concat([db.Project for db in dbs], ignore_index=True)
    merged_project.drop_duplicates(inplace=True)
    ProjectSchema.validate(merged_project)

    merged_location = pd.concat([db.Location for db in dbs], ignore_index=True)
    merged_location.drop_duplicates(inplace=True)
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
    for insitu_table in insitu_tables:
        insitu_df = pd.concat(
            [db.InSituTests.get(insitu_table) for db in dbs], ignore_index=True
        )
        insitu_df.drop_duplicates(inplace=True)
        InSituTestSchema.validate(insitu_df)
        check_foreign_key("project_uid", merged_project, insitu_df)
        check_foreign_key("location_uid", merged_location, insitu_df)
        merged_insitu[insitu_table] = insitu_df

    sample_dfs = [db.Sample for db in dbs if db.Sample is not None]
    if sample_dfs:
        merged_sample = pd.concat(sample_dfs, ignore_index=True)
        merged_sample.drop_duplicates(inplace=True)
        SampleSchema.validate(merged_sample)
        check_foreign_key("project_uid", merged_project, merged_sample)

    merged_lab: dict[str, pd.DataFrame] = {}
    for lab_table in lab_tables:
        lab_dfs = [
            db.LabTests.get(lab_table)
            for db in dbs
            if db.LabTests.get(lab_table) is not None
        ]
        lab_df = pd.concat(lab_dfs, ignore_index=True)
        lab_df.drop_duplicates(inplace=True)
        check_foreign_key("project_uid", merged_project, lab_df)
        check_foreign_key("sample_uid", merged_sample, lab_df)
        merged_lab[lab_table] = lab_df

    merged_other: dict[str, pd.DataFrame] = {}
    for other_table in other_tables:
        other_dfs = [
            db.Other.get(other_table)
            for db in dbs
            if db.Other.get(other_table) is not None
        ]
        other_df = pd.concat(other_dfs, ignore_index=True)
        other_df.drop_duplicates(inplace=True)
        check_foreign_key("project_uid", merged_project, other_df)
        merged_other[other_table] = other_df

    return BedrockGIDatabase(
        Project=merged_project,
        Location=merged_location,
        InSituTests=merged_insitu,
        Sample=merged_sample,
        LabTests=merged_lab,
        Other=merged_other,
    )
