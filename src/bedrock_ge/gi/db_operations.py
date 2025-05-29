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

    target_db = dbs.pop(0)

    merged_db = {
        "Project": "merged_project",
        "Location": "merged_location",
    }

    # merged_db = BedrockGIDatabase(
    #     Project=target_db.Project.append(incoming_db.Project),
    #     Location=target_db.Location.append(incoming_db.Location),
    #     InSituTests={
    #         k: target_db.InSituTests[k].append(incoming_db.InSituTests[k])
    #         for k in target_db.InSituTests
    #         if k in incoming_db.InSituTests
    #     },
    #     Sample=target_db.Sample.append(incoming_db.Sample),
    #     LabTests={
    #         k: target_db.LabTests[k].append(incoming_db.LabTests[k])
    #         for k in target_db.LabTests
    #         if k in incoming_db.LabTests
    #     },
    #     Other={
    #         k: target_db.Other[k].append(incoming_db.Other[k])
    #         for k in target_db.Other
    #         if k in incoming_db.Other
    #     },
    # )
    return merged_db
