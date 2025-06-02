import pandas as pd

from bedrock_ge.gi.mapping_models import BedrockGIMapping
from bedrock_ge.gi.schemas import (
    BedrockGIDatabase,
    InSituTestSchema,
    LabTestSchema,
    LocationSchema,
    ProjectSchema,
    SampleSchema,
)


def map_to_brgi_db(brgi_db_mapping: BedrockGIMapping) -> BedrockGIDatabase:
    """Creates a Bedrock GI Database for a single project from a BedrockGIMapping.

    This function takes a BedrockGIDatabaseMapping, which contains various table mappings
    for project, location, in-situ tests, samples, lab tests, and other tables, and
    converts it into a BedrockGIDatabase object. It creates pandas DataFrames for each
    table, validates them against their respective schemas, and constructs the final
    BedrockGIDatabase object.

    Args:
        brgi_db_mapping (BedrockGIDatabaseMapping): The mapping object containing GI
            data and metadata for mapping to Bedrock's schema.

    Returns:
        BedrockGIDatabase: The transformed Bedrock GI database containing validated
            DataFrames for each table type.
    """
    project_uid = brgi_db_mapping.Project.project_uid
    project_data = brgi_db_mapping.Project.data or {}
    brgi_project = pd.DataFrame(
        {
            "project_uid": project_uid,
            "horizontal_crs": brgi_db_mapping.Project.horizontal_crs.to_string(),
            "horizontal_crs_wkt": brgi_db_mapping.Project.horizontal_crs.to_wkt(),
            "vertical_crs": brgi_db_mapping.Project.vertical_crs.to_string(),
            "vertical_crs_wkt": brgi_db_mapping.Project.vertical_crs.to_wkt(),
            **project_data,
        },
        index=[0],
    )
    ProjectSchema.validate(brgi_project)

    location_df = pd.DataFrame(
        {
            "location_uid": brgi_db_mapping.Location.data[
                brgi_db_mapping.Location.location_id_column
            ]
            + f"_{project_uid}",
            "location_source_id": brgi_db_mapping.Location.data[
                brgi_db_mapping.Location.location_id_column
            ],
            "project_uid": project_uid,
            "easting": brgi_db_mapping.Location.data[
                brgi_db_mapping.Location.easting_column
            ],
            "northing": brgi_db_mapping.Location.data[
                brgi_db_mapping.Location.northing_column
            ],
            "ground_level_elevation": brgi_db_mapping.Location.data[
                brgi_db_mapping.Location.ground_level_elevation_column
            ],
            "depth_to_base": brgi_db_mapping.Location.data[
                brgi_db_mapping.Location.depth_to_base_column
            ],
        }
    )
    location_df = pd.concat([location_df, brgi_db_mapping.Location.data], axis=1)
    LocationSchema.validate(location_df)

    insitu_tests = {}
    for insitu_mapping in brgi_db_mapping.InSitu:
        insitu_df = pd.DataFrame(
            {
                "project_uid": project_uid,
                "location_uid": insitu_mapping.data[insitu_mapping.location_id_column]
                + f"_{project_uid}",
                "depth_to_top": insitu_mapping.data[insitu_mapping.depth_to_top_column],
            }
        )
        if insitu_mapping.depth_to_base_column:
            insitu_df["depth_to_base"] = insitu_mapping.data[
                insitu_mapping.depth_to_base_column
            ]
        insitu_df = pd.concat([insitu_df, insitu_mapping.data], axis=1)
        InSituTestSchema.validate(insitu_df)
        insitu_tests[insitu_mapping.table_name] = insitu_df.copy()

    if brgi_db_mapping.Sample:
        sample_df = pd.DataFrame(
            {
                "sample_uid": brgi_db_mapping.Sample.data[
                    brgi_db_mapping.Sample.sample_id_column
                ]
                + f"_{project_uid}",
                "sample_source_id": brgi_db_mapping.Sample.data[
                    brgi_db_mapping.Sample.sample_id_column
                ],
                "project_uid": project_uid,
                "location_uid": brgi_db_mapping.Sample.data[
                    brgi_db_mapping.Sample.location_id_column
                ]
                + f"_{project_uid}",
                "depth_to_top": brgi_db_mapping.Sample.data[
                    brgi_db_mapping.Sample.depth_to_top_column
                ],
            }
        )
        if brgi_db_mapping.Sample.depth_to_base_column:
            sample_df["depth_to_base"] = brgi_db_mapping.Sample.data[
                brgi_db_mapping.Sample.depth_to_top_column
            ]
        sample_df = pd.concat([sample_df, brgi_db_mapping.Sample.data], axis=1)
        SampleSchema.validate(sample_df)

    brgi_lab_tests = {}
    if brgi_db_mapping.Lab:
        for lab_mapping in brgi_db_mapping.Lab:
            lab_df = pd.DataFrame(
                {
                    "project_uid": project_uid,
                    "sample_uid": lab_mapping.data[lab_mapping.sample_id_column]
                    + f"_{project_uid}",
                }
            )
            if lab_mapping.location_id_column:
                lab_df["location_uid"] = lab_mapping.data[
                    lab_mapping.location_id_column
                ]
            lab_df = pd.concat(
                [lab_df, lab_mapping.data.copy()],
                axis=1,
            )
            LabTestSchema.validate(lab_df)
            brgi_lab_tests[lab_mapping.table_name] = lab_df.copy()

    brgi_db = BedrockGIDatabase(
        Project=brgi_project,
        Location=pd.DataFrame(brgi_db_mapping.Location.data),
        InSituTests=insitu_tests,
        Sample=sample_df if brgi_db_mapping.Sample else None,
        LabTests=brgi_lab_tests,
        Other=(
            {
                other_table.table_name: pd.DataFrame(other_table.data)
                for other_table in brgi_db_mapping.Other
            }
            if brgi_db_mapping.Other
            else {}
        ),
    )
    return brgi_db
