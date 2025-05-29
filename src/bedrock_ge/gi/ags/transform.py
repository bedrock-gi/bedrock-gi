"""Transforms, i.e. maps, AGS data to Bedrock's schema."""

from typing import Dict

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame
from pyproj import CRS

from bedrock_ge.gi.ags.schemas import Ags3HOLE, Ags3SAMP, BaseSAMP
from bedrock_ge.gi.schemas import (
    InSituTestSchema,
    LocationSchema,
    ProjectSchema,
    SampleSchema,
)
from bedrock_ge.gi.validate import check_foreign_key


# What this function really does, is add the CRS and Bedrock columns:
# - `project_uid`
# - `location_uid`
# - `sample_id`
# - `sample_uid`
# - `depth_to_`
# There really isn't any mapping going on here...
# TODO: Make sure that the name of the function and docstrings reflect this.
def ags3_db_to_no_gis_brgi_db(
    ags3_db: Dict[str, pd.DataFrame], crs: CRS
) -> Dict[str, pd.DataFrame]:
    """Maps a database with GI data from a single AGS 3 file to a database with Bedrock's schema.

    This function converts an AGS 3 formatted geotechnical database into Bedrock's
    internal database format, maintaining data relationships and structure. It handles
    various types of geotechnical data including project information, locations,
    samples, lab tests, and in-situ measurements.

    The mapping process:
    1. Project Data: Converts AGS 3 'PROJ' group to Bedrock's 'Project' table
    2. Location Data: Converts AGS 3 'HOLE' group to Bedrock's 'Location' table
    3. Sample Data: Converts AGS 3 'SAMP' group to Bedrock's 'Sample' table
    4. Other Data: Handles lab tests, in-situ measurements, and miscellaneous tables

    Args:
        ags3_db (Dict[str, pd.DataFrame]): A dictionary containing AGS 3 data tables,
            where keys are table names and values are pandas DataFrames.
        crs (CRS): Coordinate Reference System for the project data.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing Bedrock GI database tables,
            where keys are table names and values are transformed pandas DataFrames.

    Note:
        The function creates a copy of the input database to avoid modifying the original data.
        It performs foreign key checks to maintain data integrity during the mapping.
    """
    # Make sure that the AGS 3 database is not changed outside this function.
    ags3_db = ags3_db.copy()

    print("Transforming AGS 3 groups to Bedrock tables...")

    # Instantiate Bedrock dictionary of pd.DataFrames
    brgi_db = {}

    # Project
    print("Transforming AGS 3 group 'PROJ' to Bedrock GI 'Project' table...")
    brgi_db["Project"] = ags_proj_to_brgi_project(ags3_db["PROJ"], crs)
    project_uid = brgi_db["Project"]["project_uid"].item()
    del ags3_db["PROJ"]

    # Locations
    if "HOLE" in ags3_db.keys():
        print("Transforming AGS 3 group 'HOLE' to Bedrock GI 'Location' table...")
        brgi_db["Location"] = ags3_hole_to_brgi_location(ags3_db["HOLE"], project_uid)  # type: ignore
        del ags3_db["HOLE"]
    else:
        print(
            "Your AGS 3 data doesn't contain a HOLE group, i.e. Ground Investigation locations."
        )

    # Samples
    if "SAMP" in ags3_db.keys():
        print("Transforming AGS 3 group 'SAMP' to Bedrock GI 'Sample' table...")
        check_foreign_key("HOLE_ID", brgi_db["Location"], ags3_db["SAMP"])
        ags3_db["SAMP"] = generate_sample_ids_for_ags3(ags3_db["SAMP"])  # type: ignore
        brgi_db["Sample"] = ags3_samp_to_brgi_sample(ags3_db["SAMP"], project_uid)  # type: ignore
        del ags3_db["SAMP"]
    else:
        print("Your AGS 3 data doesn't contain a SAMP group, i.e. samples.")

    # The rest of the tables: 1. Lab Tests 2. In-Situ Measurements 3. Other tables
    for group, group_df in ags3_db.items():
        if "SAMP_REF" in ags3_db[group].columns:
            print(f"Project {project_uid} has lab test data: {group}.")
            brgi_db[group] = group_df  # type: ignore
        elif "HOLE_ID" in ags3_db[group].columns:
            print(
                f"Transforming AGS 3 group '{group}' to Bedrock GI 'InSitu_{group}' table..."
            )
            check_foreign_key("HOLE_ID", brgi_db["Location"], group_df)
            brgi_db[f"InSitu_{group}"] = ags3_in_situ_to_brgi_in_situ(  # type: ignore
                group, group_df, project_uid
            )
        else:
            brgi_db[group] = ags3_db[group]  # type: ignore

    print(
        "Done",
        "The Bedrock database contains the following tables:",
        list(brgi_db.keys()),
        sep="\n",
        end="\n\n",
    )
    return brgi_db  # type: ignore


@pa.check_types(lazy=True)
def ags_proj_to_brgi_project(
    ags_proj: pd.DataFrame, horizontal_crs: CRS, vertical_crs: CRS = CRS(3855)
) -> DataFrame[ProjectSchema]:
    """Maps the AGS 3 'PROJ' group to a Bedrock GI 'Project' table.

    Args:
        ags_proj (pd.DataFrame): The AGS 3 'PROJ' group.
        crs (CRS): The coordinate reference system of the project.

    Returns:
        DataFrame[Project]: The Bedrock GI 'Project' table.
    """
    if "project_uid" not in ags_proj.columns:
        ags_proj["project_uid"] = ags_proj["PROJ_ID"]

    ags_proj["horizontal_crs"] = horizontal_crs.to_string()
    ags_proj["horizontal_crs_wkt"] = horizontal_crs.to_wkt()
    ags_proj["vertical_crs"] = vertical_crs.to_string()
    ags_proj["vertical_crs_wkt"] = vertical_crs.to_wkt()

    return ags_proj  # type: ignore


@pa.check_types(lazy=True)
def ags3_hole_to_brgi_location(
    ags3_hole: DataFrame[Ags3HOLE], project_uid: str
) -> DataFrame[LocationSchema]:
    brgi_location = ags3_hole
    brgi_location["project_uid"] = project_uid
    brgi_location["location_source_id"] = ags3_hole["HOLE_ID"]
    brgi_location["location_uid"] = (
        ags3_hole["HOLE_ID"] + "_" + ags3_hole["project_uid"]
    )
    brgi_location["location_type"] = ags3_hole["HOLE_TYPE"]
    brgi_location["easting"] = ags3_hole["HOLE_NATE"]
    brgi_location["northing"] = ags3_hole["HOLE_NATN"]
    brgi_location["ground_level_elevation"] = ags3_hole["HOLE_GL"]
    brgi_location["depth_to_base"] = ags3_hole["HOLE_FDEP"]

    return ags3_hole  # type: ignore


@pa.check_types(lazy=True)
def ags3_samp_to_brgi_sample(
    ags3_samp: DataFrame[Ags3SAMP],
    project_uid: str,
) -> DataFrame[SampleSchema]:
    brgi_sample = ags3_samp
    brgi_sample["project_uid"] = project_uid
    brgi_sample["location_source_id"] = ags3_samp["HOLE_ID"]
    brgi_sample["location_uid"] = ags3_samp["HOLE_ID"] + "_" + ags3_samp["project_uid"]
    brgi_sample["sample_source_id"] = ags3_samp["sample_id"]
    brgi_sample["sample_uid"] = ags3_samp["sample_id"] + "_" + ags3_samp["project_uid"]
    brgi_sample["depth_to_top"] = ags3_samp["SAMP_TOP"]
    brgi_sample["depth_to_base"] = ags3_samp["SAMP_BASE"]

    return brgi_sample  # type: ignore


@pa.check_types(lazy=True)
def ags3_in_situ_to_brgi_in_situ(
    group_name: str, ags3_in_situ: pd.DataFrame, project_uid: str
) -> DataFrame[InSituTestSchema]:
    """Maps AGS 3 in-situ measurement data to Bedrock's in-situ data schema.

    Args:
        group_name (str): The AGS 3 group name.
        ags3_data (pd.DataFrame): The AGS 3 data.
        project_uid (str): The project uid.

    Returns:
        DataFrame[BaseInSitu]: The Bedrock in-situ data.
    """
    brgi_in_situ = ags3_in_situ
    brgi_in_situ["project_uid"] = project_uid
    brgi_in_situ["location_uid"] = ags3_in_situ["HOLE_ID"] + "_" + project_uid

    top_depth = f"{group_name}_TOP"
    base_depth = f"{group_name}_BASE"

    if group_name == "CDIA":
        top_depth = "CDIA_CDEP"
    elif group_name == "FLSH":
        top_depth = "FLSH_FROM"
        base_depth = "FLSH_TO"
    elif group_name == "CORE":
        base_depth = "CORE_BOT"
    elif group_name == "HDIA":
        top_depth = "HDIA_HDEP"
    elif group_name == "PTIM":
        top_depth = "PTIM_DEP"
    elif group_name == "IVAN":
        top_depth = "IVAN_DPTH"
    elif group_name == "STCN":
        top_depth = "STCN_DPTH"
    elif group_name == "POBS" or group_name == "PREF":
        top_depth = "PREF_TDEP"
    elif group_name == "DREM":
        top_depth = "DREM_DPTH"
    elif group_name == "PRTD" or group_name == "PRTG" or group_name == "PRTL":
        top_depth = "PRTD_DPTH"
    elif group_name == "IPRM":
        if top_depth not in ags3_in_situ.columns:
            print(
                "\n🚨 CAUTION: The IPRM group in this AGS 3 file does not contain a 'IPRM_TOP' heading!",
                "🚨 CAUTION: Making the 'IPRM_BASE' heading the 'depth_to_top'...",
                sep="\n",
                end="\n\n",
            )
            top_depth = "IPRM_BASE"
            base_depth = "None"

    brgi_in_situ["depth_to_top"] = ags3_in_situ[top_depth]
    brgi_in_situ["depth_to_base"] = ags3_in_situ.get(base_depth)

    return brgi_in_situ  # type: ignore


@pa.check_types(lazy=True)
def generate_sample_ids_for_ags3(
    ags3_with_samp: DataFrame[BaseSAMP],
) -> DataFrame[Ags3SAMP]:
    ags3_with_samp["sample_id"] = (
        ags3_with_samp["SAMP_REF"].astype(str)
        + "_"
        + ags3_with_samp["SAMP_TYPE"].astype(str)
        + "_"
        + ags3_with_samp["SAMP_TOP"].astype(str)
        + "_"
        + ags3_with_samp["HOLE_ID"].astype(str)
    )
    # try:
    #     # SAMP_REF really should not be able to be null... Right?
    #     # Maybe SAMP_REF can be null when the
    #     Ags3SAMP_REF.validate(ags3_samp)
    #     print(
    #         "Generating unique sample IDs for AGS 3 data: 'sample_id'='{SAMP_REF}_{HOLE_ID}'"
    #     )
    #     ags3_samp["sample_id"] = (
    #         ags3_samp["SAMP_REF"].astype(str) + "_" + ags3_samp["HOLE_ID"].astype(str)
    #     )
    # except pa.errors.SchemaError as exc:
    #     print(f"🚨 CAUTION: The AGS 3 SAMP group contains rows without SAMP_REF:\n{exc}")

    #     if "non-nullable series 'SAMP_REF'" in str(exc):
    #         print(
    #             "\nTo ensure unique sample IDs: 'sample_id'='{SAMP_REF}_{SAMP_TOP}_{HOLE_ID}'\n"
    #         )
    #         ags3_samp["sample_id"] = (
    #             ags3_samp["SAMP_REF"].astype(str)
    #             + "_"
    #             + ags3_samp["SAMP_TOP"].astype(str)
    #             + "_"
    #             + ags3_samp["HOLE_ID"].astype(str)
    #         )

    return ags3_with_samp  # type: ignore
