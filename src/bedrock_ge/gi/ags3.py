from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import pandas as pd
from pyproj import CRS

from bedrock_ge.gi.ags_schemas import Ags3HOLE, Ags3SAMP, check_ags_proj_group
from bedrock_ge.gi.io_utils import coerce_string, open_text_data_source
from bedrock_ge.gi.mapping_models import (
    BedrockGIMapping,
    InSituTestTableMapping,
    LabTestTableMapping,
    LocationTableMapping,
    OtherTable,
    ProjectTableMapping,
    SampleTableMapping,
)


def ags3_to_dfs(
    source: str | Path | IO[str] | IO[bytes] | bytes, encoding: str
) -> dict[str, pd.DataFrame]:
    """Converts AGS 3 data to a dictionary of pandas DataFrames.

    Also strips '?' from non-standard AGS 3 group and header names, in order to
    make the rest of the code more generic.

    Args:
        source (str | Path | IO[str] | IO[bytes] | bytes): The AGS 3 file (str or Path)
            or a file-like object that represents the AGS 3 file.
        encoding (str):  Encoding of file or object.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary of pandas DataFrames, i.e. a database,
            where each key is an AGS 3 group, and the corresponding value is
            a pandas DataFrame containing the data for that group.
    """
    # Initialize dictionary and variables used in the AGS 3 read loop
    ags3_dfs = {}
    line_type = "line_0"
    group = ""
    headers: list[str] = ["", "", ""]
    group_data: list[list[Any]] = [[], [], []]

    with open_text_data_source(source, encoding=encoding) as file:
        for i, line in enumerate(file):
            line = line.strip()
            last_line_type = line_type

            # In AGS 3.1 group names are prefixed with **
            if line.startswith('"**'):
                line_type = "group_name"
                if group:
                    ags3_dfs[group] = pd.DataFrame(group_data, columns=headers)

                group = line.strip(' ,"*?')
                group_data = []

            # In AGS 3 header names are prefixed with "*
            elif line.startswith('"*'):
                line_type = "headers"
                new_headers = line.split('","')
                new_headers = [h.strip(' ,"*?') for h in new_headers]

                # Some groups have so many headers that they span multiple lines.
                # Therefore we need to check whether the new headers are
                # a continuation of the previous headers from the last line.
                if line_type == last_line_type:
                    headers = headers + new_headers
                else:
                    headers = new_headers

            # Skip lines where group units are defined, these are defined in the AGS 3 data dictionary.
            elif line.startswith('"<UNITS>"'):
                line_type = "units"
                continue

            # The rest of the lines contain:
            # 1. GI data
            # 2. a continuation of the previous line. These lines contain "<CONT>" in the first column.
            # 3. are empty or contain worthless data
            else:
                line_type = "data_row"
                data_row = line.split('","')
                if len("".join(data_row)) == 0:
                    # print(f"Line {i} is empty. Last Group: {group}")
                    continue
                elif len(data_row) != len(headers):
                    print(
                        f"\n🚨 CAUTION: The number of columns ({len(data_row)}) on line {i + 1} doesn't match the number of columns ({len(headers)}) of group {group}!",
                        f"{group} headers: {headers}",
                        f"Line {i + 1}:      {data_row}",
                        sep="\n",
                        end="\n\n",
                    )
                    continue
                # Append continued lines (<CONT>) to the last data_row
                elif data_row[0] == '"<CONT>':
                    last_data_row = group_data[-1]
                    for j, data in enumerate(data_row):
                        data = data.strip(' "')
                        if data and data != "<CONT>":
                            if last_data_row[j] is None:
                                # Last data row didn't contain data for this column
                                last_data_row[j] = coerce_string(data)
                            else:
                                # Last data row already contains data for this column
                                last_data_row[j] = str(last_data_row[j]) + data
                # Lines that are assumed to contain valid data are added to the group data
                else:
                    cleaned_data_row = []
                    for data in data_row:
                        cleaned_data_row.append(coerce_string(data.strip(' "')))
                    group_data.append(cleaned_data_row)

    # Also add the last group's df to the dictionary of AGS dfs
    ags3_dfs[group] = pd.DataFrame(group_data, columns=headers)

    if not group:
        print(
            '🚨 ERROR: The provided AGS 3 data does not contain any groups, i.e. lines starting with "**'
        )

    return ags3_dfs


# TODO: AGS 3 table validation based on the AGS 3 data dictionary.
def ags3_to_brgi_db_mapping(
    source: str | Path | IO[str] | IO[bytes] | bytes,
    projected_crs: CRS,
    vertical_crs: CRS,
    encoding: str,
) -> BedrockGIMapping:
    """Map AGS 3 data to the Bedrock GI data model.

    Args:
        ags3_db (dict[str, pd.DataFrame]): A dictionary of pandas DataFrames, i.e. database,
            where each key is an AGS 3 group, and the corresponding value is
            a pandas DataFrame containing the data for that group.
        projected_crs (CRS): Projected coordinate reference system (CRS).
        vertical_crs (CRS, optional): Vertical CRS. Defaults to EGM2008 height, EPSG:3855
            which measures the orthometric height w.r.t. the Earth Gravitational Model 2008.
        encoding (str): Encoding of the text file or bytes stream.

    Returns:
        BedrockGIDatabaseMapping: Object that maps AGS 3 data to Bedrock GI data model.
    """
    ags3_dfs = ags3_to_dfs(source, encoding)

    check_ags_proj_group(ags3_dfs["PROJ"])
    ags3_project = ProjectTableMapping(
        data=ags3_dfs["PROJ"].to_dict(orient="records")[0],
        project_id=ags3_dfs["PROJ"]["PROJ_ID"].iloc[0],
        horizontal_crs=projected_crs,
        vertical_crs=vertical_crs,
    )
    del ags3_dfs["PROJ"]

    Ags3HOLE.validate(ags3_dfs["HOLE"])
    ags3_location = LocationTableMapping(
        data=ags3_dfs["HOLE"],
        location_id_column="HOLE_ID",
        easting_column="HOLE_NATE",
        northing_column="HOLE_NATN",
        ground_level_elevation_column="HOLE_GL",
        depth_to_base_column="HOLE_FDEP",
    )
    del ags3_dfs["HOLE"]

    if "SAMP" in ags3_dfs.keys():
        Ags3SAMP.validate(ags3_dfs["SAMP"])
        samp_df = ags3_dfs["SAMP"]
        samp_df = _add_sample_source_id(samp_df)
        ags3_sample = SampleTableMapping(
            data=samp_df,
            location_id_column="HOLE_ID",
            sample_id_column="sample_source_id",
            depth_to_top_column="SAMP_TOP",
        )
        del ags3_dfs["SAMP"]
    else:
        ags3_sample = None

    ags3_lab_tests = []
    ags3_insitu_tests = []
    ags3_other_tables = []

    for group, df in ags3_dfs.items():
        # Non-standard group names contain the "?" prefix.
        # => checking that "SAMP_TOP" / "HOLE_ID" is in the columns is too restrictive.
        if "SAMP_TOP" in df.columns:
            df = _add_sample_source_id(df)
            ags3_lab_tests.append(
                LabTestTableMapping(
                    table_name=group,
                    data=df,
                    location_id_column="HOLE_ID",
                    sample_id_column="sample_source_id",
                )
            )
        elif "HOLE_ID" in df.columns:
            top_depth, base_depth = _get_depth_columns(group, list(df.columns))
            ags3_insitu_tests.append(
                InSituTestTableMapping(
                    table_name=group,
                    data=df,
                    location_id_column="HOLE_ID",
                    depth_to_top_column=top_depth,
                    depth_to_base_column=base_depth,
                )
            )
        else:
            ags3_other_tables.append(OtherTable(table_name=group, data=df))

    brgi_db_mapping = BedrockGIMapping(
        Project=ags3_project,
        Location=ags3_location,
        InSitu=ags3_insitu_tests,
        Sample=ags3_sample,
        Lab=ags3_lab_tests,
        Other=ags3_other_tables,
    )
    return brgi_db_mapping


def _add_sample_source_id(df: pd.DataFrame) -> pd.DataFrame:
    df["sample_source_id"] = (
        df["SAMP_REF"].astype(str)
        + "-"
        + df["SAMP_TYPE"].astype(str)
        + "-"
        + df["SAMP_TOP"].astype(str)
        + "-"
        + df["HOLE_ID"].astype(str)
    )
    return df


def _get_depth_columns(group: str, headers: list[str]) -> tuple[str, str | None]:
    top_depth = f"{group}_TOP"
    base_depth: str | None = f"{group}_BASE"

    if group == "CDIA":
        top_depth = "CDIA_CDEP"
    elif group == "FLSH":
        top_depth = "FLSH_FROM"
        base_depth = "FLSH_TO"
    elif group == "CORE":
        base_depth = "CORE_BOT"
    elif group == "HDIA":
        top_depth = "HDIA_HDEP"
    elif group == "PTIM":
        top_depth = "PTIM_DEP"
    elif group == "IVAN":
        top_depth = "IVAN_DPTH"
    elif group == "STCN":
        top_depth = "STCN_DPTH"
    elif group == "POBS" or group == "PREF":
        top_depth = "PREF_TDEP"
    elif group == "DREM":
        top_depth = "DREM_DPTH"
    elif group == "PRTD" or group == "PRTG" or group == "PRTL":
        top_depth = "PRTD_DPTH"
    elif group == "IPRM":
        if top_depth not in headers:
            print(
                "\n🚨 CAUTION: The IPRM group in this AGS 3 file does not contain a 'IPRM_TOP' heading!",
                "🚨 CAUTION: Making the 'IPRM_BASE' heading the 'depth_to_top'...",
                sep="\n",
                end="\n\n",
            )
            top_depth = "IPRM_BASE"
            base_depth = None

    if top_depth not in headers:
        raise KeyError(
            f"The {group} group in this AGS 3 file does not contain a '{top_depth}' heading!"
        )

    if base_depth not in headers:
        base_depth = None

    return top_depth, base_depth
