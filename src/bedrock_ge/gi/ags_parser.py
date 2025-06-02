from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd
from pyproj import CRS

from bedrock_ge.gi.ags3 import ags3_to_brgi_db_mapping
from bedrock_ge.gi.io_utils import detect_encoding, open_text_data_source
from bedrock_ge.gi.mapping_models import BedrockGIMapping


def ags_to_brgi_db_mapping(
    source: str | Path | IO[str] | IO[bytes] | bytes,
    projected_crs: CRS,
    vertical_crs: CRS = CRS(3855),
    encoding: str | None = None,
) -> BedrockGIMapping:
    """Map AGS 3 or AGS 4 data to the Bedrock GI data model.

    Args:
        source (str | Path | IO[str] | IO[bytes] | bytes): The AGS file (str or Path)
            or a file-like object that represents the AGS file.
        projected_crs (CRS): Projected Coordinate Reference System (CRS). For example:
            - OSGB36 / British National Grid: `pyproj.CRS("EPSG:27700")`
            - Hong Kong 1980 Grid System: `pyproj.CRS("EPSG:2326")`
        vertical_crs (CRS, optional): Vertical CRS. Defaults to EGM2008 height, EPSG:3855
            which measures the orthometric height w.r.t. the Earth Gravitational Model 2008.
            - Ordnance Datum Newlyn (ODN) Height: `pyproj.CRS("EPSG:5701")`
            - Hong Kong Principle Datum (HKPD) Height: `pyproj.CRS("EPSG:5738")`
        encoding (str | None, optional): Encoding of the text file or bytes stream.
            Defaults to None. An attempt at detecting the encoding will be made if None.

    Raises:
        ValueError: If the data does not match AGS 3 or AGS 4 format.

    Returns:
        BedrockGIDatabaseMapping: Object that maps AGS 3 or AGS 4 data to Bedrock GI data model.
    """
    if not encoding:
        encoding = detect_encoding(source)

    # Get first non-blank line, None if all lines are blank
    with open_text_data_source(source, encoding=encoding) as f:
        first_line = next((line.strip() for line in f if line.strip()), None)

    if first_line:
        if first_line.startswith('"**'):
            ags_version = 3
            brgi_db_mapping = ags3_to_brgi_db_mapping(
                source, projected_crs, vertical_crs, encoding
            )
        elif first_line.startswith('"GROUP"'):
            ags_version = 4
            # brgi_db_mapping = ags4_to_brgi_db_mapping(
            #     source, projected_crs, vertical_crs, encoding
            # )
        else:
            # If first non-empty line doesn't match AGS 3 or AGS 4 format
            raise ValueError("The data provided is not valid AGS 3 or AGS 4 data.")
    else:
        raise ValueError("The file provided has only blank lines")

    # Log information about the mapped AGS 3 or AGS 4 data
    project_uid = brgi_db_mapping.Project.project_id
    n_gi_locations = len(brgi_db_mapping.Location.data)
    n_samples = len(brgi_db_mapping.Sample.data) if brgi_db_mapping.Sample else 0
    print_args = [
        f"AGS {ags_version} data was read for Project {project_uid}",
        f"This GI data contains {n_gi_locations} GI locations, {n_samples} samples and:",
        f"  - In-Situ Tests: {[insitu_test.table_name for insitu_test in brgi_db_mapping.InSitu]}",
    ]
    if brgi_db_mapping.Lab:
        print_args.append(
            f"  - Lab Tests: {[lab_test.table_name for lab_test in brgi_db_mapping.Lab]}"
        )
    if brgi_db_mapping.Other:
        print_args.append(
            f"  - Other Tables: {[other_table.table_name for other_table in brgi_db_mapping.Other]}"
        )
    print(*print_args, sep="\n", end="\n\n")

    return brgi_db_mapping
