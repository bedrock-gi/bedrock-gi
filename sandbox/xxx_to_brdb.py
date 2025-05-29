import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    from typing import Optional, Union

    import marimo as mo
    import pandas as pd
    from pydantic import BaseModel, Field, ConfigDict
    import pyproj
    return BaseModel, ConfigDict, Field, Optional, Union, pd, pyproj


@app.cell
def _(BaseModel, ConfigDict, Field, Optional, Union, pd, pyproj):
    class ProjectTableMapping(BaseModel):
        data: Optional[Union[dict, pd.DataFrame]] = None
        project_uid: str
        horizontal_crs: pyproj.CRS
        vertical_crs: pyproj.CRS = Field(default=pyproj.CRS(3855))
        # "compound_crs": Optional[CRS] = None # In case a

        model_config = ConfigDict(arbitrary_types_allowed=True)


    class LocationTableMapping(BaseModel):
        data: pd.DataFrame
        location_id_column: str
        easting_column: str
        northing_column: str
        ground_level_elevation_column: str
        depth_to_base_column: str

        model_config = ConfigDict(arbitrary_types_allowed=True)


    class SampleTableMapping(BaseModel):
        data: pd.DataFrame
        location_id_column: str
        sample_id_column: Union[str, list[str]]
        depth_to_top_column: str
        depth_to_base_column: Optional[str]

        model_config = ConfigDict(arbitrary_types_allowed=True)


    class OtherTable(BaseModel):
        table_name: str
        data: pd.DataFrame

        model_config = ConfigDict(arbitrary_types_allowed=True)


    class InSituTestTableMapping(OtherTable):
        location_id_column: str
        depth_to_top_column: str
        depth_to_base_column: Optional[str]


    class LabTestTableMapping(OtherTable):
        sample_id_column: Union[str, list[str]]


    class BedrockGIDatabaseMapping(BaseModel):
        Project: ProjectTableMapping
        Location: LocationTableMapping
        Sample: Optional[SampleTableMapping]
        InSitu: list[InSituTestTableMapping]
        Lab: Optional[list[LabTestTableMapping]]
        Other: Optional[list[OtherTable]]
    return


app._unparsable_cell(
    r"""
    ags3_brgi_db_mapping = BedrockGIDatabaseMapping(
        Project = ProjectTableMapping(
            data=
        ),
        Location = LocationTableMapping(
            data=
        ),
        Sample = SampleTableMapping(
            data=
        ),
        InSitu = 
    )
    """,
    name="_"
)


@app.cell
def _(pd):


    # The list of dictionaries that contains all the data to append one 
    # Is this the input model or what is this?
    ags4_brdb_model = [
        {
            "Project": {
                "data": pd.DataFrame,
                "project_uid_column": "PROJ_ID",
                "crs": 7415
            },
            "Location": {
                "data": pd.DataFrame,
                "location_id_column": "LOCA_ID",
                "easting_column": "LOCA_NATE",
                "northing_column": "LOCA_NATN",
                "ground_level_elevation_column": "LOCA_GL",
                "depth_to_base_column": "LOCA_FDEP"

            },
            "Sample": {
                "data": pd.DataFrame,
                "sample_id_column": ["SAMP_ID", "SAMP_REF", "SAMP_TYPE", "SAMP_TOP"]
            },
            "InSitu_XXXX": {

            },
            "Lab_XXXX": {
                "project_id": "x",
                "location_id": "loc",
            }
        }
    ]
    return


@app.cell
def _(pd):
    ags3_brdb_model = [
        {
            "Project": {
                "data": pd.DataFrame,
                "project_uid_column": "PROJ_ID",
                "crs": 7415
            },
            "Location": {
                "data": pd.DataFrame,
                "location_id_column": "HOLE_ID",
                "easting_column": "HOLE_NATE",
                "northing_column": "HOLE_NATN",
                "ground_level_elevation_column": "HOLE_GL",
                "depth_to_base_column": "HOLE_FDEP"
            },
            "Sample": {
                "data": pd.DataFrame,
                "sample_id_column": ["SAMP_ID", "SAMP_REF", "SAMP_TYPE", "SAMP_TOP"]
            },
            "InSitu_XXXX": {

            },
            "Lab_XXXX": {
                "project_id": "x",
                "location_id": "loc",
            }
        }
    ]
    return


@app.cell
def _(pd):
    wekahills_brdb_mapping = [
        {
            "Project": {
                "data": pd.DataFrame,
                "project_uid_column": "PROJ_ID", # hier nog over nadenken?!?!
                "crs": 7415
            },
            "Location": {
                "data": pd.DataFrame,
                "location_id_column": "LocationID",
                "easting_column": "Easting",
                "northing_column": "Northing",
                "ground_level_elevation_column": "GroundLevel",
                "depth_to_base_column": "FinalDepth"
            },
            "Sample": {
                "data": pd.DataFrame,
                "sample_id_column": ["SAMP_ID", "SAMP_REF", "SAMP_TYPE", "SAMP_TOP"]
            },
            "InSitu_XXXX": {
                "data": pd.DataFrame,
                "location_id_column": "LocationID",

            },
            "Lab_XXXX": {
                "project_id": "x",
                "location_id": "loc",
            }
        }
    ]
    return


if __name__ == "__main__":
    app.run()
