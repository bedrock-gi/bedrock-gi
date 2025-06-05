from typing import Optional, Union

import pandas as pd
import pyproj
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProjectTableMapping(BaseModel):
    data: dict = {}
    project_id: str
    horizontal_crs: pyproj.CRS
    vertical_crs: pyproj.CRS = Field(default=pyproj.CRS(3855))
    # "compound_crs": Optional[CRS] = None

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
    sample_id_column: str
    location_id_column: str
    depth_to_top_column: str
    depth_to_base_column: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class OtherTable(BaseModel):
    table_name: str
    data: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)


class InSituTestTableMapping(OtherTable):
    location_id_column: str
    depth_to_top_column: Optional[str] = None
    depth_to_base_column: Optional[str] = None

    @model_validator(mode="after")
    def validate_at_least_one_depth_column(self):
        if not (self.depth_to_top_column or self.depth_to_base_column):
            raise ValueError("At least one depth column must be specified.")
        return self


class LabTestTableMapping(OtherTable):
    sample_id_column: str
    location_id_column: Optional[str] = None


class BedrockGIMapping(BaseModel):
    Project: ProjectTableMapping
    Location: LocationTableMapping
    InSitu: list[InSituTestTableMapping]
    Sample: Union[SampleTableMapping, None] = None
    Lab: list[LabTestTableMapping] = []
    Other: list[OtherTable] = []
