"""pandera schemas for Bedrock GI data. Base schemas refer to schemas that have no calculated GIS geometry or values."""

from typing import Optional

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series
from pydantic import BaseModel, ConfigDict


class ProjectSchema(pa.DataFrameModel):
    project_uid: Series[str] = pa.Field(
        # primary_key=True,
        unique=True,
    )
    horizontal_crs_wkt: Series[str] = pa.Field(
        description="Horizontal Coordinate Reference System (CRS) in Well-known Text (WKT) format."
    )
    vertical_crs_wkt: Series[str] = pa.Field(
        description="Vertical Coordinate Reference System (CRS) in Well-known Text (WKT) format."
    )


class LocationSchema(pa.DataFrameModel):
    location_uid: Series[str] = pa.Field(
        # primary_key=True,
        unique=True,
    )
    project_uid: Series[str] = pa.Field(
        # foreign_key="project.project_uid"
    )
    location_source_id: Series[str]
    easting: Series[float] = pa.Field(coerce=True)
    northing: Series[float] = pa.Field(coerce=True)
    ground_level_elevation: Series[float] = pa.Field(
        coerce=True,
        description="Elevation w.r.t. a local datum. Usually the orthometric height from the geoid, i.e. mean sea level, to the ground level.",
    )
    depth_to_base: Series[float]


class LonLatHeightSchema(pa.DataFrameModel):
    project_uid: Series[str] = pa.Field(
        # foreign_key="project.project_uid"
    )
    location_uid: Series[str] = pa.Field(
        # foreign_key="location.location_uid",
        unique=True,
    )
    longitude: Series[float]
    latitude: Series[float]
    egm2008_ground_level_height: Series[float] = pa.Field(
        description="Ground level orthometric height w.r.t. the EGM2008 (Earth Gravitational Model 2008).",
        nullable=True,
    )


class InSituTestSchema(pa.DataFrameModel):
    project_uid: Series[str] = pa.Field(
        # foreign_key="project.project_uid"
    )
    location_uid: Series[str] = pa.Field(
        # foreign_key="location.location_uid"
    )
    depth_to_top: Series[float] = pa.Field(coerce=True)
    depth_to_base: Optional[Series[float]] = pa.Field(coerce=True, nullable=True)


class SampleSchema(InSituTestSchema):
    sample_uid: Series[str] = pa.Field(
        # primary_key=True,
        unique=True,
    )
    sample_source_id: Series[str]


class LabTestSchema(pa.DataFrameModel):
    project_uid: Series[str] = pa.Field(
        # foreign_key="project.project_uid"
    )
    location_uid: Series[str] = pa.Field(
        # foreign_key="location.location_uid"
    )
    sample_uid: Series[str] = pa.Field(
        # foreign_key="sample.sample_uid"
    )


class BedrockGIDatabase(BaseModel):
    Project: pd.DataFrame
    Location: pd.DataFrame
    InSituTests: list[pd.DataFrame]
    Sample: Optional[pd.DataFrame] = None
    LabTests: Optional[list[pd.DataFrame]] = None
    Other: Optional[list[pd.DataFrame]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BedrockGIGeospatialDatabase(BedrockGIDatabase):
    LonLatHeight: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)
