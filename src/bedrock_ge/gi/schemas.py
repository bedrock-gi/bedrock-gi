"""pandera schemas for Bedrock GI data. Base schemas refer to schemas that have no calculated GIS geometry or values."""

from typing import Optional, Union

import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from pydantic import BaseModel, ConfigDict


class ProjectSchema(pa.DataFrameModel):
    project_uid: Series[str] = pa.Field(
        # primary_key=True,
        unique=True,
    )
    horizontal_crs: Series[str] = pa.Field(
        description="Horizontal Coordinate Reference System (CRS)."
    )
    horizontal_crs_wkt: Series[str] = pa.Field(
        description="Horizontal CRS in Well-known Text (WKT) format."
    )
    vertical_crs: Series[str] = pa.Field(
        description="Vertical Coordinate Reference System (CRS)."
    )
    vertical_crs_wkt: Series[str] = pa.Field(
        description="Vertical CRS in Well-known Text (WKT) format."
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
    InSituTests: dict[str, pd.DataFrame]
    Sample: Union[pd.DataFrame, None] = None
    LabTests: dict[str, pd.DataFrame] = {}
    Other: dict[str, pd.DataFrame] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BedrockGIGeospatialDatabase(BaseModel):
    Project: pd.DataFrame
    Location: gpd.GeoDataFrame
    LonLatHeight: gpd.GeoDataFrame
    InSituTests: dict[str, gpd.GeoDataFrame]
    Sample: Union[gpd.GeoDataFrame, None] = None
    LabTests: dict[str, pd.DataFrame] = {}
    Other: dict[str, pd.DataFrame] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)
