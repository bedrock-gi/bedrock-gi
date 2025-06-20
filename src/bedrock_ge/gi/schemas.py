"""pandera schemas for Bedrock GI data. Base schemas refer to schemas that have no calculated GIS geometry or values."""

from typing import Optional

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
    depth_to_base: Series[float] = pa.Field(coerce=True, gt=0)


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
    depth_to_top: Optional[Series[float]] = pa.Field(nullable=True, coerce=True, ge=0)
    depth_to_base: Optional[Series[float]] = pa.Field(nullable=True, coerce=True, gt=0)

    # https://pandera.readthedocs.io/en/stable/dataframe_models.html#dataframe-checks
    # Check depth column completeness such that either shapely.Point's or
    # shapely.LineString's can be created.
    @pa.dataframe_check
    def depth_column_completeness(cls, df: pd.DataFrame) -> pd.Series:
        has_top = "depth_to_top" in df.columns
        has_base = "depth_to_base" in df.columns

        # If neither column exists, this check should fail
        if not has_top and not has_base:
            return pd.Series([False] * len(df), index=df.index)

        # If only one column exists, check that it's all non-null
        if has_top and not has_base:
            return df["depth_to_top"].notna()
        if has_base and not has_top:
            return df["depth_to_base"].notna()

        # If both columns exist:
        #   Either depth_to_top or depth_to_base must be non-null => Point
        #   OR
        #   Both depth_to_top and depth_to_base must be non-null => LineString
        # ! Commented out, because some In-Situ tests have a mix of
        # ! Point's and LineString's, such as IPRM
        # top_has_value = df["depth_to_top"].notna()
        # base_has_value = df["depth_to_base"].notna()
        # either_has_value = top_has_value ^ base_has_value
        # both_have_values = top_has_value & base_has_value

        # if either_has_value.all():
        #     return either_has_value
        # elif both_have_values.all():
        #     return both_have_values
        # else:
        #     if either_has_value.sum() < both_have_values.sum():
        #         return either_has_value
        #     else:
        #         return both_have_values

        # ! Incorrect check
        # If both columns exist, at least one must be non-null
        return ~(df["depth_to_top"].isna() & df["depth_to_base"].isna())

    @pa.dataframe_check
    def top_above_base(cls, df: pd.DataFrame) -> pd.Series:
        """Check that depth_to_top <= depth_to_base when both columns are present.

        If either column is missing, this check passes (nothing to compare).
        If both columns are present, the check fails if any row has
        depth_to_top > depth_to_base.

        Returns:
            pd.Series: pandas.Series of bools indicating successful checks.
        """
        has_top = "depth_to_top" in df.columns
        has_base = "depth_to_base" in df.columns

        # If either column is missing, this check passes (nothing to compare)
        if not has_top or not has_base:
            return pd.Series([True] * len(df), index=df.index)

        # Only compare when both values are non-null
        mask = df["depth_to_top"].notna() & df["depth_to_base"].notna()
        # Use where() to conditionally apply the comparison
        result = (~mask) | (df["depth_to_top"] <= df["depth_to_base"])

        # Debug: Show failing cases
        failing_mask = mask & ~result
        if failing_mask.any():
            print("🚨 ERROR: depth_to_top > depth_to_base:")
            print(
                df.loc[
                    failing_mask,
                    ["location_uid", "depth_to_top", "depth_to_base", df.columns[5]],
                ]
            )

        return result


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
    Sample: pd.DataFrame | None = None
    LabTests: dict[str, pd.DataFrame] = {}
    Other: dict[str, pd.DataFrame] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BedrockGIGeospatialDatabase(BaseModel):
    Project: pd.DataFrame
    Location: gpd.GeoDataFrame
    LonLatHeight: gpd.GeoDataFrame
    InSituTests: dict[str, gpd.GeoDataFrame]
    Sample: gpd.GeoDataFrame | None = None
    LabTests: dict[str, pd.DataFrame] = {}
    Other: dict[str, pd.DataFrame] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)
