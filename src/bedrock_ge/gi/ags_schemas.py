import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


def check_ags_proj_group(ags_proj: pd.DataFrame) -> bool:
    """Checks if the AGS 3 or AGS 4 PROJ group is correct.

    Args:
        ags_proj (pd.DataFrame): The DataFrame with the PROJ group.

    Raises:
        ValueError: If AGS 3 of AGS 4 PROJ group is not correct.

    Returns:
        bool: Returns True if the AGS 3 or AGS 4 PROJ group is correct.
    """
    if len(ags_proj) != 1:
        raise ValueError("The PROJ group must contain exactly one row.")

    msg = 'The project ID ("PROJ_ID" in the "PROJ" group) is missing from the AGS data.'
    try:
        project_id = ags_proj.at[ags_proj.index[0], "PROJ_ID"]
    except KeyError:
        raise ValueError(msg)

    if pd.isna(project_id) or str(project_id).strip() == "":
        raise ValueError(msg)

    return True


class Ags3HOLE(pa.DataFrameModel):
    HOLE_ID: Series[str] = pa.Field(
        # primary_key=True,
        unique=True,
        coerce=True,
        description="Exploratory hole or location equivalent",
        # example="327/16A",
    )
    HOLE_TYPE: Series[str] = pa.Field(
        coerce=True,
        # isin=["CP", "TP", "TPS", "TPS2", "TPS3", "TPS4", "TPS5", "TPS6", "TPS7", "TPS8"],
        description="Type of exploratory hole",
        # example="CP (See Appendix 1)",
    )
    HOLE_NATE: Series[float] = pa.Field(coerce=True)
    HOLE_NATN: Series[float] = pa.Field(coerce=True)
    HOLE_GL: Series[float] = pa.Field(coerce=True)
    HOLE_FDEP: Series[float] = pa.Field(
        coerce=True,
        description="Final depth of hole",
        # example=32.60,
        metadata={"unit": "m"},
    )


class BaseSAMP(pa.DataFrameModel):
    SAMP_REF: Series[str] = pa.Field(
        coerce=True,
        nullable=True,
        description="Sample reference number",
        # example="24",
    )
    SAMP_TYPE: Series[str] = pa.Field(
        coerce=True,
        nullable=True,
        description="Sample type",
        # example="U (See Appendix 1)",
    )
    SAMP_TOP: Series[float] = pa.Field(
        coerce=True,
        description="Depth to TOP of sample",
        # example=24.55,
        metadata={"unit": "m"},
    )
    SAMP_BASE: Series[float] = pa.Field(
        coerce=True,
        nullable=True,
        description="Depth to BASE of sample",
        # example=24.55,
        metadata={"unit": "m"},
    )


class Ags3SAMP(BaseSAMP):
    HOLE_ID: Series[str] = pa.Field(
        # foreign_key="Ags3HOLE.HOLE_ID",
        description="Exploratory hole or location equivalent",
        # example="327/16A",
    )


class Ags4SAMP(BaseSAMP):
    SAMP_ID: Series[str] = pa.Field(
        # primary_key=True,
        unique=True,
        coerce=True,
        description="Sample unique identifier",
        # example="ABC121415010",
    )
    LOCA_ID: Series[str] = pa.Field(
        # foreign_key="Ags4LOCA.LOCA_ID",
        coerce=True,
        description="Location identifier",
        # example="327/16A",
    )


class BaseGEOL(pa.DataFrameModel):
    GEOL_TOP: Series[float] = pa.Field(
        coerce=True,
        description="Depth to the top of stratum",
        # example=16.21,
        metadata={"unit": "m"},
    )
    GEOL_BASE: Series[float] = pa.Field(
        coerce=True,
        description="Depth to the base of description",
        # example=17.25,
        metadata={"unit": "m"},
    )
    GEOL_DESC: Series[str] = pa.Field(
        coerce=True,
        description="General description of stratum",
        # example="Stiff grey silty CLAY",
    )
    GEOL_LEG: Series[str] = pa.Field(
        nullable=True,
        description="Legend code",
        # example="102",
    )
    GEOL_GEOL: Series[str] = pa.Field(
        coerce=True,
        description="Geology code",
        # example="LC",
    )
    GEOL_GEO2: Series[str] = pa.Field(
        coerce=True,
        nullable=True,
        description="Second geology code",
        # example="SAND",
    )


class Ags3GEOL(BaseGEOL):
    HOLE_ID: Series[str] = pa.Field(
        # foreign_key="Ags3HOLE.HOLE_ID",
        coerce=True,
        description="Exploratory hole or location equivalent",
        # example="6421/A",
    )


class Ags4GEOL(BaseGEOL):
    LOCA_ID: Series[str] = pa.Field(
        # foreign_key="Ags4LOCA.LOCA_ID",
        coerce=True,
        description="Location identifier",
        # example="327/16A",
    )


class BaseISPT(pa.DataFrameModel):
    ISPT_TOP: Series[float] = pa.Field(
        coerce=True,
        description="Depth to top of test",
        # example=13.50,
        metadata={"unit": "m"},
    )
    ISPT_NVAL: Series[int] = pa.Field(
        coerce=True,
        description="Depth to the base of description",
        # example=35,
        ge=0,
    )


class Ags3ISPT(BaseISPT):
    HOLE_ID: Series[str] = pa.Field(
        # foreign_key="Ags3HOLE.HOLE_ID",
        coerce=True,
        description="Exploratory hole or location equivalent",
        # example="6421/A",
    )


class Ags4ISPT(BaseISPT):
    LOCA_ID: Series[str] = pa.Field(
        # foreign_key="Ags4LOCA.LOCA_ID",
        coerce=True,
        description="Location identifier",
        # example="327/16A",
    )


class BaseCORE(pa.DataFrameModel):
    CORE_TOP: Series[float] = pa.Field(
        coerce=True,
        description="Depth to TOP of core run",
        # example=2.54,
        metadata={"unit": "m"},
    )
    CORE_PREC: Series[int] = pa.Field(
        coerce=True,
        nullable=True,
        description="Percentage of core recovered in core run (TCR)",
        # example="32",
        metadata={"unit": "%"},
        ge=0,
        le=100,
    )
    CORE_SREC: Series[int] = pa.Field(
        coerce=True,
        nullable=True,
        description="Percentage of solid core recovered in core run (SCR)",
        # example="23",
        metadata={"unit": "%"},
        ge=0,
        le=100,
    )
    CORE_RQD: Series[int] = pa.Field(
        coerce=True,
        nullable=True,
        description="Rock Quality Designation for core run (RQD)",
        # example="20",
        metadata={"unit": "%"},
        ge=0,
        le=100,
    )


class Ags3CORE(BaseCORE):
    HOLE_ID: Series[str] = pa.Field(
        # foreign_key="Ags3HOLE.HOLE_ID",
        coerce=True,
        description="Exploratory hole or location equivalent",
        # example="6421/A",
    )
    CORE_BOT: Series[float] = pa.Field(
        coerce=True,
        description="Depth to BOTTOM of core run",
        # example=3.54,
        metadata={"unit": "m"},
    )


class Ags4CORE(BaseCORE):
    LOCA_ID: Series[str] = pa.Field(
        # foreign_key="Ags4LOCA.LOCA_ID",
        coerce=True,
        description="Location identifier",
        # example="327/16A",
    )
    CORE_BASE: Series[float] = pa.Field(
        coerce=True,
        description="Depth to BASE of core run",
        # example=3.54,
        metadata={"unit": "m"},
    )


class BaseWETH(pa.DataFrameModel):
    WETH_TOP: Series[float] = pa.Field(
        coerce=True,
        description="Depth to top of weathering subdivision",
        # example=3.50,
        metadata={"unit": "m"},
    )
    WETH_BASE: Series[float] = pa.Field(
        coerce=True,
        description="Depth to base of weathering subdivision",
        # example=3.95,
        metadata={"unit": "m"},
    )


class Ags3WETH(BaseWETH):
    HOLE_ID: Series[str] = pa.Field(
        # foreign_key="Ags3HOLE.HOLE_ID",
        coerce=True,
        description="Exploratory hole or location equivalent",
        # example="6421/A",
    )
    WETH_GRAD: Series[str] = pa.Field(
        coerce=True,
        description="Weather Gradient",
        # example="IV",
    )


class Ags4WETH(BaseWETH):
    LOCA_ID: Series[str] = pa.Field(
        # foreign_key="Ags4LOCA.LOCA_ID",
        coerce=True,
        description="Location identifier",
        # example="327/16A",
    )
    WETH_WETH: Series[str] = pa.Field(
        coerce=True,
        description="Weathering classifier for WETH_SCH and WETH_SYS",
        # example="IV",
    )
