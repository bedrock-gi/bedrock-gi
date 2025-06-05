from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd
from python_ags4 import AGS4


def ags4_to_dfs(
    source: str | Path | IO[str] | IO[bytes] | bytes,
) -> dict[str, pd.DataFrame]:
    """Converts AGS 4 data to a dictionary of pandas DataFrames.

    Args:
        source (str | Path | IO[str] | IO[bytes] | bytes): The AGS4 file (str or Path) or a file-like
            object that represents and AGS4 file.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary of pandas DataFrames, where each key
            represents a group name from AGS 4 data, and the corresponding value is a
            pandas DataFrame containing the data for that group.
    """
    ags4_tups = AGS4.AGS4_to_dataframe(source)

    ags4_dfs = {}
    for group, df in ags4_tups[0].items():
        df = df.loc[2:].drop(columns=["HEADING"]).reset_index(drop=True)
        ags4_dfs[group] = df

    return ags4_dfs
