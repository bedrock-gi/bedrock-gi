from __future__ import annotations

import io
from contextlib import contextmanager, nullcontext
from io import TextIOBase
from pathlib import Path
from typing import IO, Any, ContextManager, Dict, List

import chardet
import pandas as pd
from python_ags4 import AGS4

from bedrock_ge.gi.ags.validate import check_ags_proj_group

DEFAULT_ENCODING = "utf-8"


def detect_encoding(source: str | Path | IO[str] | IO[bytes] | bytes) -> str:
    """Detect the character encoding of various input types.

    Args:
        source (str | Path | IO[str] | IO[bytes] | bytes): The source to detect encoding from.
            - str: Treated as a file path if it exists, otherwise as text (returns `DEFAULT_ENCODING`)
            - Path: File path to read and detect encoding
            - IO[str]: Already decoded text stream (returns `DEFAULT_ENCODING`)
            - IO[bytes]: Binary stream to detect encoding from
            - bytes: Binary data to detect encoding from

    Returns:
        str: The detected encoding name (e.g., 'utf-8', 'iso-8859-1', etc.)

    Raises:
        TypeError: If the source type is unsupported
        FileNotFoundError: If a file path doesn't exist
    """
    # Set number of bytes to read for detection and required confidence
    SAMPLE_SIZE = 10000
    REQUIRED_CONFIDENCE = 0.7

    def _detect_from_bytes(data: bytes) -> str:
        """Detect encoding from bytes data."""
        sample = data[: min(len(data), SAMPLE_SIZE)]
        result = chardet.detect(sample)
        encoding = result.get("encoding", DEFAULT_ENCODING)
        confidence = result.get("confidence", 0.0)

        if not encoding or confidence < REQUIRED_CONFIDENCE:
            return DEFAULT_ENCODING

        return encoding

    def _read_from_path(path: Path):
        """Read contents from path."""
        if path.exists() and path.is_file():
            with open(path, "rb") as file:
                sample = file.read(SAMPLE_SIZE)
                return _detect_from_bytes(sample)
        else:
            raise FileNotFoundError(
                f"Path does not exist or is not a file: {path.__str__()[0:40]}"
            )

    # bytes
    if isinstance(source, bytes):
        return _detect_from_bytes(source)

    # String, if not a path, still returns DEFAULT_ENCODING
    if isinstance(source, str):
        path = Path(source)
        try:
            return _read_from_path(path)
        except FileNotFoundError:
            return DEFAULT_ENCODING

    # Path object
    if isinstance(source, Path):
        return _read_from_path(source)

    # IO[str] object
    if hasattr(source, "encoding"):
        if source.encoding:
            # Could be `None`
            return source.encoding
        else:
            return DEFAULT_ENCODING

    # IO[bytes]
    if isinstance(source, io.BytesIO):
        original_position = source.tell()
        try:
            source.seek(0)
            sample = source.read(SAMPLE_SIZE)
            encoding = _detect_from_bytes(sample)
            if isinstance(sample, bytes):
                encoding = _detect_from_bytes(sample)
            else:
                # if not bytes, then its a custom string-like type that was not caught
                encoding = DEFAULT_ENCODING
            source.seek(original_position)
            return encoding
        except (AttributeError, IOError):
            # use default if the stream does not have a `read()` or `seek()` attribute
            return DEFAULT_ENCODING

    raise TypeError(f"Unsupported input type for encoding detection: {type(source)}")


def read_ags_source(
    source: str | Path | IO[str] | IO[bytes] | bytes, encoding=None
) -> ContextManager[TextIOBase]:
    """Opens or wraps a given source for reading AGS (text-based) data.

    Args:
        source (str | Path | IO[str] | IO[bytes] | bytes): The source to read from.
            - str or Path: File path or direct string content.
            - IO[str]: A file-like text stream.
            - IO[bytes]: Byte stream
            - bytes: Binary content or stream (will be decoded).
        encoding (str | None): Encoding to use for decoding bytes. Default is None.

    Returns:
        ContextManager[TextIOBase]: A context manager yielding a text stream.

    Raises:
        TypeError: If the source type is unsupported or binary streams are not decoded.
    """

    @contextmanager
    def string_source(content: str):
        string_io = io.StringIO(content)
        try:
            yield string_io
        finally:
            string_io.close()

    if isinstance(source, str):
        path = Path(source)
        if path.exists() and path.is_file():
            return open(path, "r", encoding=encoding)
        raise FileNotFoundError(f"Path does not exist or is not a file: {source}")

    elif isinstance(source, Path):
        if source.exists() and source.is_file():
            return open(source, "r", encoding=encoding)
        raise FileNotFoundError(f"Path does not exist or is not a file: {source}")

    elif isinstance(source, bytes):
        return string_source(source.decode(encoding))

    elif isinstance(source, io.BytesIO):
        return string_source(source.getvalue().decode(encoding))

    elif hasattr(source, "read"):
        # reset the cursor to the beginning
        try:
            source.seek(0)
        except (AttributeError, io.UnsupportedOperation):
            pass
        return nullcontext(source)

    raise TypeError(f"Unsupported input type: {type(source)}")


def ags_to_dfs(
    source: str | Path | IO[str] | IO[bytes] | bytes, encoding=None
) -> Dict[str, pd.DataFrame]:
    """Converts AGS 3 or AGS 4 file to a dictionary of pandas DataFrames.

    Args:
        source (str | Path | IO[str] | IO[bytes] | bytes): The AGS file (str or Path)
            or a file-like object that represents the AGS file.
        encoding (str): default=None
            Encoding of text file, an attempt at detecting the encoding will be made if `None`

    Raises:
        ValueError: If the data does not match AGS 3 or AGS 4 format.

    Returns:
        Dict[str, pd.DataFrame]]: A dictionary where keys represent AGS group
        names with corresponding DataFrames for the corresponding group data.
    """
    # if bytes are provided, convert to IO[bytes] to be file-like
    if isinstance(source, bytes):
        source = io.BytesIO(source)

    if not encoding:
        encoding = detect_encoding(source)

    # Get first non-blank line, `None` if all lines are blank
    with read_ags_source(source, encoding=encoding) as f:
        first_line = next((line.strip() for line in f if line.strip()), None)

    if first_line:
        if first_line.startswith('"**'):
            ags_version = 3
            ags_dfs = ags3_to_dfs(source, encoding=encoding)
        elif first_line.startswith('"GROUP"'):
            ags_version = 4
            ags_dfs = ags4_to_dfs(source)
        else:
            # If first non-empty line doesn't match AGS 3 or AGS 4 format
            raise ValueError("The data provided is not valid AGS 3 or AGS 4 data.")
    else:
        raise ValueError("The file provided has only blank lines")

    is_proj_group_correct = check_ags_proj_group(ags_dfs["PROJ"])
    if is_proj_group_correct:
        project_id = ags_dfs["PROJ"]["PROJ_ID"].iloc[0]
        print(
            f"AGS {ags_version} data was read for Project {project_id}",
            "This Ground Investigation data contains groups:",
            list(ags_dfs.keys()),
            sep="\n",
            end="\n\n",
        )

    return ags_dfs


def ags3_to_dfs(
    source: str | Path | IO[str] | IO[bytes] | bytes, encoding: str
) -> Dict[str, pd.DataFrame]:
    """Converts AGS 3 data to a dictionary of pandas DataFrames.

    Args:
        source (str | Path | IO[str] | IO[bytes] | bytes): The AGS 3 file (str or Path)
            or a file-like object that represents the AGS 3 file.
        encoding (str):  Encoding of file or object.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary of pandas DataFrames, where each key represents a group name from AGS 3 data,
        and the corresponding value is a pandas DataFrame containing the data for that group.
    """
    # Initialize dictionary and variables used in the AGS 3 read loop
    ags3_dfs = {}
    line_type = "line_0"
    group = ""
    headers: List[str] = ["", "", ""]
    group_data: List[List[Any]] = [[], [], []]

    with read_ags_source(source, encoding=encoding) as file:
        for i, line in enumerate(file):
            line = line.strip()
            last_line_type = line_type

            # In AGS 3.1 group names are prefixed with **
            if line.startswith('"**'):
                line_type = "group_name"
                if group:
                    ags3_dfs[group] = pd.DataFrame(group_data, columns=headers)

                group = line.strip(' ,"*')
                group_data = []

            # In AGS 3 header names are prefixed with "*
            elif line.startswith('"*'):
                line_type = "headers"
                new_headers = line.split('","')
                new_headers = [h.strip(' ,"*') for h in new_headers]

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
                        f"\n🚨 CAUTION: The number of columns on line {i + 1} ({len(data_row)}) doesn't match the number of columns of group {group} ({len(headers)})!",
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
    ags3_dfs[group] = pd.DataFrame(group_data, columns=headers).dropna(
        axis=1, how="all"
    )

    if not group:
        print(
            '🚨 ERROR: The provided AGS 3 data does not contain any groups, i.e. lines starting with "**'
        )

    return ags3_dfs


def ags4_to_dfs(
    source: str | Path | IO[str] | IO[bytes] | bytes,
) -> Dict[str, pd.DataFrame]:
    """Converts AGS 4 data to a dictionary of pandas DataFrames.

    Args:
        source (str | Path | IO[str] | IO[bytes] | bytes): The AGS4 file (str or Path) or a file-like
            object that represents and AGS4 file.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary of pandas DataFrames, where each key represents a group name from AGS 4 data,
        and the corresponding value is a pandas DataFrame containing the data for that group.
    """
    ags4_tups = AGS4.AGS4_to_dataframe(source)

    ags4_dfs = {}
    for group, df in ags4_tups[0].items():
        df = df.loc[2:].drop(columns=["HEADING"]).reset_index(drop=True)
        ags4_dfs[group] = df

    return ags4_dfs


def coerce_string(string: str) -> None | bool | float | str:
    """Converts a string to an appropriate Python data type.

    Args:
        string: The input string to be converted.

    Returns:
        None: If the string is 'none', 'null', or empty.
        bool: If the string is 'true' or 'false' (case insensitive).
        int: If the string can be converted to a float and has no decimal part.
        float: If the string can be converted to a float with a decimal part.
        str: If the string cannot be converted to any of the above types.

    """
    if string.lower() in {"none", "null", ""}:
        return None
    elif string.lower() == "true":
        return True
    elif string.lower() == "false":
        return False
    else:
        try:
            value = float(string)
            if value.is_integer():
                return int(value)
            else:
                return value
        except ValueError:
            return string
