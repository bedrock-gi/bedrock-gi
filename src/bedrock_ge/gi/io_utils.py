"""Utility functions for reading, parsing and writing data."""

from __future__ import annotations

import codecs
import io
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import IO, ContextManager

import chardet
import pandas as pd

DEFAULT_ENCODING = "utf-8"


def detect_encoding(source: str | Path | IO[str] | IO[bytes] | bytes) -> str:
    """Detect the character encoding of various input types.

    Args:
        source (str | Path | IO[str] | IO[bytes] | bytes): The source to detect encoding from.
            - str or Path: File path.
            - IO[str]: Already decoded text stream (returns `DEFAULT_ENCODING`)
            - IO[bytes]: Binary stream to detect encoding from
            - bytes: Binary data to detect encoding from

    Returns:
        str: The detected encoding name (e.g., 'utf-8', 'iso-8859-1', 'ascii', etc.)

    Raises:
        TypeError: If the source type is unsupported
        FileNotFoundError: If a file path doesn't exist
    """
    # Set number of bytes to read for detection and required confidence
    SAMPLE_SIZE = 1_000_000
    REQUIRED_CONFIDENCE = 0.7

    def _detect_from_bytes(data: bytes) -> str:
        """Detect encoding from bytes data."""
        sample = data[: min(len(data), SAMPLE_SIZE)]
        result = chardet.detect(sample)
        encoding = result.get("encoding", DEFAULT_ENCODING)
        confidence = result.get("confidence", 0.0)

        if not encoding or confidence < REQUIRED_CONFIDENCE:
            return DEFAULT_ENCODING

        if encoding.lower() == "ascii":
            return "utf-8"

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
            # Could be `None`, e.g. io.StringIO has an encoding attribute which is None.
            return source.encoding
        else:
            return DEFAULT_ENCODING

    # IO[bytes]
    if isinstance(source, io.BufferedIOBase):
        try:
            original_position = source.tell()
            source.seek(0)
            sample = source.read(SAMPLE_SIZE)
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


def open_text_data_source(
    source: str | Path | IO[str] | IO[bytes] | bytes, encoding=None
) -> ContextManager[io.TextIOBase]:
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
    try:
        codecs.lookup(encoding)
    except LookupError:
        raise ValueError(f"Unsupported encoding: {encoding}")

    @contextmanager
    def _bytes_source(bytes_content: bytes):
        string_io = io.StringIO(bytes_content.decode(encoding))
        try:
            yield string_io
        finally:
            string_io.close()

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists() and path.is_file():
            return open(path, "r", encoding=encoding)
        raise FileNotFoundError(f"Path does not exist or is not a file: {source}")

    elif isinstance(source, io.TextIOBase):
        source.seek(0)
        return nullcontext(source)

    elif isinstance(source, io.BufferedIOBase):
        text_stream = io.TextIOWrapper(source, encoding=encoding)
        text_stream.seek(0)
        return nullcontext(text_stream)

    elif isinstance(source, bytes):
        return _bytes_source(source)

    else:
        raise TypeError(
            f"Unsupported source type: {type(source)}. "
            "Expected str, Path, IO[str], IO[bytes], or bytes."
        )


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


def convert_dtypes_object_to_string(dataframe: pd.DataFrame) -> pd.DataFrame:
    object_cols = dataframe.select_dtypes(include=["object"]).columns
    dataframe[object_cols] = dataframe[object_cols].astype("string")
    return dataframe
