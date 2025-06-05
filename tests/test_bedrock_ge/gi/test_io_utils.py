import io
from pathlib import Path

import pytest

from bedrock_ge.gi.io_utils import detect_encoding

data_dir = Path(__file__).parent / "data"


def test_detect_encoding():
    ags3 = data_dir / "ags3_sample.ags"
    ags4 = data_dir / "ags4_sample.ags"

    default_encoding = "utf-8"
    ags4_encoding = "ISO-8859-1"
    ags3_encoding = "ascii"

    ags3_path = Path(ags3)
    ags4_path = Path(ags4)

    with open(ags3, encoding=ags3_encoding) as file:
        ags3_str = file.read()
    with open(ags4, encoding=ags4_encoding) as file:
        ags4_str = file.read()

    ags3_byte = ags3_str.encode(ags3_encoding)
    ags4_byte = ags4_str.encode(ags4_encoding)

    ags4_sio = io.StringIO(ags4_str)
    ags3_sio = io.StringIO(ags3_str)

    ags3_bio = io.BytesIO(ags3_byte)
    ags4_bio = io.BytesIO(ags4_byte)

    sources = {
        ags3: default_encoding,
        ags4: ags4_encoding,
        ags3_path: default_encoding,
        ags4_path: ags4_encoding,
        ags3_byte: default_encoding,
        ags4_byte: ags4_encoding,
        ags3_sio: default_encoding,
        ags4_sio: default_encoding,
        ags3_bio: default_encoding,
        ags4_bio: ags4_encoding,
    }
    for source, expected in sources.items():
        result = detect_encoding(source)
        assert expected == result

    # test non-existent file
    non_existent = data_dir / "empty.ags"
    with pytest.raises(FileNotFoundError):
        detect_encoding(non_existent)

    # test empty file (should still return default encoding)
    empty = data_dir / "empty.ags"
    empty.touch()
    expected = default_encoding
    result = detect_encoding(empty)
    empty.unlink()
    assert result == expected
