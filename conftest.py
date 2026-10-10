import pathlib
import tempfile

import pytest


@pytest.fixture
def tmp_path():
    """Give each test its own clean temporary directory."""
    root = pathlib.Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="pytest_", dir=root) as tmp_dir:
        yield pathlib.Path(tmp_dir)
