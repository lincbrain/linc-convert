import subprocess
from pathlib import Path

import pytest

DOWNLOAD_CMD = ["dandi",
                "download",
                "https://lincbrain.org/dandiset/000051/draft",
                "-i",
                "linc",
                "-e",
                "refresh",
                "--preserve-tree"
                ]


@pytest.fixture(scope="session")
def test_data_heavy_dir(request):
    data_dir = Path(__file__).parent / "data"
    if not any(data_dir.iterdir()):
        subprocess.check_call(DOWNLOAD_CMD)
    yield data_dir / "000051" / "sourcedata"
    if request.node.get_closest_marker("golden"):
        print("updating dandi archive")
    # if request.config.getoption("--upload-data"):
    #     # subprocess.check_call()
    #     pass


@pytest.fixture(scope="module", params=["zarr-python", "tensorstore"])
def driver(request):
    return request.param


@pytest.fixture(scope="module", params=[2, 3])
def zarr_version(request):
    return request.param


@pytest.mark.heavy
def test_heavy(test_data_heavy_dir):
    assert True
