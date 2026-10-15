import pytest

# one shard of each 2013 partition from the Hub dataset: ~135 MB + ~90 MB, cached after the first run
HUB_FILES = {
    "train": ["data/train/2013/part-00000-of-00014.parquet"],
    "test": ["data/test/2013/part-00000-of-00003.parquet"],
}


@pytest.fixture(scope="session")
def hub_files() -> dict:
    from tornado_onset.data import load_split

    try:
        load_split("test", [], data_files=HUB_FILES)
    except Exception as e:  # offline, or the Hub unreachable
        pytest.skip(f"Hub dataset not reachable: {e}")
    return HUB_FILES
