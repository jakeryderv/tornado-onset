import os

import pytest


@pytest.fixture(scope="session")
def tornet_root() -> str:
    root = os.environ.get("TORNET_ROOT")
    if not root or not os.path.exists(os.path.join(root, "catalog.csv")):
        pytest.skip("TORNET_ROOT not set or has no catalog.csv")
    return root
