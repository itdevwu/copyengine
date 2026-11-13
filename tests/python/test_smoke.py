import re

import copyengine as ce


def test_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+", ce.__version__)
    assert ce.ABI_VERSION == 0
