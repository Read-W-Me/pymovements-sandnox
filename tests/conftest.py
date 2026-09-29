import pytest

from tests.synthetic import EASY_POINTS, synth_scanpath


@pytest.fixture
def easy_df():
    return synth_scanpath(EASY_POINTS, fix_ms=400)
