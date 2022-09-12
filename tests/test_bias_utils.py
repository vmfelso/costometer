import pytest

from costometer.utils.bias_utils import (  # noqa : F401
    calculate_overplanning,
    calculate_present_bias,
    calculate_present_bias_late,
)


# some setup where you load fake / simulate data as 'pytest fixture'
@pytest.fixture(
    params=[
        # where you put the test case info (paths to files, etc)
        [],
        [],
        [],
    ]
)
def bias_fixutre(request):
    # should yield outputs that will be used for testing biases
    yield request


def test_calculate_overplanning(bias_fixutre):
    return NotImplementedError


def test_calculate_present_bias(bias_fixture):
    return NotImplementedError


def test_calculate_present_bias_late(bias_fixutre):
    return NotImplementedError
