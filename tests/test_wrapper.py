import os
import sys
import numpy as np
import pytest

from dgst.ffi.wrapper import sum_array

@pytest.mark.parametrize("arr, expected", [
    (np.array([1, 2, 3, 4], dtype=np.uint32), 10),
])
def test_sum_array(arr, expected):
    result = sum_array(arr)
    assert result == expected