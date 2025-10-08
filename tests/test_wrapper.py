"""Tests for the ffi wrapper."""

import numpy as np
import pytest

from dgst.ffi.wrapper import sum_array

# Test cases for the sum_array function.
@pytest.mark.parametrize("arr, expected", [
    (np.array([1, 2, 3, 4], dtype=np.uint32), 10),
    (np.array([0, 0, 0], dtype=np.uint32), 0),
    (np.array([100, 200, 300], dtype=np.uint32), 600),
    (np.array([], dtype=np.uint32), 0),
    (np.array([4294967295, 1], dtype=np.uint32), 0),  # Overflow case
])
def test_sum_array(arr, expected):
    result = sum_array(arr)
    assert result == expected