"""Tests for the ffi box filter."""

import numpy as np
import pytest
import cv2
import os

from dgst.ffi.wrapper import box_filter

TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images/lenna.png")

# ---------------------------------------------------------------
# Tests for box_filter.
# ---------------------------------------------------------------

def test_box_filter_wrong_dtype():
    img = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        box_filter(img, filter_size=3)


def test_box_filter_wrong_shape():
    img = np.ones((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        box_filter(img, filter_size=3)


def test_box_filter_opencv_equivalence():
    img = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "images/lenna.png not found"

    filter_size = 7

    # Compare results.
    filtered_custom = box_filter(img, filter_size=filter_size)
    kernel = np.ones((filter_size, filter_size), np.float32) / (
        filter_size * filter_size
    )
    filtered_cv = cv2.filter2D(img, -1, kernel)

    assert np.allclose(filtered_custom, filtered_cv, atol=5)
