import numpy as np
import pytest
import cv2
import os

from dgst.ffi.wrapper import gaussian_filter

TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images/lenna.png")

# ---------------------------------------------------------------
# Tests for gaussian_filter.
# ---------------------------------------------------------------

def test_gaussian_filter_wrong_dtype():
    img = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        gaussian_filter(img, sigma=1.0)


def test_gaussian_filter_wrong_shape():
    img = np.ones((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        gaussian_filter(img, sigma=1.0)


def test_gaussian_filter_opencv_equivalence():
    img = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "images/lenna.png not found"

    sigma = 5

    # Compare results.
    filtered_custom = gaussian_filter(img, sigma=sigma)
    filtered_cv = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)

    assert np.allclose(filtered_custom, filtered_cv, atol=10)
