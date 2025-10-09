"""Tests for the ffi wrapper."""

import numpy as np
import pytest
import cv2
import time
import os

from dgst.ffi.wrapper import sum_array
from dgst.ffi.wrapper import box_filter
from dgst.ffi.wrapper import gaussian_filter

TEST_IMAGE_PATH = "images/lenna.png"

# ---------------------------------------------------------------
# Tests for sum_array.
# ---------------------------------------------------------------

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

@pytest.mark.parametrize("arr", [np.array([1, 2, 3], dtype=np.int32)])
def test_sum_array_wrong_dtype(arr):
    with pytest.raises(ValueError):
        sum_array(arr)

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

    filter_size = 5
    
    # Custom box filter
    start_time = time.time()
    filtered_custom = box_filter(img, filter_size=filter_size)
    custom_time = time.time() - start_time

    # OpenCV box filter
    start_time = time.time()
    kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size * filter_size)
    filtered_cv = cv2.filter2D(img, -1, kernel)
    opencv_time = time.time() - start_time

    print(f"\n\nCustom box_filter time: {custom_time:.6f}s")
    print(f"OpenCV filter2D time: {opencv_time:.6f}s\n")

    # Compare results.
    filtered_custom = box_filter(img, filter_size=filter_size)
    kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size * filter_size)
    filtered_cv = cv2.filter2D(img, -1, kernel)

    assert np.allclose(filtered_custom, filtered_cv, atol=5) 

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

    sigma = 1.5
    
    # Custom Gaussian filter
    start_time = time.time()
    filtered_custom = gaussian_filter(img, sigma=sigma)
    custom_time = time.time() - start_time

    # OpenCV Gaussian filter
    start_time = time.time()
    filtered_cv = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    opencv_time = time.time() - start_time

    print(f"\n\nCustom gaussian_filter time: {custom_time:.6f}s")
    print(f"OpenCV GaussianBlur time: {opencv_time:.6f}s\n")

    # Compare results.
    filtered_custom = gaussian_filter(img, sigma=sigma)
    filtered_cv = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)

    assert np.allclose(filtered_custom, filtered_cv, atol=10)