"""Tests for the ffi wrapper."""

import numpy as np
import pytest

from dgst.ffi.wrapper import sum_array
import cv2
from dgst.ffi.wrapper import box_filter
import time

import urllib.request

IMAGE_URL = "https://gist.github.com/anishLearnsToCode/120f0ae1cc457b68a814f4697cd8deab/raw/fb822024733ff18803feef8e5b986bdeba8a293b/lenna.png"
IMAGE_PATH = "lenna.png"

def setup_module(module):
    # Download image before tests
    urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)

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

@pytest.mark.parametrize("arr", [np.array([], dtype=np.uint32)])
def test_sum_array_empty(arr):
    result = sum_array(arr)
    assert result == 0

@pytest.mark.parametrize("arr", [np.array([1, 2, 3], dtype=np.int32)])
def test_sum_array_wrong_dtype(arr):
    with pytest.raises(ValueError):
        sum_array(arr)

def test_box_filter_wrong_dtype():
    img = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        box_filter(img, filter_size=3)

def test_box_filter_wrong_shape():
    img = np.ones((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        box_filter(img, filter_size=3)

def test_box_filter_opencv_equivalence():
    img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
    assert img is not None, "lenna.png not found"
    filter_size = 5
    start_time = time.time()
    # Custom box filter
    filtered_custom = box_filter(img, filter_size=filter_size)
    custom_time = time.time() - start_time

    start_time = time.time()
    # OpenCV box filter
    kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size * filter_size)
    filtered_cv = cv2.filter2D(img, -1, kernel)
    opencv_time = time.time() - start_time

    print(f"Custom box_filter time: {custom_time:.6f}s")
    print(f"OpenCV filter2D time: {opencv_time:.6f}s")
    filtered_custom = box_filter(img, filter_size=filter_size)
    kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size * filter_size)
    filtered_cv = cv2.filter2D(img, -1, kernel)
    assert np.allclose(filtered_custom, filtered_cv, atol=5)

def teardown_module(module):
    # Remove image after tests
    if os.path.exists(IMAGE_PATH):
        os.remove(IMAGE_PATH)
