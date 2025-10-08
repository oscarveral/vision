import numpy as np

from ctypes import cdll, c_uint32, c_uint8, c_size_t, POINTER
from os import path
import cv2

# Load the shared library.
lib_path = path.join(path.dirname(__file__), "libfilters.so")
libfilter = cdll.LoadLibrary(lib_path)

# Function sum_array from the C library.
libfilter.sum_array.argtypes = (POINTER(c_uint32), c_size_t)
libfilter.sum_array.restype = c_uint32

# Python wrapper for the sum_array function.
def sum_array(array: np.ndarray) -> int:
    """Sum the elements of a uint32 numpy array using the C function."""
    if array.dtype != np.uint32:
        raise ValueError("Array must be of type uint32")
    length = array.size
    c_array = array.ctypes.data_as(POINTER(c_uint32))
    return libfilter.sum_array(c_array, length)


def box_filter(input_image: np.ndarray, filter_size: int) -> np.ndarray:
    """Apply a box filter to a grayscale image using the C function."""
    if input_image.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8")
    if len(input_image.shape) != 2:
        raise ValueError("Input image must be a 2D array (grayscale)")
    
    height, width = input_image.shape
    output_image = np.zeros_like(input_image)
    
    c_input = input_image.ctypes.data_as(POINTER(c_uint8))
    c_output = output_image.ctypes.data_as(POINTER(c_uint8))
    
    result = libfilter.box_filter(c_input, c_output, width, height, filter_size)
    if result != 0:
        raise RuntimeError("Box filter failed in C library")
    
    return output_image

# Test

if __name__ == "__main__":
    arr = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    result = sum_array(arr)
    print(f"Sum of array: {result}")  # Output should be 15

    img = cv2.imread("Pytest.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Pytest.png not found")

    filtered_img = box_filter(img, filter_size=11)
    cv2.imwrite("Pytest_filtered.png", filtered_img)
    print("Filtered image saved as Pytest_filtered.png")
