import numpy as np
import ctypes as ffi
import os

# Load the shared library.
lib_path = os.path.join(os.path.dirname(__file__), "__objs__/libfilters.so.1.0.0")
libfilter = ffi.cdll.LoadLibrary(lib_path)

# Function sum_array from the C library.
libfilter.sum_array.argtypes = (ffi.POINTER(ffi.c_uint32), ffi.c_size_t)
libfilter.sum_array.restype = ffi.c_uint32


# Python wrapper for the sum_array function.
def sum_array(array: np.ndarray) -> int:
    """Sum the elements of a uint32 numpy array using the C function."""
    if array.dtype != np.uint32:
        raise ValueError("Array must be of type uint32.")
    length = array.size
    c_array = array.ctypes.data_as(ffi.POINTER(ffi.c_uint32))
    return libfilter.sum_array(c_array, length)


# Function box_filter from the C library.
libfilter.box_filter.argtypes = (
    ffi.POINTER(ffi.c_uint8),
    ffi.POINTER(ffi.c_uint8),
    ffi.c_size_t,
    ffi.c_size_t,
    ffi.c_size_t,
)
libfilter.box_filter.restype = ffi.c_int32


# Python wrapper for the box_filter function.
def box_filter(input_image: np.ndarray, filter_size: int) -> np.ndarray:
    """Apply a box filter to a grayscale image using the C function."""
    if input_image.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8.")
    if len(input_image.shape) != 2:
        raise ValueError("Input image must be a 2D array (grayscale).")

    height, width = input_image.shape
    output_image = np.zeros_like(input_image)

    c_input = input_image.ctypes.data_as(ffi.POINTER(ffi.c_uint8))
    c_output = output_image.ctypes.data_as(ffi.POINTER(ffi.c_uint8))

    result = libfilter.box_filter(c_input, c_output, width, height, filter_size)
    if result != 0:
        raise RuntimeError("Box filter failed in C library.")

    return output_image


# Function gaussian_filter from the C library.
libfilter.gaussian_filter.argtypes = (
    ffi.POINTER(ffi.c_uint8),
    ffi.POINTER(ffi.c_uint8),
    ffi.c_size_t,
    ffi.c_size_t,
    ffi.c_float,
)
libfilter.gaussian_filter.restype = ffi.c_int32


# Python wrapper for the gaussian_filter function.
def gaussian_filter(input_image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a Gaussian filter to a grayscale image using the C function."""
    if input_image.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8.")
    if len(input_image.shape) != 2:
        raise ValueError("Input image must be a 2D array (grayscale).")

    height, width = input_image.shape
    output_image = np.zeros_like(input_image)

    c_input = input_image.ctypes.data_as(ffi.POINTER(ffi.c_uint8))
    c_output = output_image.ctypes.data_as(ffi.POINTER(ffi.c_uint8))

    result = libfilter.gaussian_filter(
        c_input, c_output, width, height, ffi.c_float(sigma)
    )
    if result != 0:
        raise RuntimeError("Gaussian filter failed in C library.")

    return output_image
