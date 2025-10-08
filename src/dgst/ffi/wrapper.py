import numpy as np

from ctypes import cdll, c_uint32, c_size_t, POINTER
from os import path

libfilter = cdll.LoadLibrary(path.join(path.dirname(__file__), "libfilters.so"))

libfilter.sum_array.argtypes = (POINTER(c_uint32), c_size_t)
libfilter.sum_array.restype = c_uint32

def sum_array(array: np.ndarray) -> int:
    """Sum the elements of a uint32 numpy array using the C function."""
    if array.dtype != np.uint32:
        raise ValueError("Array must be of type uint32")
    length = array.size
    c_array = array.ctypes.data_as(POINTER(c_uint32))
    return libfilter.sum_array(c_array, length)


if __name__ == "__main__":
    arr = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    result = sum_array(arr)
    print(f"Sum of array: {result}")  # Output should be 15
