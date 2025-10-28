import numpy as np
import ctypes as ffi
import os

# Load the shared library.
lib_path = os.path.join(os.path.dirname(__file__), "__objs__/libfilters.so.1.0.0")
libfilter = ffi.cdll.LoadLibrary(lib_path)

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


# Function canny_edge_detection from the C library.
libfilter.canny_edge_detection.argtypes = (
    ffi.POINTER(ffi.c_uint8),
    ffi.POINTER(ffi.c_uint8),
    ffi.c_size_t,
    ffi.c_size_t,
    ffi.c_float,
    ffi.c_float,
)
libfilter.canny_edge_detection.restype = ffi.c_int32


# Python wrapper for the canny_edge_detection function.
def canny_edge_detection(input_image: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    """Apply Canny edge detection to a pre-smoothed grayscale image using the C function.
    
    This function expects a pre-smoothed input. For best results, apply Gaussian 
    smoothing (sigma ~1.4) before calling this function.
    
    Args:
        input_image: Pre-smoothed grayscale image as uint8 2D numpy array
        low_threshold: Lower threshold for hysteresis (weak edges)
        high_threshold: Upper threshold for hysteresis (strong edges)
    
    Returns:
        Binary edge map with 255 for edges, 0 for non-edges
    """
    if input_image.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8.")
    if len(input_image.shape) != 2:
        raise ValueError("Input image must be a 2D array (grayscale).")

    height, width = input_image.shape
    output_image = np.zeros_like(input_image)

    c_input = input_image.ctypes.data_as(ffi.POINTER(ffi.c_uint8))
    c_output = output_image.ctypes.data_as(ffi.POINTER(ffi.c_uint8))

    result = libfilter.canny_edge_detection(
        c_input, c_output, width, height, ffi.c_float(low_threshold), ffi.c_float(high_threshold)
    )
    if result != 0:
        raise RuntimeError(f"Canny edge detection failed in C library with error code {result}.")

    return output_image


# Function kannala_brandt_undistort from the C library.
libfilter.kannala_brandt_undistort.argtypes = (
    ffi.POINTER(ffi.c_uint8),
    ffi.POINTER(ffi.c_uint8),
    ffi.c_size_t,
    ffi.c_size_t,
    ffi.c_size_t,
    ffi.POINTER(ffi.c_float),
    ffi.POINTER(ffi.c_float),
)
libfilter.kannala_brandt_undistort.restype = ffi.c_int32


# Python wrapper for the kannala_brandt_undistort function.
def kannala_brandt_undistort(input_image: np.ndarray, intrinsics_3x3: np.ndarray, 
                              distortion_4: np.ndarray) -> np.ndarray:
    """Apply Kannala-Brandt fisheye undistortion to an image using the C function.
    
    Args:
        input_image: Input image as uint8 numpy array (grayscale or RGB)
        intrinsics_3x3: Camera intrinsic matrix (3x3) containing [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        distortion_4: Array of 4 distortion coefficients [k1, k2, k3, k4]
    
    Returns:
        Undistorted image with same shape as input
    """
    if input_image.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8.")
    
    if len(input_image.shape) == 2:
        # Grayscale image
        height, width = input_image.shape
        channels = 1
    elif len(input_image.shape) == 3:
        # Color image
        height, width, channels = input_image.shape
        if channels != 3:
            raise ValueError("Color images must have 3 channels (RGB/BGR).")
    else:
        raise ValueError("Input image must be a 2D (grayscale) or 3D (color) array.")
    
    if intrinsics_3x3.shape != (3, 3):
        raise ValueError("Intrinsics matrix must be 3x3.")
    
    if distortion_4.shape != (4,):
        raise ValueError("Distortion coefficients must have 4 elements.")
    
    # Ensure contiguous arrays
    input_image = np.ascontiguousarray(input_image)
    intrinsics_flat = np.ascontiguousarray(intrinsics_3x3.flatten().astype(np.float32))
    distortion_flat = np.ascontiguousarray(distortion_4.astype(np.float32))
    
    output_image = np.zeros_like(input_image)
    
    c_input = input_image.ctypes.data_as(ffi.POINTER(ffi.c_uint8))
    c_output = output_image.ctypes.data_as(ffi.POINTER(ffi.c_uint8))
    c_intrinsics = intrinsics_flat.ctypes.data_as(ffi.POINTER(ffi.c_float))
    c_distortion = distortion_flat.ctypes.data_as(ffi.POINTER(ffi.c_float))
    
    result = libfilter.kannala_brandt_undistort(
        c_input, c_output, width, height, channels, c_intrinsics, c_distortion
    )
    
    if result != 0:
        raise RuntimeError(f"Kannala-Brandt undistortion failed in C library with error code {result}.")
    
    return output_image
