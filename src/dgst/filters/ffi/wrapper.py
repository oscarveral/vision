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


# Function to map points from distorted to undistorted pixel coordinates
libfilter.kannala_brandt_map_points_to_undistorted.argtypes = (
    ffi.POINTER(ffi.c_float),  # points_in
    ffi.POINTER(ffi.c_float),  # points_out
    ffi.c_size_t,              # n_points
    ffi.POINTER(ffi.c_float),  # intrinsics_3x3
    ffi.POINTER(ffi.c_float),  # distortion_4
)
libfilter.kannala_brandt_map_points_to_undistorted.restype = ffi.c_int32


def kannala_brandt_map_points_to_undistorted(points: np.ndarray, intrinsics_3x3: np.ndarray, distortion_4: np.ndarray) -> np.ndarray:
    """Map an array of points (Nx2) from distorted image coordinates to undistorted coordinates.

    Args:
        points: Nx2 array of (u, v) pixel coordinates in distorted image.
        intrinsics_3x3: 3x3 intrinsics matrix.
        distortion_4: array of 4 distortion coefficients.

    Returns:
        Nx2 float32 array with mapped coordinates in undistorted image.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be a Nx2 array of (u,v) coordinates")

    if intrinsics_3x3.shape != (3, 3):
        raise ValueError("Intrinsics matrix must be 3x3")
    if distortion_4.shape != (4,):
        raise ValueError("Distortion coefficients must have 4 elements")

    n_points = pts.shape[0]
    pts_flat = np.ascontiguousarray(pts.flatten(), dtype=np.float32)
    out_flat = np.empty_like(pts_flat)

    c_in = pts_flat.ctypes.data_as(ffi.POINTER(ffi.c_float))
    c_out = out_flat.ctypes.data_as(ffi.POINTER(ffi.c_float))
    c_intrinsics = np.ascontiguousarray(intrinsics_3x3.flatten(), dtype=np.float32).ctypes.data_as(ffi.POINTER(ffi.c_float))
    c_dist = np.ascontiguousarray(distortion_4.astype(np.float32)).ctypes.data_as(ffi.POINTER(ffi.c_float))

    result = libfilter.kannala_brandt_map_points_to_undistorted(c_in, c_out, n_points, c_intrinsics, c_dist)
    if result != 0:
        raise RuntimeError(f"Kannala-Brandt point mapping failed in C library with error code {result}.")

    return out_flat.reshape((n_points, 2))


# phase_congruency from the C library.
libfilter.phase_congruency.argtypes = (
    ffi.POINTER(ffi.c_uint8),  # input
    ffi.POINTER(ffi.c_uint8),  # output (uint8, values in [0,255])
    ffi.c_size_t,              # width
    ffi.c_size_t,              # height
    ffi.c_int32,               # nscale
    ffi.c_int32,               # norient
    ffi.c_float,               # min_wavelength
    ffi.c_float,               # mult
    ffi.c_float,               # sigma_onf
    ffi.c_float,               # eps
)
libfilter.phase_congruency.restype = ffi.c_int32


def phase_congruency(input_image: np.ndarray,
                     nscale: int,
                     norient: int,
                     min_wavelength: float,
                     mult: float,
                     sigma_onf: float,
                     eps: float) -> np.ndarray:
    """Compute phase congruency on a grayscale image using the C implementation.

    Args:
        input_image: 2D uint8 grayscale image
        nscale: number of scales
        norient: number of orientations
        min_wavelength: minimum wavelength
        mult: scaling factor between successive filters
        sigma_onf: ratio of the standard deviation of the Gaussian describing
                  the log Gabor filter's transfer function in the frequency
                  domain to the filter center frequency
        eps: small epsilon to avoid division by zero

    Returns:
        uint8 2D numpy array with phase congruency result (0-255)
    """
    if input_image.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8.")
    if len(input_image.shape) != 2:
        raise ValueError("Input image must be a 2D array (grayscale).")

    input_image = np.ascontiguousarray(input_image)

    height, width = input_image.shape
    # Output is a uint8 map in range [0,255]
    output_image = np.zeros((height, width), dtype=np.uint8)

    c_input = input_image.ctypes.data_as(ffi.POINTER(ffi.c_uint8))
    c_output = output_image.ctypes.data_as(ffi.POINTER(ffi.c_uint8))

    result = libfilter.phase_congruency(
        c_input,
        c_output,
        width,
        height,
        ffi.c_int32(nscale),
        ffi.c_int32(norient),
        ffi.c_float(min_wavelength),
        ffi.c_float(mult),
        ffi.c_float(sigma_onf),
        ffi.c_float(eps),
    )

    if result != 0:
        raise RuntimeError(f"Phase congruency failed in C library with error code {result}.")

    return output_image

# C-backed threshold filter: expects float32 input and produces float32 output (0.0/1.0)
libfilter.threshold_filter.argtypes = (
    ffi.POINTER(ffi.c_uint8),
    ffi.POINTER(ffi.c_uint8),
    ffi.c_size_t,
    ffi.c_size_t,
    ffi.c_float,
)
libfilter.threshold_filter.restype = ffi.c_int32


def threshold_filter(input_image: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold a uint8 image using the C implementation.

    Args:
        input_image: 2D numpy array of uint8 (values in [0,255]).
        threshold: float in [0,1]. Values normalized by /255 are compared to threshold.

    Returns:
        2D numpy array of dtype uint8 with values 0 or 255.
    """
    if not (isinstance(threshold, float) or isinstance(threshold, (int,))):
        raise ValueError("threshold must be a float between 0 and 1")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be between 0 and 1")

    if input_image.ndim != 2:
        raise ValueError("Input image must be a 2D array (grayscale).")

    if input_image.dtype != np.uint8:
        raise ValueError("Input image must be uint8 with values in [0,255].")

    img = np.ascontiguousarray(input_image)
    height, width = img.shape
    output_image = np.zeros((height, width), dtype=np.uint8)

    c_input = img.ctypes.data_as(ffi.POINTER(ffi.c_uint8))
    c_output = output_image.ctypes.data_as(ffi.POINTER(ffi.c_uint8))

    result = libfilter.threshold_filter(c_input, c_output, width, height, ffi.c_float(threshold))
    if result != 0:
        raise RuntimeError(f"Threshold filter failed in C library with error code {result}.")

    return output_image
