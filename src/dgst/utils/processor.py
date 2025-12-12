from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np

from dgst.filters.ffi import (
    box_filter,
    canny_edge_detection,
    gaussian_filter,
    kannala_brandt_map_points_to_undistorted,
    kannala_brandt_undistort,
    threshold_filter,
)
from dgst.filters.ffi import phase_congruency as pc_ffi
from dgst.filters.python import (
    add_channel_weight,
    clahe_filter,
    dilate_edges,
    filter_connected_components,
    filtro_rojo_azul,
    median_blur,
    otsu_threshold,
    scale_inter_area,
)
from dgst.filters.python import phase_congruency as pc_python
from dgst.utils.exceptions import ValidationError
from dgst.utils.loader import Image, RegionOfInterest


class ProcessingTechnique:
    BOX_FILTER = "box_filter"
    GAUSSIAN_FILTER = "gaussian_filter"
    CANNY_EDGE_DETECTION = "canny_edge_detection"
    GRAYSCALE = "grayscale"
    KANNALA_BRANDT_UNDISTORTION = "kannala_brandt_undistortion"
    PHASE_CONGRUENCY = "phase_congruency"
    THRESHOLD_FILTER = "threshold_filter"
    CLAHE = "clahe"
    OTSU_THRESHOLD = "otsu_threshold"
    # Geometric transforms
    ROTATION = "rotation"
    SCALE = "scale"
    # Augmentation transforms
    GAUSSIAN_NOISE = "gaussian_noise"



class ProcessingStep(ABC):
    """Abstract base class for processing steps."""

    @abstractmethod
    def process(self, image: Image) -> Image:
        """Apply the processing operation to the image.

        Args:
            image: Input image

        Returns:
            Processed image
        """
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get current parameters of the processing step."""
        pass


class GrayscaleStep(ProcessingStep):
    """Convert image to grayscale."""

    def process(self, image: Image) -> Image:
        if image.data is None:
            raise ValidationError("GrayscaleStep: Image.data is None")

        # Processing
        image.to_grayscale()
        
        # Post-processing validation
        if not image.is_grayscale:
            raise ValidationError(f"GrayscaleStep: Expected grayscale image, but got shape {image.data.shape}")
        if image.data.dtype != np.uint8:
            raise ValidationError(f"GrayscaleStep: Expected dtype uint8, but got {image.data.dtype}")
        
        # Update metadata
        image.metadata.add_step({
            "technique": ProcessingTechnique.GRAYSCALE,
            "output_shape": image.data.shape,
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {"technique": ProcessingTechnique.GRAYSCALE}


class BoxFilterStep(ProcessingStep):
    """Apply box filter using custom C implementation."""

    def __init__(self, filter_size: int = 5):
        if filter_size % 2 == 0:
            raise ValueError("filter_size must be odd")
        self.filter_size = filter_size

    def process(self, image: Image) -> Image:
        if image.data is None:
            raise ValidationError("BoxFilterStep: Image.data is None")
        if not image.is_grayscale:
            raise ValidationError(f"BoxFilterStep: Expected grayscale image, but got shape {image.data.shape}")
        
        # Ensure data is contiguous for C function
        if not image.data.flags['C_CONTIGUOUS']:
            image.data = np.ascontiguousarray(image.data)

        # Processing
        image.data = box_filter(image.data, self.filter_size)
        
        # Post-processing validation
        if image.data.ndim != 2:
            raise ValidationError(f"BoxFilterStep: Expected 2D array, but got {image.data.ndim}D array")
        if image.data.dtype != np.uint8:
            raise ValidationError(f"BoxFilterStep: Expected dtype uint8, but got {image.data.dtype}")
        
        # Update metadata
        image.metadata.add_step({
            "technique": ProcessingTechnique.BOX_FILTER,
            "filter_size": self.filter_size,
            "output_shape": image.data.shape
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": ProcessingTechnique.BOX_FILTER,
            "filter_size": self.filter_size,
        }


class GaussianFilterStep(ProcessingStep):
    """Apply Gaussian filter using custom C implementation."""
    
    def __init__(self, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = sigma

    def process(self, image: Image) -> Image:
        if image.data is None:
            raise ValidationError("GaussianFilterStep: Image.data is None")
        
        # Helper to process a single 2D channel
        def process_channel(channel: np.ndarray) -> np.ndarray:
            if not channel.flags['C_CONTIGUOUS']:
                channel = np.ascontiguousarray(channel)
            return gaussian_filter(channel, self.sigma)

        if image.data.ndim == 2:
            # Single channel
            image.data = process_channel(image.data)
        elif image.data.ndim == 3:
            # Multi-channel: process each channel independently
            channels = [process_channel(image.data[:, :, i]) for i in range(image.data.shape[2])]
            image.data = np.dstack(channels)
        else:
            raise ValidationError(f"GaussianFilterStep: Unsupported dimensions {image.data.ndim}")
            
        # Post-processing validation
        if image.data.dtype != np.uint8:
            raise ValidationError(f"GaussianFilterStep: Expected dtype uint8, but got {image.data.dtype}")
        
        # Update metadata
        image.metadata.add_step({
            "technique": ProcessingTechnique.GAUSSIAN_FILTER,
            "sigma": self.sigma,
            "output_shape": image.data.shape
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": ProcessingTechnique.GAUSSIAN_FILTER,
            "sigma": self.sigma,
        }


class CannyEdgeDetectionStep(ProcessingStep):
    """Apply Canny edge detection using custom C implementation."""
    
    def __init__(self, low_threshold: float = 50.0, high_threshold: float = 150.0):
        if high_threshold < low_threshold:
            raise ValueError("high_threshold must be >= low_threshold")
        if low_threshold < 0:
            raise ValueError("low_threshold must be >= 0")
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def process(self, image: Image) -> Image:
        if image.data is None:
            raise ValidationError("CannyEdgeDetectionStep: Image.data is None")
        
        # Helper to process a single 2D channel
        def process_channel(channel: np.ndarray) -> np.ndarray:
            if not channel.flags['C_CONTIGUOUS']:
                channel = np.ascontiguousarray(channel)
            return canny_edge_detection(channel, self.low_threshold, self.high_threshold)

        if image.data.ndim == 2:
            # Single channel
            image.data = process_channel(image.data)
        elif image.data.ndim == 3:
            # Multi-channel: process each channel independently
            channels = [process_channel(image.data[:, :, i]) for i in range(image.data.shape[2])]
            image.data = np.dstack(channels)
        else:
            raise ValidationError(f"CannyEdgeDetectionStep: Unsupported dimensions {image.data.ndim}")
            
        # Post-processing validation
        if image.data.dtype != np.uint8:
            raise ValidationError(f"CannyEdgeDetectionStep: Expected dtype uint8, but got {image.data.dtype}")
        
        # Update metadata
        image.metadata.add_step({
            "technique": ProcessingTechnique.CANNY_EDGE_DETECTION,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "output_shape": image.data.shape
        })

        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": ProcessingTechnique.CANNY_EDGE_DETECTION,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
        }


class KannalaBrandtUndistortionStep(ProcessingStep):
    """Apply Kannala-Brandt undistortion using calibration data from the image."""

    def __init__(self):
        pass

    def process(self, image: Image) -> Image:
        # Precondition validation
        if image.calibration is None:
            raise ValidationError("KannalaBrandtUndistortionStep: Image does not contain calibration data")
        if image.calibration.camera_type != "kannala":
            raise ValidationError(
                f"KannalaBrandtUndistortionStep: Expected camera type 'kannala', "
                f"but got '{image.calibration.camera_type}'"
            )
        if not image.is_color:
            raise ValidationError(
                f"KannalaBrandtUndistortionStep: Expected color image, "
                f"but got shape {image.data.shape}"
            )
        
        # Validate intrinsics and distortion parameters
        if image.calibration.intrinsics is None:
            raise ValidationError("KannalaBrandtUndistortionStep: Calibration missing intrinsics")
        if image.calibration.distortion is None:
            raise ValidationError("KannalaBrandtUndistortionStep: Calibration missing distortion")
        
        # Extract intrinsic parameters (3x3 matrix from 3x4)
        K = image.calibration.intrinsics[:3, :3]
        if K.shape != (3, 3):
            raise ValidationError(
                f"KannalaBrandtUndistortionStep: Expected 3x3 intrinsics matrix, got {K.shape}"
            )

        # Extract distortion coefficients (first 4)
        D = np.array(image.calibration.distortion[:4], dtype=np.float32)
        if len(D) != 4:
            raise ValidationError(
                f"KannalaBrandtUndistortionStep: Expected 4 distortion coefficients, got {len(D)}"
            )
        
        # Ensure data is contiguous
        if not image.data.flags['C_CONTIGUOUS']:
            image.data = np.ascontiguousarray(image.data)

        # Apply the undistortion using C implementation
        original_shape = image.data.shape
        image.data = kannala_brandt_undistort(image.data, K, D)
        
        # Post-processing validation
        if not image.is_color:
            raise ValidationError(
                f"KannalaBrandtUndistortionStep: Expected color image, "
                f"but got shape {image.data.shape}"
            )
        if image.data.shape != original_shape:
            raise ValidationError(
                f"KannalaBrandtUndistortionStep: Shape changed unexpectedly from {original_shape} to {image.data.shape}"
            )

        # Remap ROI coordinates (if any) from distorted -> undistorted pixel coordinates
        if image.rois:
            # Build points array Nx2 (4 points per ROI)
            n_rois = len(image.rois)
            pts = np.zeros((n_rois * 4, 2), dtype=np.float32)
            for i, roi in enumerate(image.rois):
                base = i * 4
                pts[base + 0, :] = (float(roi.p1[0]), float(roi.p1[1]))
                pts[base + 1, :] = (float(roi.p2[0]), float(roi.p2[1]))
                pts[base + 2, :] = (float(roi.p3[0]), float(roi.p3[1]))
                pts[base + 3, :] = (float(roi.p4[0]), float(roi.p4[1]))

            mapped = kannala_brandt_map_points_to_undistorted(pts, K, D)

            new_rois = []
            for i in range(n_rois):
                base = i * 4
                p1 = (float(mapped[base + 0, 0]), float(mapped[base + 0, 1]))
                p2 = (float(mapped[base + 1, 0]), float(mapped[base + 1, 1]))
                p3 = (float(mapped[base + 2, 0]), float(mapped[base + 2, 1]))
                p4 = (float(mapped[base + 3, 0]), float(mapped[base + 3, 1]))
                new_rois.append(RegionOfInterest(p1=p1, p2=p2, p3=p3, p4=p4))

            image.rois = new_rois
        
        # Update metadata
        image.metadata.add_step({
            "technique": ProcessingTechnique.KANNALA_BRANDT_UNDISTORTION,
            "camera_type": image.calibration.camera_type,
            "rois_remapped": len(image.rois) if image.rois else 0,
            "output_shape": image.data.shape
        })

        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": ProcessingTechnique.KANNALA_BRANDT_UNDISTORTION,
        }


class PhaseCongruencyStep(ProcessingStep):
    """Compute phase congruency map (multi-scale, multi-orientation)."""

    def __init__(
        self,
        nscale: int = 4,
        norient: int = 6,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_onf: float = 0.55,
        eps: float = 1e-4,
        use_own: bool = False,
    ):
        if nscale < 1:
            raise ValueError("nscale must be >= 1")
        if norient < 1:
            raise ValueError("norient must be >= 1")
        self.nscale = nscale
        self.norient = norient
        self.min_wavelength = float(min_wavelength)
        self.mult = float(mult)
        self.sigma_onf = float(sigma_onf)
        self.eps = float(eps)
        self.use_own = use_own

    def process(self, image: Image) -> Image:
        # Precondition validation
        if image.data is None:
            raise ValidationError("PhaseCongruencyStep: Image.data is None")
        if not image.is_grayscale:
            raise ValidationError(f"PhaseCongruencyStep: Expected grayscale image, but got shape {image.data.shape}")
        
        # Ensure data is contiguous
        if not image.data.flags['C_CONTIGUOUS']:
            image.data = np.ascontiguousarray(image.data)

        func = pc_ffi if self.use_own else pc_python

        # Processing
        result = func(
            image.data,
            nscale=self.nscale,
            norient=self.norient,
            min_wavelength=self.min_wavelength,
            mult=self.mult,
            sigma_onf=self.sigma_onf,
            eps=self.eps,
        )
        
        # Validate output
        if result is None:
            raise ValidationError("PhaseCongruencyStep: Output is None")
        if not isinstance(result, np.ndarray):
            raise ValidationError(
                f"PhaseCongruencyStep: Expected numpy array output, got {type(result)}"
            )
        
        image.data = result
        
        # Post-processing validation
        if image.data is None:
            raise ValidationError("PhaseCongruencyStep: Image.data is None")
        if image.data.ndim != 2:
            raise ValidationError(f"PhaseCongruencyStep: Expected 2D array, but got {image.data.ndim}D array")
        
        # Update metadata
        image.metadata.add_step({
            "technique": ProcessingTechnique.PHASE_CONGRUENCY,
            "nscale": self.nscale,
            "norient": self.norient,
            "min_wavelength": self.min_wavelength,
            "mult": self.mult,
            "sigma_onf": self.sigma_onf,
            "eps": self.eps,
            "use_own": self.use_own,
            "output_shape": image.data.shape,
            "output_dtype": str(image.data.dtype)
        })

        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": ProcessingTechnique.PHASE_CONGRUENCY,
            "nscale": self.nscale,
            "norient": self.norient,
            "min_wavelength": self.min_wavelength,
            "mult": self.mult,
            "sigma_onf": self.sigma_onf,
            "eps": self.eps,
        }


class ThresholdFilterStep(ProcessingStep):
    """Threshold a float image using the C implementation."""

    def __init__(self, threshold: float = 0.5):
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = float(threshold)

    def process(self, image: Image) -> Image:
        # Precondition validation
        if image.data is None:
            raise ValidationError("ThresholdFilterStep: Image.data is None")
        if not image.is_grayscale:
            raise ValidationError(f"ThresholdFilterStep: Expected grayscale image, but got shape {image.data.shape}")
        
        # Ensure data is contiguous for C function
        if not image.data.flags['C_CONTIGUOUS']:
            image.data = np.ascontiguousarray(image.data)

        # Call the C-backed threshold_filter which expects uint8 input and returns uint8 0/255
        result = threshold_filter(image.data, self.threshold)
        
        # Post-processing validation
        if result is None:
            raise ValidationError("ThresholdFilterStep: Output is None")
        if not isinstance(result, np.ndarray):
            raise ValidationError(
                f"ThresholdFilterStep: Expected numpy array output, got {type(result)}"
            )
        if result.ndim != 2:
            raise ValidationError(
                f"ThresholdFilterStep: Expected 2D output, got {result.ndim}D"
            )

        # Store result (uint8) back into image.data
        image.data = result
        
        # Update metadata
        image.metadata.add_step({
            "technique": ProcessingTechnique.THRESHOLD_FILTER,
            "threshold": self.threshold,
            "output_shape": image.data.shape,
            "output_dtype": str(image.data.dtype)
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": ProcessingTechnique.THRESHOLD_FILTER,
            "threshold": self.threshold,
        }


class CLAHEStep(ProcessingStep):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size=(8, 8)):
        if clip_limit <= 0:
            raise ValueError("clip_limit must be positive")
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = (int(tile_grid_size[0]), int(tile_grid_size[1]))

    def process(self, image: Image) -> Image:
        # Precondition validation
        if image.data is None:
            raise ValidationError("CLAHEStep: Image.data is None")
        
        # Ensure data is contiguous
        if not image.data.flags['C_CONTIGUOUS']:
            image.data = np.ascontiguousarray(image.data)

        # Processing
        result = clahe_filter(image.data, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
        
        # Post-processing validation
        if result is None:
            raise ValidationError("CLAHEStep: Output is None")
        if not isinstance(result, np.ndarray):
            raise ValidationError(
                f"CLAHEStep: Expected numpy array output, got {type(result)}"
            )
        if result.shape != image.data.shape:
            raise ValidationError(
                f"CLAHEStep: Output shape {result.shape} doesn't match input shape {image.data.shape}"
            )
        if result.dtype != np.uint8:
            raise ValidationError(
                f"CLAHEStep: Expected uint8 output, got {result.dtype}"
            )
        
        image.data = result
        
        # Update metadata
        image.metadata.add_step({
            "technique": ProcessingTechnique.CLAHE,
            "clip_limit": self.clip_limit,
            "tile_grid_size": self.tile_grid_size,
            "output_shape": image.data.shape
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": ProcessingTechnique.CLAHE,
            "clip_limit": self.clip_limit,
            "tile_grid_size": self.tile_grid_size,
        }


class OtsuThresholdStep(ProcessingStep):
    """Apply Otsu automatic threshold to a 2D image and return uint8 mask."""

    def __init__(self):
        pass

    def process(self, image: Image) -> Image:
        # Precondition validation
        if image.data is None:
            raise ValidationError("OtsuThresholdStep: Image.data is None")
        if not image.is_grayscale:
            raise ValidationError(f"OtsuThresholdStep: Expected grayscale image, but got shape {image.data.shape}")
        
        # Ensure data is contiguous
        if not image.data.flags['C_CONTIGUOUS']:
            image.data = np.ascontiguousarray(image.data)

        # Processing
        result = otsu_threshold(image.data)
        
        # Post-processing validation
        if result is None:
            raise ValidationError("OtsuThresholdStep: Output is None")
        if not isinstance(result, np.ndarray):
            raise ValidationError(
                f"OtsuThresholdStep: Expected numpy array output, got {type(result)}"
            )
        if result.ndim != 2:
            raise ValidationError(
                f"OtsuThresholdStep: Expected 2D output, got {result.ndim}D"
            )
        if result.dtype != np.uint8:
            raise ValidationError(
                f"OtsuThresholdStep: Expected uint8 output, got {result.dtype}"
            )
        
        image.data = result
        
        # Update metadata
        image.metadata.add_step({
            "technique": ProcessingTechnique.OTSU_THRESHOLD,
            "output_shape": image.data.shape
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {"technique": ProcessingTechnique.OTSU_THRESHOLD}
    
class DilateEdgesStep(ProcessingStep):
    """Dilate edges in a binary edge image."""
    
    def __init__(self, kernel_size: int = 3, iterations: int = 1):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.kernel_size = kernel_size
        self.iterations = iterations

    def process(self, image: Image) -> Image:
        if image.data is None:
            raise ValidationError("DilateEdgesStep: Image.data is None")
        
        # Helper to process a single 2D channel
        def process_channel(channel: np.ndarray) -> np.ndarray:
            if not channel.flags['C_CONTIGUOUS']:
                channel = np.ascontiguousarray(channel)
            dilated = dilate_edges(channel, self.kernel_size, self.iterations)
            # Convert back to uint8 binary mask if needed (assuming dilate_edges returns something compatible)
            # The original code did: np.where(result > 0, 255, 0).astype(np.uint8)
            return np.where(dilated > 0, 255, 0).astype(np.uint8)

        if image.data.ndim == 2:
            # Single channel
            image.data = process_channel(image.data)
        elif image.data.ndim == 3:
            # Multi-channel: process each channel independently
            channels = [process_channel(image.data[:, :, i]) for i in range(image.data.shape[2])]
            image.data = np.dstack(channels)
        else:
            raise ValidationError(f"DilateEdgesStep: Unsupported dimensions {image.data.ndim}")
            
        # Update metadata
        image.metadata.add_step({
            "technique": "dilate_edges",
            "kernel_size": self.kernel_size,
            "iterations": self.iterations,
            "output_shape": image.data.shape
        })

        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": "dilate_edges",
            "kernel_size": self.kernel_size,
            "iterations": self.iterations
        }
    
class ScaleInterAreaStep(ProcessingStep):
    """Scale image using area interpolation."""
    
    def __init__(self, scale_factor: float):
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        self.scale_factor = scale_factor

    def process(self, image: Image) -> Image:
        # Precondition validation
        if image.data is None:
            raise ValidationError("ScaleInterAreaStep: Image.data is None")
        
        original_shape = image.data.shape
        
        # Ensure data is contiguous
        if not image.data.flags['C_CONTIGUOUS']:
            image.data = np.ascontiguousarray(image.data)

        # Processing
        result = scale_inter_area(image.data, self.scale_factor)
        
        # Post-processing validation
        if result is None:
            raise ValidationError("ScaleInterAreaStep: Output is None")
        if not isinstance(result, np.ndarray):
            raise ValidationError(
                f"ScaleInterAreaStep: Expected numpy array output, got {type(result)}"
            )
        
        # Validate output shape is scaled correctly
        expected_height = int(original_shape[0] * self.scale_factor)
        expected_width = int(original_shape[1] * self.scale_factor)
        
        if result.shape[0] != expected_height or result.shape[1] != expected_width:
            raise ValidationError(
                f"ScaleInterAreaStep: Expected output shape approximately "
                f"({expected_height}, {expected_width}), got {result.shape[:2]}"
            )
        
        image.data = result
        
        # Update metadata
        image.metadata.add_step({
            "technique": "scale_inter_area",
            "scale_factor": self.scale_factor,
            "original_shape": original_shape,
            "output_shape": image.data.shape
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": "scale_inter_area",
            "scale_factor": self.scale_factor
        }

class MedianBlurStep(ProcessingStep):
    """Apply median blur to an image."""
    
    def __init__(self, kernel_size: int):
        if kernel_size % 2 == 0 or kernel_size <= 1:
            raise ValueError("kernel_size must be an odd number greater than 1")
        self.kernel_size = kernel_size

    def process(self, image: Image) -> Image:
        # Precondition validation
        if image.data is None:
            raise ValidationError("MedianBlurStep: Image.data is None")
        
        original_shape = image.data.shape
        original_dtype = image.data.dtype
        
        # Ensure data is contiguous
        if not image.data.flags['C_CONTIGUOUS']:
            image.data = np.ascontiguousarray(image.data)

        # Processing
        result = median_blur(image.data, self.kernel_size)
        
        # Post-processing validation
        if result is None:
            raise ValidationError("MedianBlurStep: Output is None")
        if not isinstance(result, np.ndarray):
            raise ValidationError(
                f"MedianBlurStep: Expected numpy array output, got {type(result)}"
            )
        if result.shape != original_shape:
            raise ValidationError(
                f"MedianBlurStep: Output shape {result.shape} doesn't match input shape {original_shape}"
            )
        if result.dtype != original_dtype:
            raise ValidationError(
                f"MedianBlurStep: Output dtype {result.dtype} doesn't match input dtype {original_dtype}"
            )
        
        image.data = result
        
        # Update metadata
        image.metadata.add_step({
            "technique": "median_blur",
            "kernel_size": self.kernel_size,
            "output_shape": image.data.shape
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": "median_blur",
            "kernel_size": self.kernel_size
        }
    
class IntoHSVChannelsStep(ProcessingStep):
    """Convert BGR image into HSV color space."""
    
    def __init__(self):
        pass

    def process(self, image: Image) -> Image:
        # Precondition validation
        if image.data is None:
            raise ValidationError("IntoHSVChannelsStep: Image.data is None")
        if not image.is_color:
            raise ValidationError(f"IntoHSVChannelsStep: Expected color image, but got shape {image.data.shape}")
        
        # Ensure data is contiguous
        if not image.data.flags['C_CONTIGUOUS']:
            image.data = np.ascontiguousarray(image.data)

        # Processing: Convert BGR to HSV
        # We use cv2 directly here as it's efficient and standard
        image.data = cv2.cvtColor(image.data, cv2.COLOR_BGR2HSV)
        
        # Post-processing validation
        if not image.is_color:
            raise ValidationError(f"IntoHSVChannelsStep: Expected color image, but got shape {image.data.shape}")
        
        # Update metadata
        image.metadata.add_step({
            "technique": "into_hsv_channels",
            "output_shape": image.data.shape,
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": "into_hsv_channels",
        }
    
class CombineChannelsStep(ProcessingStep):
    """Combine two channels with a specified weight for the second channel."""

    def __init__(self, channel1: str, channel2: str, weight: float):
        if not (0.0 <= weight <= 1.0):
            raise ValueError("weight must be between 0 and 1")
        
        if channel1 not in ['H', 'S', 'V'] or channel2 not in ['H', 'S', 'V']:
            raise ValueError("channel1 and channel2 must be one of 'H', 'S', or 'V'")
        
        self.channel1 = channel1
        self.channel2 = channel2
        self.weight = weight
        self.channel_map = {'H': 0, 'S': 1, 'V': 2}

    def process(self, image: Image) -> Image:
        # Precondition validation
        if image.data is None:
            raise ValidationError("CombineChannelsStep: Image.data is None")
        if not image.is_color:
            raise ValidationError(f"CombineChannelsStep: Expected color image, but got shape {image.data.shape}")
        
        # Ensure data is contiguous
        if not image.data.flags['C_CONTIGUOUS']:
            image.data = np.ascontiguousarray(image.data)
        
        # Extract channels
        idx1 = self.channel_map[self.channel1]
        idx2 = self.channel_map[self.channel2]
        
        c1 = image.data[:, :, idx1]
        c2 = image.data[:, :, idx2]
        
        # Processing
        result = add_channel_weight(c1, c2, self.weight)
        
        # Post-processing validation
        if result is None:
            raise ValidationError("CombineChannelsStep: Output is None")
        if not isinstance(result, np.ndarray):
            raise ValidationError(
                f"CombineChannelsStep: Expected numpy array output, got {type(result)}"
            )
        if result.ndim != 2:
            raise ValidationError(
                f"CombineChannelsStep: Expected 2D output, got {result.ndim}D"
            )
        
        # Update image data to be the combined result (grayscale)
        image.data = result
        
        # Update metadata
        image.metadata.add_step({
            "technique": "combine_channels",
            "channel1": self.channel1,
            "channel2": self.channel2,
            "weight": self.weight,
            "output_shape": image.data.shape
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": "combine_channels",
            "channel1": self.channel1,
            "channel2": self.channel2,
            "weight": self.weight
        }
    
class RedBlueFilterStep(ProcessingStep):
    """Filter red and blue colors in a BGR image."""
    
    def __init__(self):
        pass

    def process(self, image: Image) -> Image:
        # Precondition validation
        if image.data is None:
            raise ValidationError("RedBlueFilterStep: Image.data is None")
        if not image.is_color:
            raise ValidationError(f"RedBlueFilterStep: Expected color image, but got shape {image.data.shape}")
        
        # Ensure data is contiguous
        if not image.data.flags['C_CONTIGUOUS']:
            image.data = np.ascontiguousarray(image.data)

        # Processing
        result = filtro_rojo_azul(image.data)
        
        # Post-processing validation
        if result is None:
            raise ValidationError("RedBlueFilterStep: Output is None")
        if not isinstance(result, np.ndarray):
            raise ValidationError(
                f"RedBlueFilterStep: Expected numpy array output, got {type(result)}"
            )
        if result.ndim != 2:
            raise ValidationError(
                f"RedBlueFilterStep: Expected 2D output, got {result.ndim}D"
            )
        
        image.data = result
        
        # Update metadata
        image.metadata.add_step({
            "technique": "red_blue_filter",
            "output_shape": image.data.shape
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": "red_blue_filter",
        }

class FilterMaskConnectedComponentsStep(ProcessingStep):
    """Filter connected components in a binary mask by size."""
    
    def __init__(self, min_size: int = 20):
        if min_size < 0:
            raise ValueError("min_size must be non-negative")
        self.min_size = min_size

    def process(self, image: Image) -> Image:
        # Precondition validation
        if image.data is None:
            raise ValidationError("FilterMaskConnectedComponentsStep: Image.data is None")
        if not image.is_grayscale:
            raise ValidationError(
                f"FilterMaskConnectedComponentsStep: Expected grayscale image, "
                f"but got shape {image.data.shape}"
            )
        
        # Ensure data is contiguous
        if not image.data.flags['C_CONTIGUOUS']:
            image.data = np.ascontiguousarray(image.data)

        # Processing
        filtered_mask = filter_connected_components(image.data, self.min_size)
        
        # Post-processing validation
        if filtered_mask is None:
            raise ValidationError("FilterMaskConnectedComponentsStep: Output is None")
        if not isinstance(filtered_mask, np.ndarray):
            raise ValidationError(
                f"FilterMaskConnectedComponentsStep: Expected numpy array output, got {type(filtered_mask)}"
            )
        if filtered_mask.ndim != 2:
            raise ValidationError(
                f"FilterMaskConnectedComponentsStep: Expected 2D output, got {filtered_mask.ndim}D"
            )
        if filtered_mask.dtype != np.uint8:
            raise ValidationError(
                f"FilterMaskConnectedComponentsStep: Expected uint8 output, got {filtered_mask.dtype}"
            )
        if filtered_mask.shape != image.data.shape:
            raise ValidationError(
                f"FilterMaskConnectedComponentsStep: Output shape {filtered_mask.shape} "
                f"doesn't match input shape {image.data.shape}"
            )
        
        image.data = filtered_mask
        
        # Update metadata
        image.metadata.add_step({
            "technique": "filter_connected_components",
            "min_size": self.min_size,
            "output_shape": image.data.shape
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": "filter_connected_components",
            "min_size": self.min_size
        }

class IntoBooleanMaskStep(ProcessingStep):
    """Convert a grayscale image to a boolean mask."""

    def __init__(self):
        pass

    def process(self, image: Image) -> Image:
        # Precondition validation
        if image.data is None:
            raise ValidationError("IntoBooleanMaskStep: Image.data is None")
        if not image.is_grayscale:
            raise ValidationError(f"IntoBooleanMaskStep: Expected grayscale image, but got shape {image.data.shape}")
        
        # Convert to boolean
        bool_mask = image.data.astype(np.bool_)
        
        # Post-processing validation
        if bool_mask.ndim != 2:
            raise ValidationError(
                f"IntoBooleanMaskStep: Expected 2D output, got {bool_mask.ndim}D"
            )
        if bool_mask.dtype != np.bool_:
            raise ValidationError(
                f"IntoBooleanMaskStep: Expected bool dtype, got {bool_mask.dtype}"
            )
        
        image.data = bool_mask
        
        # Update metadata
        image.metadata.add_step({
            "technique": "into_boolean_mask",
            "output_shape": image.data.shape,
        })
        
        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": "into_boolean_mask",
        }


class RotationStep(ProcessingStep):
    """Apply rotation transform to an image.
    
    Useful for testing detector/descriptor invariance to rotation.
    """

    def __init__(self, angle_degrees: float, keep_size: bool = True):
        """Initialize rotation step.
        
        Args:
            angle_degrees: Angle of rotation in degrees (positive = counter-clockwise).
            keep_size: If True, output has same size as input (may crop corners).
                      If False, output is expanded to fit rotated image.
        """
        self.angle_degrees = float(angle_degrees)
        self.keep_size = keep_size

    def process(self, image: Image) -> Image:
        if image.data is None:
            raise ValidationError("RotationStep: Image.data is None")

        h, w = image.data.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, self.angle_degrees, 1.0)
        
        if self.keep_size:
            new_w, new_h = w, h
        else:
            # Calculate new image bounds
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            # Adjust rotation matrix
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

        result = cv2.warpAffine(image.data, M, (new_w, new_h))

        if result is None:
            raise ValidationError("RotationStep: Output is None")

        image.data = result
        image.metadata.add_step({
            "technique": ProcessingTechnique.ROTATION,
            "angle_degrees": self.angle_degrees,
            "keep_size": self.keep_size,
            "output_shape": image.data.shape,
        })

        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": ProcessingTechnique.ROTATION,
            "angle_degrees": self.angle_degrees,
            "keep_size": self.keep_size,
        }


class ScaleTransformStep(ProcessingStep):
    """Apply scale transform to an image.
    
    Useful for testing detector/descriptor invariance to scale changes.
    """

    def __init__(self, scale_factor: float, restore_size: bool = False):
        """Initialize scale step.
        
        Args:
            scale_factor: Factor to scale by (e.g., 0.5 = half size, 2.0 = double).
            restore_size: If True, scale down/up and then restore to original size.
                         This simulates resolution loss while maintaining dimensions.
        """
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        self.scale_factor = float(scale_factor)
        self.restore_size = restore_size

    def process(self, image: Image) -> Image:
        if image.data is None:
            raise ValidationError("ScaleTransformStep: Image.data is None")

        original_h, original_w = image.data.shape[:2]
        new_w = int(original_w * self.scale_factor)
        new_h = int(original_h * self.scale_factor)

        # Choose interpolation based on scaling direction
        if self.scale_factor < 1.0:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR

        scaled = cv2.resize(image.data, (new_w, new_h), interpolation=interp)

        if self.restore_size:
            # Restore to original size
            result = cv2.resize(scaled, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        else:
            result = scaled

        if result is None:
            raise ValidationError("ScaleTransformStep: Output is None")

        image.data = result
        image.metadata.add_step({
            "technique": ProcessingTechnique.SCALE,
            "scale_factor": self.scale_factor,
            "restore_size": self.restore_size,
            "output_shape": image.data.shape,
        })

        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": ProcessingTechnique.SCALE,
            "scale_factor": self.scale_factor,
            "restore_size": self.restore_size,
        }


class GaussianNoiseStep(ProcessingStep):
    """Add Gaussian noise to an image.
    
    Useful for testing detector/descriptor robustness to noise.
    """

    def __init__(self, sigma: float = 25.0, seed: int | None = None):
        """Initialize Gaussian noise step.
        
        Args:
            sigma: Standard deviation of the Gaussian noise.
            seed: Random seed for reproducibility. If None, uses random state.
        """
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.sigma = float(sigma)
        self.seed = seed

    def process(self, image: Image) -> Image:
        if image.data is None:
            raise ValidationError("GaussianNoiseStep: Image.data is None")

        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0, self.sigma, image.data.shape).astype(np.float32)
        
        noisy = image.data.astype(np.float32) + noise
        result = np.clip(noisy, 0, 255).astype(np.uint8)

        if result is None:
            raise ValidationError("GaussianNoiseStep: Output is None")

        image.data = result
        image.metadata.add_step({
            "technique": ProcessingTechnique.GAUSSIAN_NOISE,
            "sigma": self.sigma,
            "seed": self.seed,
            "output_shape": image.data.shape,
        })

        return image

    def get_params(self) -> dict[str, Any]:
        return {
            "technique": ProcessingTechnique.GAUSSIAN_NOISE,
            "sigma": self.sigma,
            "seed": self.seed,
        }



class ImageProcessor:
    """Flexible image processor for chaining filters and edge detection.

    Example:
        >>> processor = ImageProcessor()
        >>> processor.add_gaussian_filter(sigma=1.4)
        >>> processor.add_canny_edge_detection(low_threshold=50, high_threshold=150)
        >>> result = processor.process(image)

    Or using the builder pattern:
        >>> processor = (ImageProcessor()
        ...     .add_grayscale()
        ...     .add_gaussian_filter(sigma=1.4)
        ...     .add_canny_edge_detection(low_threshold=50, high_threshold=150))
        >>> result = processor.process(image)
    """

    def __init__(self):
        self.steps: list[ProcessingStep] = []
        self.original_image: Image | None = None
        self.processed_image: Image | None = None
        self.intermediate_results: list[Image] = []

    def add_step(self, step: ProcessingStep) -> "ImageProcessor":
        """Add a processing step to the pipeline.

        Args:
            step: ProcessingStep instance

        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        return self

    def add_grayscale(self) -> "ImageProcessor":
        """Add grayscale conversion step.

        Returns:
            Self for method chaining
        """
        return self.add_step(GrayscaleStep())

    def add_box_filter(self, filter_size: int = 5) -> "ImageProcessor":
        """Add box filter step.

        Args:
            filter_size: Size of box filter kernel (must be odd)

        Returns:
            Self for method chaining
        """
        return self.add_step(BoxFilterStep(filter_size))

    def add_gaussian_filter(self, sigma: float = 1.0) -> 'ImageProcessor':
        """Add Gaussian filter step.

        Args:
            sigma: Standard deviation of Gaussian kernel

        Returns:
            Self for method chaining
        """
        return self.add_step(GaussianFilterStep(sigma))
    
    def add_canny_edge_detection(self, low_threshold: float = 50.0, 
                                  high_threshold: float = 150.0) -> 'ImageProcessor':
        """Add Canny edge detection step.

        Note: For best results, apply Gaussian smoothing before Canny edge detection.

        Args:
            low_threshold: Lower threshold for hysteresis (weak edges)
            high_threshold: Upper threshold for hysteresis (strong edges)
        Returns:
            Self for method chaining
        """
        return self.add_step(CannyEdgeDetectionStep(low_threshold, high_threshold))
    
    def add_kannala_brandt_undistortion(self) -> 'ImageProcessor':
        """Add Kannala-Brandt undistortion step.

        The calibration data will be taken from the image being processed.

        Returns:
            Self for method chaining
        """
        return self.add_step(KannalaBrandtUndistortionStep())

    def add_phase_congruency(
        self,
        nscale: int = 4,
        norient: int = 6,
        min_wavelength: float = 3.0,
        mult: float = 2.1,
        sigma_onf: float = 0.55,
        use_own: bool = False,
    ) -> "ImageProcessor":
        """Add phase congruency computation step.

        Args:
            nscale: Number of scales
            norient: Number of orientations
            min_wavelength: Smallest filter wavelength
            mult: Scaling factor between successive wavelengths
            sigma_onf: Bandwidth parameter for log-Gabor

        Returns:
            Self for method chaining
        """
        return self.add_step(
            PhaseCongruencyStep(
                nscale=nscale,
                norient=norient,
                min_wavelength=min_wavelength,
                mult=mult,
                sigma_onf=sigma_onf,
                use_own=use_own,
            )
        )

    def add_threshold(self, threshold: float = 0.5) -> "ImageProcessor":
        """Add threshold filter step.

        The input to this step should be a 2D image. If it's uint8, it will be
        converted to float32 in [0,1] before thresholding.
        """
        return self.add_step(ThresholdFilterStep(threshold))

    def add_clahe(
        self, clip_limit: float = 2.0, tile_grid_size=(8, 8)
    ) -> "ImageProcessor":
        """Add CLAHE (adaptive histogram equalization) step.

        Args:
            clip_limit: Contrast limit for CLAHE.
            tile_grid_size: Grid size (width, height) for CLAHE tiles.

        Returns:
            Self for method chaining
        """
        return self.add_step(
            CLAHEStep(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        )

    def add_otsu_threshold(self) -> "ImageProcessor":
        """Add Otsu automatic threshold step.

        Returns:
            Self for method chaining
        """
        return self.add_step(OtsuThresholdStep())

    def add_dilate_edges(self, kernel_size: int = 3, iterations: int = 1) -> 'ImageProcessor':
        """Add edge dilation step.

        Args:
            kernel_size: Size of the structuring element (must be odd)
            iterations: Number of dilation iterations

        Returns:
            Self for method chaining
        """
        return self.add_step(DilateEdgesStep(kernel_size=kernel_size, iterations=iterations))

    def add_scale_inter_area(self, scale_factor: float) -> 'ImageProcessor':
        """Add image scaling step using area interpolation.

        Args:
            scale_factor: Factor to scale the image by (e.g., 0.5 reduces size by half)

        Returns:
            Self for method chaining
        """
        return self.add_step(ScaleInterAreaStep(scale_factor=scale_factor))

    def add_median_blur(self, kernel_size: int) -> 'ImageProcessor':
        """Add median blur step.

        Args:
            kernel_size: Size of the median filter kernel (must be odd and > 1)

        Returns:
            Self for method chaining
        """
        return self.add_step(MedianBlurStep(kernel_size=kernel_size))

    def add_into_hsv_channels(self) -> 'ImageProcessor':
        """Add step to convert BGR image into its HSV channels.

        Returns:
            Self for method chaining
        """
        return self.add_step(IntoHSVChannelsStep())

    def add_channel_weight(self, channel1: str, channel2: str, weight: float) -> "ImageProcessor":
        """Add step to combine two channels with a specified weight for the second channel.

        Args:
            channel1: First channel to combine ('H', 'S', or 'V')
            channel2: Second channel to combine ('H', 'S', or 'V')
            weight: Weight to apply to the second channel (between 0 and 1)

        Returns:
            Self for method chaining
        """
        return self.add_step(CombineChannelsStep(channel1, channel2, weight))

    def add_red_blue_filter(self) -> "ImageProcessor":
        """Add step to filter red and blue colors in a BGR image.

        Returns:
            Self for method chaining
        """
        return self.add_step(RedBlueFilterStep())
    
    def add_filter_connected_components(self, min_size: int = 20) -> "ImageProcessor":
        """Add step to filter out small connected components in a binary mask.

        Args:
            min_size: Minimum size of connected components to keep
        Returns:
            Self for method chaining
        """
        return self.add_step(FilterMaskConnectedComponentsStep(min_size=min_size))

    def add_into_boolean_mask(self) -> "ImageProcessor":
        """Add step to convert a grayscale image to a boolean mask.

        Returns:
            Self for method chaining
        """
        return self.add_step(IntoBooleanMaskStep())

    def add_rotation(self, angle_degrees: float, keep_size: bool = True) -> "ImageProcessor":
        """Add rotation transform step.

        Args:
            angle_degrees: Angle of rotation in degrees (positive = counter-clockwise).
            keep_size: If True, output has same size as input (may crop corners).

        Returns:
            Self for method chaining
        """
        return self.add_step(RotationStep(angle_degrees, keep_size))

    def add_scale_transform(self, scale_factor: float, restore_size: bool = False) -> "ImageProcessor":
        """Add scale transform step.

        Args:
            scale_factor: Factor to scale by (e.g., 0.5 = half size, 2.0 = double).
            restore_size: If True, scale and restore to original size (simulates resolution loss).

        Returns:
            Self for method chaining
        """
        return self.add_step(ScaleTransformStep(scale_factor, restore_size))

    def add_gaussian_noise(self, sigma: float = 25.0, seed: int | None = None) -> "ImageProcessor":
        """Add Gaussian noise step.

        Args:
            sigma: Standard deviation of the noise.
            seed: Random seed for reproducibility.

        Returns:
            Self for method chaining
        """
        return self.add_step(GaussianNoiseStep(sigma, seed))


    def process(self, image: Image, 
                keep_intermediate: bool = False) -> Image:
        """Process image through all steps in the pipeline.

        Args:
            image: Input Image object
            keep_intermediate: Store intermediate results for debugging

        Returns:
            Final processed Image object
        """
        # Store original image with all metadata
        self.original_image = image.clone()

        # Create a working copy with all metadata preserved
        current = image.clone()

        for step in self.steps:
            current = step.process(current)
            if keep_intermediate:
                self.intermediate_results.append(current.clone())

        self.processed_image = current
        return current

    def get_intermediate_results(self) -> list[Image]:
        """Get intermediate results from last processing run.

        Returns:
            List of Image objects after each processing step
        """
        return self.intermediate_results

    def get_pipeline_info(self) -> list[dict[str, Any]]:
        """Get information about all steps in the pipeline.

        Returns:
            List of dictionaries describing each step
        """
        return [step.get_params() for step in self.steps]

    def clear(self) -> "ImageProcessor":
        """Clear all processing steps."""
        self.steps = []
        return self

    def __len__(self) -> int:
        """Return number of steps in pipeline."""
        return len(self.steps)

    def __repr__(self) -> str:
        """String representation of the processor."""
        return f"ImageProcessor(steps={len(self.steps)})"
