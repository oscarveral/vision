import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from dgst.filters.ffi import box_filter, gaussian_filter, canny_edge_detection, kannala_brandt_undistort, kannala_brandt_map_points_to_undistorted, phase_congruency as pc_ffi, threshold_filter
from dgst.filters.python import phase_congruency as pc_python, otsu_threshold, clahe_filter, dilate_edges, scale_inter_area, median_blur, into_hsv_channels, add_channel_weight, filtro_rojo_azul, filter_connected_components
from dgst.utils.loader import Image, RegionOfInterest, ImageFormat
from dgst.utils.validation import ImageValidator, FormatConverter, ValidationError

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
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters of the processing step."""
        pass


class GrayscaleStep(ProcessingStep):
    """Convert image to grayscale."""

    def process(self, image: Image) -> Image:
        # Precondition validation
        ImageValidator.validate_format(image, ImageFormat.BGR, "GrayscaleStep")
        ImageValidator.validate_bgr_image(image, "GrayscaleStep")

        # Processing
        if len(image.data.shape) == 3:
            image.data = cv2.cvtColor(image.data, cv2.COLOR_BGR2GRAY)
            image.format = ImageFormat.GRAYSCALE
        
        # Post-processing validation
        ImageValidator.validate_data_dimensions(image, 2, "GrayscaleStep")
        ImageValidator.validate_data_type(image, np.uint8, "GrayscaleStep")
        
        # Update metadata
        image.metadata.add_step({
            "technique": ProcessingTechnique.GRAYSCALE,
            "output_shape": image.data.shape,
            "output_format": image.format.name
        })
        
        return image

    def get_params(self) -> Dict[str, Any]:
        return {"technique": ProcessingTechnique.GRAYSCALE}


class BoxFilterStep(ProcessingStep):
    """Apply box filter using custom C implementation."""

    def __init__(self, filter_size: int = 5):
        if filter_size % 2 == 0:
            raise ValueError("filter_size must be odd")
        self.filter_size = filter_size

    def process(self, image: Image) -> Image:
        # Precondition validation
        ImageValidator.validate_grayscale_image(image, "BoxFilterStep")
        
        # Ensure data is contiguous for C function
        image.data = FormatConverter.ensure_contiguous(image.data)

        # Processing
        image.data = box_filter(image.data, self.filter_size)
        
        # Post-processing validation
        ImageValidator.validate_data_dimensions(image, 2, "BoxFilterStep")
        ImageValidator.validate_data_type(image, np.uint8, "BoxFilterStep")
        
        # Update metadata
        image.metadata.add_step({
            "technique": ProcessingTechnique.BOX_FILTER,
            "filter_size": self.filter_size,
            "output_shape": image.data.shape
        })
        
        return image

    def get_params(self) -> Dict[str, Any]:
        return {
            "technique": ProcessingTechnique.BOX_FILTER,
            "filter_size": self.filter_size,
        }


class GaussianFilterStep(ProcessingStep):
    """Apply Gaussian filter using custom C implementation."""
    
    def __init__(self, sigma: float = 1.0, on_hsv: bool = False):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = sigma
        self.on_hsv = on_hsv

    def process(self, image: Image) -> Image:
        if self.on_hsv:
            # Precondition validation for HSV mode
            ImageValidator.validate_hsv_channels(image, "GaussianFilterStep", expected_channels=3)
            
            hsv_filtered = []
            for idx, channel in enumerate(image.hsv_channels):
                # Validate each channel
                if channel.dtype != np.uint8:
                    raise ValidationError(
                        f"GaussianFilterStep: HSV channel {idx} must be uint8, got {channel.dtype}"
                    )
                if channel.ndim != 2:
                    raise ValidationError(
                        f"GaussianFilterStep: HSV channel {idx} must be 2D, got {channel.ndim}D"
                    )
                
                # Ensure contiguous
                channel = FormatConverter.ensure_contiguous(channel)
                filtered = gaussian_filter(channel, self.sigma)
                hsv_filtered.append(filtered)
            
            image.hsv_channels = hsv_filtered
            
            # Update metadata
            image.metadata.add_step({
                "technique": ProcessingTechnique.GAUSSIAN_FILTER,
                "sigma": self.sigma,
                "on_hsv": True,
                "channels_processed": len(hsv_filtered)
            })
        else:
            # Precondition validation for grayscale mode
            ImageValidator.validate_grayscale_image(image, "GaussianFilterStep")
            
            # Ensure data is contiguous for C function
            image.data = FormatConverter.ensure_contiguous(image.data)

            # Processing
            image.data = gaussian_filter(image.data, self.sigma)
            
            # Post-processing validation
            ImageValidator.validate_data_dimensions(image, 2, "GaussianFilterStep")
            ImageValidator.validate_data_type(image, np.uint8, "GaussianFilterStep")
            
            # Update metadata
            image.metadata.add_step({
                "technique": ProcessingTechnique.GAUSSIAN_FILTER,
                "sigma": self.sigma,
                "on_hsv": False,
                "output_shape": image.data.shape
            })
        
        return image

    def get_params(self) -> Dict[str, Any]:
        return {
            "technique": ProcessingTechnique.GAUSSIAN_FILTER,
            "sigma": self.sigma,
        }


class CannyEdgeDetectionStep(ProcessingStep):
    """Apply Canny edge detection using custom C implementation."""
    
    def __init__(self, low_threshold: float = 50.0, high_threshold: float = 150.0, on_hsv: bool = False):
        if high_threshold < low_threshold:
            raise ValueError("high_threshold must be >= low_threshold")
        if low_threshold < 0:
            raise ValueError("low_threshold must be >= 0")
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.on_hsv = on_hsv

    def process(self, image: Image) -> Image:
        if self.on_hsv:
            # Precondition validation for HSV mode
            ImageValidator.validate_hsv_channels(image, "CannyEdgeDetectionStep", expected_channels=3)
            
            hsv_edges = []
            for idx, channel in enumerate(image.hsv_channels):
                # Validate each channel
                if channel.dtype != np.uint8:
                    raise ValidationError(
                        f"CannyEdgeDetectionStep: HSV channel {idx} must be uint8, got {channel.dtype}"
                    )
                if channel.ndim != 2:
                    raise ValidationError(
                        f"CannyEdgeDetectionStep: HSV channel {idx} must be 2D, got {channel.ndim}D"
                    )
                
                # Ensure contiguous
                channel = FormatConverter.ensure_contiguous(channel)
                edges = canny_edge_detection(channel, self.low_threshold, self.high_threshold)
                hsv_edges.append(edges)
            
            image.hsv_channels = hsv_edges
            
            # Update metadata
            image.metadata.add_step({
                "technique": ProcessingTechnique.CANNY_EDGE_DETECTION,
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold,
                "on_hsv": True,
                "channels_processed": len(hsv_edges)
            })
        else:
            # Precondition validation for grayscale mode
            ImageValidator.validate_grayscale_image(image, "CannyEdgeDetectionStep")
            
            # Ensure data is contiguous for C function
            image.data = FormatConverter.ensure_contiguous(image.data)
            
            # Processing
            image.data = canny_edge_detection(image.data, self.low_threshold, self.high_threshold)
            
            # Post-processing validation
            ImageValidator.validate_data_dimensions(image, 2, "CannyEdgeDetectionStep")
            ImageValidator.validate_data_type(image, np.uint8, "CannyEdgeDetectionStep")
            
            # Update metadata
            image.metadata.add_step({
                "technique": ProcessingTechnique.CANNY_EDGE_DETECTION,
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold,
                "on_hsv": False,
                "output_shape": image.data.shape
            })

        return image

    def get_params(self) -> Dict[str, Any]:
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
        ImageValidator.validate_calibration(image, "KannalaBrandtUndistortionStep")
        ImageValidator.validate_calibration_type(image, "kannala", "KannalaBrandtUndistortionStep")
        ImageValidator.validate_bgr_image(image, "KannalaBrandtUndistortionStep")
        
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
        image.data = FormatConverter.ensure_contiguous(image.data)

        # Apply the undistortion using C implementation
        original_shape = image.data.shape
        image.data = kannala_brandt_undistort(image.data, K, D)
        
        # Post-processing validation
        ImageValidator.validate_bgr_image(image, "KannalaBrandtUndistortionStep")
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

    def get_params(self) -> Dict[str, Any]:
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
        ImageValidator.validate_grayscale_image(image, "PhaseCongruencyStep")
        
        # Ensure data is contiguous
        image.data = FormatConverter.ensure_contiguous(image.data)

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
        ImageValidator.validate_data_not_none(image, "PhaseCongruencyStep")
        ImageValidator.validate_data_dimensions(image, 2, "PhaseCongruencyStep")
        
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

    def get_params(self) -> Dict[str, Any]:
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
    """Threshold a float image using the C implementation.

    This step expects a 2D image. If the input is uint8 (0-255), it will be
    converted to float32 in [0,1] before calling the C function. Output is
    a float32 2D array with values 0.0 or 1.0.
    """

    def __init__(self, threshold: float = 0.5):
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = float(threshold)

    def process(self, image: Image) -> Image:
        # Precondition validation
        ImageValidator.validate_grayscale_image(image, "ThresholdFilterStep")
        
        # Ensure data is contiguous for C function
        image.data = FormatConverter.ensure_contiguous(image.data)

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

    def get_params(self) -> Dict[str, Any]:
        return {
            "technique": ProcessingTechnique.THRESHOLD_FILTER,
            "threshold": self.threshold,
        }


class CLAHEStep(ProcessingStep):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    This step will apply CLAHE to grayscale images or to the L channel of
    a BGR color image (preserving color information).
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size=(8, 8)):
        if clip_limit <= 0:
            raise ValueError("clip_limit must be positive")
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = (int(tile_grid_size[0]), int(tile_grid_size[1]))

    def process(self, image: Image) -> Image:
        # Precondition validation
        ImageValidator.validate_data_not_none(image, "CLAHEStep")
        ImageValidator.validate_multiple_formats(
            image, [ImageFormat.BGR, ImageFormat.GRAYSCALE], "CLAHEStep"
        )
        
        # Validate dimensions based on format
        if image.format == ImageFormat.BGR:
            ImageValidator.validate_bgr_image(image, "CLAHEStep")
        else:
            ImageValidator.validate_grayscale_image(image, "CLAHEStep")
        
        # Ensure data is contiguous
        image.data = FormatConverter.ensure_contiguous(image.data)

        # Processing
        original_format = image.format
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
            "input_format": original_format.name,
            "output_shape": image.data.shape
        })
        
        return image

    def get_params(self) -> Dict[str, Any]:
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
        ImageValidator.validate_grayscale_image(image, "OtsuThresholdStep")
        
        # Ensure data is contiguous
        image.data = FormatConverter.ensure_contiguous(image.data)

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

    def get_params(self) -> Dict[str, Any]:
        return {"technique": ProcessingTechnique.OTSU_THRESHOLD}
    
class DilateEdgesStep(ProcessingStep):
    """Dilate edges in a binary edge image."""
    
    def __init__(self, kernel_size: int = 3, iterations: int = 1, on_hsv: bool = False):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.on_hsv = on_hsv

    def process(self, image: Image) -> Image:
        if self.on_hsv:
            # Precondition validation for HSV mode
            ImageValidator.validate_hsv_channels(image, "DilateEdgesStep", expected_channels=3)
            
            hsv_dilated = []
            for idx, channel in enumerate(image.hsv_channels):
                # Validate each channel
                if channel.ndim != 2:
                    raise ValidationError(
                        f"DilateEdgesStep: HSV channel {idx} must be 2D, got {channel.ndim}D"
                    )
                
                # Ensure contiguous
                channel = FormatConverter.ensure_contiguous(channel)
                dilated = dilate_edges(channel, self.kernel_size, self.iterations)
                hsv_dilated.append(dilated)
            
            image.hsv_channels = hsv_dilated
            
            # Update metadata
            image.metadata.add_step({
                "technique": "dilate_edges",
                "kernel_size": self.kernel_size,
                "iterations": self.iterations,
                "on_hsv": True,
                "channels_processed": len(hsv_dilated)
            })
        else:
            # Precondition validation for grayscale mode
            ImageValidator.validate_data_not_none(image, "DilateEdgesStep")
            ImageValidator.validate_format(image, ImageFormat.GRAYSCALE, "DilateEdgesStep")
            ImageValidator.validate_data_dimensions(image, 2, "DilateEdgesStep")
            
            # Ensure contiguous
            image.data = FormatConverter.ensure_contiguous(image.data)

            # Processing
            result = dilate_edges(image.data, self.kernel_size, self.iterations)
            
            # Post-processing validation
            if result is None:
                raise ValidationError("DilateEdgesStep: Output is None")
            if result.ndim != 2:
                raise ValidationError(
                    f"DilateEdgesStep: Expected 2D output, got {result.ndim}D"
                )
            
            image.data = np.where(result > 0, 255, 0).astype(np.uint8)
            image.format = ImageFormat.GRAYSCALE
            
            # Update metadata
            image.metadata.add_step({
                "technique": "dilate_edges",
                "kernel_size": self.kernel_size,
                "iterations": self.iterations,
                "on_hsv": False,
                "output_shape": image.data.shape
            })

        return image

    def get_params(self) -> Dict[str, Any]:
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
        ImageValidator.validate_data_not_none(image, "ScaleInterAreaStep")
        
        original_shape = image.data.shape
        
        # Ensure data is contiguous
        image.data = FormatConverter.ensure_contiguous(image.data)

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

    def get_params(self) -> Dict[str, Any]:
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
        ImageValidator.validate_data_not_none(image, "MedianBlurStep")
        
        original_shape = image.data.shape
        original_dtype = image.data.dtype
        
        # Ensure data is contiguous
        image.data = FormatConverter.ensure_contiguous(image.data)

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

    def get_params(self) -> Dict[str, Any]:
        return {
            "technique": "median_blur",
            "kernel_size": self.kernel_size
        }
    
class IntoHSVChannelsStep(ProcessingStep):
    """Convert BGR image into its HSV channels."""
    
    def __init__(self):
        pass

    def process(self, image: Image) -> Image:
        # Precondition validation
        ImageValidator.validate_bgr_image(image, "IntoHSVChannelsStep")
        
        # Ensure data is contiguous
        image.data = FormatConverter.ensure_contiguous(image.data)

        # Processing
        hsv_channels = into_hsv_channels(image.data)
        
        # Post-processing validation
        if hsv_channels is None:
            raise ValidationError("IntoHSVChannelsStep: Output is None")
        if not isinstance(hsv_channels, np.ndarray):
            raise ValidationError(
                f"IntoHSVChannelsStep: Expected numpy array output, got {type(hsv_channels)}"
            )
        if hsv_channels.ndim != 3:
            raise ValidationError(
                f"IntoHSVChannelsStep: Expected 3D array, got {hsv_channels.ndim}D"
            )
        if hsv_channels.shape[2] != 3:
            raise ValidationError(
                f"IntoHSVChannelsStep: Expected 3 channels, got {hsv_channels.shape[2]}"
            )
        
        # Split into individual channel arrays
        image.hsv_channels = [hsv_channels[:, :, 0], hsv_channels[:, :, 1], hsv_channels[:, :, 2]]
        
        # Validate individual channels
        for idx, channel in enumerate(image.hsv_channels):
            if channel.ndim != 2:
                raise ValidationError(
                    f"IntoHSVChannelsStep: Channel {idx} is not 2D"
                )
            if channel.dtype != np.uint8:
                raise ValidationError(
                    f"IntoHSVChannelsStep: Channel {idx} is not uint8"
                )
        
        # Update metadata
        image.metadata.add_step({
            "technique": "into_hsv_channels",
            "num_channels": len(image.hsv_channels),
            "channel_shapes": [ch.shape for ch in image.hsv_channels]
        })
        
        return image

    def get_params(self) -> Dict[str, Any]:
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

    def process(self, image: Image) -> Image:
        # Precondition validation
        ImageValidator.validate_hsv_channels(image, "CombineChannelsStep", expected_channels=3)

        # Get channel indices
        ch1_idx = {'H': 0, 'S': 1, 'V': 2}[self.channel1]
        ch2_idx = {'H': 0, 'S': 1, 'V': 2}[self.channel2]

        ch1 = image.hsv_channels[ch1_idx]
        ch2 = image.hsv_channels[ch2_idx]
        
        # Validate channels
        if ch1.dtype != np.uint8 or ch2.dtype != np.uint8:
            raise ValidationError(
                f"CombineChannelsStep: Both channels must be uint8, "
                f"got {ch1.dtype} and {ch2.dtype}"
            )
        if ch1.shape != ch2.shape:
            raise ValidationError(
                f"CombineChannelsStep: Channel shapes must match, "
                f"got {ch1.shape} and {ch2.shape}"
            )

        # Ensure contiguous
        ch1 = FormatConverter.ensure_contiguous(ch1)
        ch2 = FormatConverter.ensure_contiguous(ch2)

        # Processing
        combined_channel = add_channel_weight(ch1, ch2, self.weight)
        
        # Post-processing validation
        if combined_channel is None:
            raise ValidationError("CombineChannelsStep: Output is None")
        if not isinstance(combined_channel, np.ndarray):
            raise ValidationError(
                f"CombineChannelsStep: Expected numpy array output, got {type(combined_channel)}"
            )
        if combined_channel.ndim != 2:
            raise ValidationError(
                f"CombineChannelsStep: Expected 2D output, got {combined_channel.ndim}D"
            )
        
        image.data = combined_channel
        image.format = ImageFormat.GRAYSCALE
        
        # Update metadata
        image.metadata.add_step({
            "technique": "combine_channels",
            "channel1": self.channel1,
            "channel2": self.channel2,
            "weight": self.weight,
            "output_shape": image.data.shape,
            "output_format": image.format.name
        })
        
        return image

    def get_params(self) -> Dict[str, Any]:
        return {
            "technique": "combine_channels",
            "weight": self.weight
        }
    
class RedBlueFilterStep(ProcessingStep):
    """Filter red and blue colors in a BGR image."""

    def __init__(self):
        pass

    def process(self, image: Image) -> Image:
        # Precondition validation
        ImageValidator.validate_bgr_image(image, "RedBlueFilterStep")
        
        # Ensure data is contiguous
        image.data = FormatConverter.ensure_contiguous(image.data)

        # Processing
        mask = filtro_rojo_azul(image.data)
        
        # Post-processing validation
        if mask is None:
            raise ValidationError("RedBlueFilterStep: Output is None")
        if not isinstance(mask, np.ndarray):
            raise ValidationError(
                f"RedBlueFilterStep: Expected numpy array output, got {type(mask)}"
            )
        if mask.ndim != 2:
            raise ValidationError(
                f"RedBlueFilterStep: Expected 2D output, got {mask.ndim}D"
            )
        if mask.dtype != np.uint8:
            raise ValidationError(
                f"RedBlueFilterStep: Expected uint8 output, got {mask.dtype}"
            )
        
        image.data = mask
        image.format = ImageFormat.GRAYSCALE
        
        # Update metadata
        image.metadata.add_step({
            "technique": "red_blue_filter",
            "output_shape": image.data.shape,
            "output_format": image.format.name
        })
        
        return image

    def get_params(self) -> Dict[str, Any]:
        return {
            "technique": "red_blue_filter",
        }

class FilterMaskConnectedComponentsStep(ProcessingStep):
    """Filter out small connected components in a binary mask."""

    def __init__(self, min_size: int = 20):
        if min_size < 1:
            raise ValueError("min_size must be at least 1")
        self.min_size = min_size

    def process(self, image: Image) -> Image:
        # Precondition validation
        ImageValidator.validate_grayscale_image(image, "FilterMaskConnectedComponentsStep")
        
        # Ensure data is contiguous
        image.data = FormatConverter.ensure_contiguous(image.data)

        # Processing
        filtered_mask = filter_connected_components(image.data, min_size=self.min_size)
        
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

    def get_params(self) -> Dict[str, Any]:
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
        ImageValidator.validate_format(image, ImageFormat.GRAYSCALE, "IntoBooleanMaskStep")
        ImageValidator.validate_data_not_none(image, "IntoBooleanMaskStep")
        ImageValidator.validate_data_dimensions(image, 2, "IntoBooleanMaskStep")
        
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
        image.format = ImageFormat.BOOLEAN
        
        # Update metadata
        image.metadata.add_step({
            "technique": "into_boolean_mask",
            "output_shape": image.data.shape,
            "output_format": image.format.name
        })
        
        return image

    def get_params(self) -> Dict[str, Any]:
        return {
            "technique": "into_boolean_mask",
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
        self.steps: List[ProcessingStep] = []
        self.original_image: Optional[Image] = None
        self.processed_image: Optional[Image] = None
        self.intermediate_results: List[Image] = []

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

    def add_gaussian_filter(self, sigma: float = 1.0, on_hsv: bool = False) -> 'ImageProcessor':
        """Add Gaussian filter step.

        Args:
            sigma: Standard deviation of Gaussian kernel

        Returns:
            Self for method chaining
        """
        return self.add_step(GaussianFilterStep(sigma, on_hsv))
    
    def add_canny_edge_detection(self, low_threshold: float = 50.0, 
                                  high_threshold: float = 150.0, on_hsv: bool = False) -> 'ImageProcessor':
        """Add Canny edge detection step.

        Note: For best results, apply Gaussian smoothing before Canny edge detection.

        Args:
            low_threshold: Lower threshold for hysteresis (weak edges)
            high_threshold: Upper threshold for hysteresis (strong edges)
            on_hsv: If True, the Canny edge detection will be applied on the HSV channels
        Returns:
            Self for method chaining
        """
        return self.add_step(CannyEdgeDetectionStep(low_threshold, high_threshold, on_hsv))
    
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

    def add_dilate_edges(self, kernel_size: int = 3, iterations: int = 1, on_hsv: bool = False) -> 'ImageProcessor':
        """Add edge dilation step.

        Args:
            kernel_size: Size of the structuring element (must be odd)
            iterations: Number of dilation iterations

        Returns:
            Self for method chaining
        """
        return self.add_step(DilateEdgesStep(kernel_size=kernel_size, iterations=iterations, on_hsv=on_hsv))

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

    def get_intermediate_results(self) -> List[Image]:
        """Get intermediate results from last processing run.

        Returns:
            List of Image objects after each processing step
        """
        return self.intermediate_results

    def get_pipeline_info(self) -> List[Dict[str, Any]]:
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
