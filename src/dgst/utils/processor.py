import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from dgst.filters.ffi import box_filter, gaussian_filter, canny_edge_detection, kannala_brandt_undistort, kannala_brandt_map_points_to_undistorted, phase_congruency as pc_ffi, threshold_filter
from dgst.filters.python import phase_congruency as pc_python, otsu_threshold, clahe_filter, dilate_edges, scale_inter_area, median_blur, into_hsv_channels, add_channel_weight
from dgst.utils.loader import Image, RegionOfInterest, ImageFormat

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

        if image.format != ImageFormat.BGR:
            raise ValueError("GrayscaleStep expects a BGR image")

        if len(image.data.shape) == 3:
            image.data = cv2.cvtColor(image.data, cv2.COLOR_BGR2GRAY)
            image.format = ImageFormat.GRAYSCALE
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

        if image.format != ImageFormat.GRAYSCALE:
            raise ValueError("BoxFilterStep expects a grayscale image")

        image.data = box_filter(image.data, self.filter_size)
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
            if image.hsv_channels is None:
                raise ValueError("Image does not contain HSV channels for Gaussian filter on HSV")  
            hsv_filtered = []
            for channel in image.hsv_channels:
                filtered = gaussian_filter(channel, self.sigma)
                hsv_filtered.append(filtered)
            image.hsv_channels = hsv_filtered
        else:
            if image.format != ImageFormat.GRAYSCALE:
                raise ValueError("GaussianFilterStep expects a grayscale image")

            image.data = gaussian_filter(image.data, self.sigma)
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
            if image.hsv_channels is None:
                raise ValueError("Image does not contain HSV channels for Canny edge detection on HSV")  
            hsv_edges = []
            for channel in image.hsv_channels:
                edges = canny_edge_detection(channel, self.low_threshold, self.high_threshold)
                hsv_edges.append(edges)
            image.hsv_channels = hsv_edges

        else:
            if image.format != ImageFormat.GRAYSCALE:
                raise ValueError("CannyEdgeDetectionStep expects a grayscale image")
            image.data = canny_edge_detection(image.data, self.low_threshold, self.high_threshold)

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
        if image.calibration is None:
            raise ValueError("Image does not contain calibration data")
        if image.calibration.camera_type != "kannala":
            raise ValueError(f"Expected 'kannala' camera type, got '{image.calibration.camera_type}'")
        

        if image.format != ImageFormat.BGR:
            raise ValueError("KannalaBrandtUndistortionStep expects a BGR image")

        # Extract intrinsic parameters (3x3 matrix from 3x4)
        K = image.calibration.intrinsics[:3, :3]

        # Extract distortion coefficients (first 4)
        D = np.array(image.calibration.distortion[:4], dtype=np.float32)

        # Apply the undistortion using C implementation
        image.data = kannala_brandt_undistort(image.data, K, D)

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

        return image

    def get_params(self) -> Dict[str, Any]:
        params = {
            "technique": ProcessingTechnique.KANNALA_BRANDT_UNDISTORTION,
        }
        if self._calibration is not None:
            params["camera_type"] = self._calibration.camera_type
            params["distortion_coeffs"] = self._calibration.distortion
        return params


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

        if image.format != ImageFormat.GRAYSCALE:
            raise ValueError("PhaseCongruencyStep expects a grayscale image")

        func = pc_ffi if self.use_own else pc_python

        image.data = func(
            image.data,
            nscale=self.nscale,
            norient=self.norient,
            min_wavelength=self.min_wavelength,
            mult=self.mult,
            sigma_onf=self.sigma_onf,
            eps=self.eps,
        )

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
        # Ensure 2D
        if image.format != ImageFormat.GRAYSCALE:
            raise ValueError("ThresholdFilterStep expects a grayscale image")

        # Enforce strict 2D uint8 input as required by the C-backed threshold filter.
        if image.data.dtype != np.uint8:
            raise ValueError(
                "ThresholdFilterStep expects image.data to be dtype=uint8"
            )

        # Call the C-backed threshold_filter which expects uint8 input and returns uint8 0/255
        result = threshold_filter(image.data, self.threshold)

        # Store result (uint8) back into image.data
        image.data = result
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
        if image.data is None:
            raise ValueError("Image.data is None")
        
        if image.format != ImageFormat.BGR and image.format != ImageFormat.GRAYSCALE:
            raise ValueError("CLAHEStep expects a BGR or GRAYSCALE image")

        image.data = clahe_filter(image.data, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
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
        if image.data is None:
            raise ValueError("Image.data is None")
        if image.format != ImageFormat.GRAYSCALE:
            raise ValueError("OtsuThresholdStep expects a grayscale image")

        image.data = otsu_threshold(image.data)
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
            if image.hsv_channels is None:
                raise ValueError("Image does not contain HSV channels for DilateEdgesStep on HSV")  
            hsv_dilated = []
            for channel in image.hsv_channels:
                dilated = dilate_edges(channel, self.kernel_size, self.iterations)
                hsv_dilated.append(dilated)
            image.hsv_channels = hsv_dilated
        else:
            if image.format != ImageFormat.GRAYSCALE:
                raise ValueError("DilateEdgesStep expects a grayscale image")

            image.data = dilate_edges(image.data, self.kernel_size, self.iterations)

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
        if image.data is None:
            raise ValueError("Image.data is None")

        image.data = scale_inter_area(image.data, self.scale_factor)
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
        if image.data is None:
            raise ValueError("Image.data is None")

        image.data = median_blur(image.data, self.kernel_size)
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
        if image.data is None:
            raise ValueError("Image.data is None")
        if image.format != ImageFormat.BGR:
            raise ValueError("IntoHSVChannelsStep expects a BGR image")

        hsv_channels = into_hsv_channels(image.data)
        image.hsv_channels = [hsv_channels[:, :, 0], hsv_channels[:, :, 1], hsv_channels[:, :, 2]]
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
        if image.hsv_channels is None:
            raise ValueError("Image does not contain enough HSV channels for CombineChannelsStep")

        ch1_idx = {'H': 0, 'S': 1, 'V': 2}[self.channel1]
        ch2_idx = {'H': 0, 'S': 1, 'V': 2}[self.channel2]

        ch1 = image.hsv_channels[ch1_idx]
        ch2 = image.hsv_channels[ch2_idx]

        combined_channel = add_channel_weight(ch1, ch2, self.weight)
        image.data = combined_channel
        image.format = ImageFormat.GRAYSCALE  
        return image

    def get_params(self) -> Dict[str, Any]:
        return {
            "technique": "combine_channels",
            "weight": self.weight
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
            to_uint8: If True, output will be uint8 scaled to 0-255

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
