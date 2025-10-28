import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from dgst.ffi.wrapper import box_filter, gaussian_filter, canny_edge_detection, kannala_brandt_undistort
from dgst.utils.loader import Image, Calibration

class ProcessingTechnique:
    BOX_FILTER = "box_filter"
    GAUSSIAN_FILTER = "gaussian_filter"
    CANNY_EDGE_DETECTION = "canny_edge_detection"
    GRAYSCALE = "grayscale"
    KANNALA_BRANDT_UNDISTORTION = "kannala_brandt_undistortion"


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
        if len(image.data.shape) == 3:
            image.data = cv2.cvtColor(image.data, cv2.COLOR_BGR2GRAY)
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
        image.data = box_filter(image.data, self.filter_size)
        return image

    def get_params(self) -> Dict[str, Any]:
        return {
            "technique": ProcessingTechnique.BOX_FILTER,
            "filter_size": self.filter_size
        }


class GaussianFilterStep(ProcessingStep):
    """Apply Gaussian filter using custom C implementation."""
    
    def __init__(self, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = sigma

    def process(self, image: Image) -> Image:
        image.data = gaussian_filter(image.data, self.sigma)
        return image

    def get_params(self) -> Dict[str, Any]:
        return {
            "technique": ProcessingTechnique.GAUSSIAN_FILTER,
            "sigma": self.sigma
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
        image.data = canny_edge_detection(image.data, self.low_threshold, self.high_threshold)
        return image

    def get_params(self) -> Dict[str, Any]:
        return {
            "technique": ProcessingTechnique.CANNY_EDGE_DETECTION,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold
        }


class KannalaBrandtUndistortionStep(ProcessingStep):
    """Apply Kannala-Brandt undistortion using calibration data from the image."""
    
    def __init__(self):
        self._calibration = None
    
    def process(self, image: Image) -> Image:
        if image.calibration is None:
            raise ValueError("Image does not contain calibration data")
        if image.calibration.camera_type != "kannala":
            raise ValueError(f"Expected 'kannala' camera type, got '{image.calibration.camera_type}'")
        
        # Extract intrinsic parameters (3x3 matrix from 3x4)
        K = image.calibration.intrinsics[:3, :3]
        
        # Extract distortion coefficients (first 4)
        D = np.array(image.calibration.distortion[:4], dtype=np.float32)
        
        # Apply the undistortion using C implementation
        image.data = kannala_brandt_undistort(image.data, K, D)
        
        self._calibration = image.calibration
        return image
    
    def get_params(self) -> Dict[str, Any]:
        params = {
            "technique": ProcessingTechnique.KANNALA_BRANDT_UNDISTORTION,
        }
        if self._calibration is not None:
            params["camera_type"] = self._calibration.camera_type
            params["distortion_coeffs"] = self._calibration.distortion
        return params


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
    
    def add_step(self, step: ProcessingStep) -> 'ImageProcessor':
        """Add a processing step to the pipeline.
        
        Args:
            step: ProcessingStep instance
            
        Returns:
            Self for method chaining
        """
        self.steps.append(step)
        return self
    
    def add_grayscale(self) -> 'ImageProcessor':
        """Add grayscale conversion step.
        
        Returns:
            Self for method chaining
        """
        return self.add_step(GrayscaleStep())
    
    def add_box_filter(self, filter_size: int = 5) -> 'ImageProcessor':
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
        self.original_image = Image(
            data=image.data.copy(),
            rois=image.rois,
            calibration=image.calibration
        )
        self.intermediate_results = []
        
        # Create a working copy with all metadata preserved
        current = Image(
            data=image.data.copy(),
            rois=image.rois,
            calibration=image.calibration
        )
        
        for step in self.steps:
            current = step.process(current)
            if keep_intermediate:
                self.intermediate_results.append(Image(
                    data=current.data.copy(),
                    rois=current.rois,
                    calibration=current.calibration
                ))
        
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
    
    def clear(self) -> 'ImageProcessor':
        """Clear all processing steps."""
        self.steps = []
        return self
    
    def __len__(self) -> int:
        """Return number of steps in pipeline."""
        return len(self.steps)
    
    def __repr__(self) -> str:
        """String representation of the processor."""
        return f"ImageProcessor(steps={len(self.steps)})"
