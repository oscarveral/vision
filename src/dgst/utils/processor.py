import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from dgst.ffi.wrapper import box_filter, gaussian_filter, canny_edge_detection

class ProcessingTechnique:
    BOX_FILTER = "box_filter"
    GAUSSIAN_FILTER = "gaussian_filter"
    CANNY_EDGE_DETECTION = "canny_edge_detection"
    GRAYSCALE = "grayscale"


class ProcessingStep(ABC):
    """Abstract base class for processing steps."""
    
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
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
    
    def process(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def get_params(self) -> Dict[str, Any]:
        return {"technique": ProcessingTechnique.GRAYSCALE}


class BoxFilterStep(ProcessingStep):
    """Apply box filter using custom C implementation."""
    
    def __init__(self, filter_size: int = 5):
        if filter_size % 2 == 0:
            raise ValueError("filter_size must be odd")
        self.filter_size = filter_size
    
    def process(self, image: np.ndarray) -> np.ndarray:
        return box_filter(image, self.filter_size)
    
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
    
    def process(self, image: np.ndarray) -> np.ndarray:
        return gaussian_filter(image, self.sigma)
    
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
    
    def process(self, image: np.ndarray) -> np.ndarray:
        return canny_edge_detection(image, self.low_threshold, self.high_threshold)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "technique": ProcessingTechnique.CANNY_EDGE_DETECTION,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold
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
        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.intermediate_results: List[np.ndarray] = []
    
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
    
    def process(self, image: np.ndarray, 
                keep_intermediate: bool = False) -> np.ndarray:
        """Process image through all steps in the pipeline.
        
        Args:
            image: Input image
            keep_intermediate: Store intermediate results for debugging
            
        Returns:
            Final processed image
        """
        self.original_image = image.copy()
        self.intermediate_results = []
        
        current = image.copy()
        
        for step in self.steps:
            current = step.process(current)
            if keep_intermediate:
                self.intermediate_results.append(current.copy())
        
        self.processed_image = current
        return current
    
    def get_intermediate_results(self) -> List[np.ndarray]:
        """Get intermediate results from last processing run.
        
        Returns:
            List of images after each processing step
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
