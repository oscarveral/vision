import numpy as np
import pytest
import cv2
import os
from dgst.utils.loader import Image, RegionOfInterest
from dgst.utils.processor import (
    ImageProcessor,
    GrayscaleStep,
    GaussianFilterStep,
    CannyEdgeDetectionStep,
    ProcessingTechnique
)
from dgst.utils.exceptions import ValidationError

TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images/lenna.png")

@pytest.fixture
def lenna_image():
    data = cv2.imread(TEST_IMAGE_PATH)
    if data is None:
        pytest.skip("lenna.png not found")
    return Image(data=data, rois=[])

def test_grayscale_step(lenna_image):
    step = GrayscaleStep()
    result = step.process(lenna_image)
    
    assert result.is_grayscale
    assert result.data.ndim == 2
    assert result.data.dtype == np.uint8
    assert "technique" in result.metadata.steps[-1]
    assert result.metadata.steps[-1]["technique"] == ProcessingTechnique.GRAYSCALE

def test_gaussian_filter_step_grayscale(lenna_image):
    # Convert to grayscale first
    lenna_image.to_grayscale()
    
    step = GaussianFilterStep(sigma=1.0)
    result = step.process(lenna_image)
    
    assert result.is_grayscale
    assert result.data.ndim == 2
    assert result.data.dtype == np.uint8

def test_gaussian_filter_step_color(lenna_image):
    # Test on BGR image (polymorphism)
    assert lenna_image.is_color
    
    step = GaussianFilterStep(sigma=1.0)
    result = step.process(lenna_image)
    
    assert result.is_color
    assert result.data.ndim == 3
    assert result.data.shape[2] == 3
    assert result.data.dtype == np.uint8

def test_canny_edge_detection_step_grayscale(lenna_image):
    lenna_image.to_grayscale()
    
    step = CannyEdgeDetectionStep(low_threshold=50, high_threshold=150)
    result = step.process(lenna_image)
    
    assert result.is_grayscale
    assert result.data.ndim == 2
    # Canny output is binary edges (0 or 255)
    assert np.all(np.isin(result.data, [0, 255]))

def test_canny_edge_detection_step_color(lenna_image):
    # Test on BGR image (polymorphism)
    assert lenna_image.is_color
    
    step = CannyEdgeDetectionStep(low_threshold=50, high_threshold=150)
    result = step.process(lenna_image)
    
    assert result.is_color
    assert result.data.ndim == 3
    assert result.data.shape[2] == 3
    # Each channel should be binary edges
    assert np.all(np.isin(result.data, [0, 255]))

def test_image_processor_chain(lenna_image):
    processor = ImageProcessor()
    processor.add_grayscale()
    processor.add_gaussian_filter(sigma=1.0)
    processor.add_canny_edge_detection(low_threshold=50, high_threshold=150)
    
    result = processor.process(lenna_image)
    
    assert result.is_grayscale
    assert len(processor.steps) == 3
    assert len(result.metadata.steps) == 3
    assert result.metadata.steps[0]["technique"] == ProcessingTechnique.GRAYSCALE
    assert result.metadata.steps[1]["technique"] == ProcessingTechnique.GAUSSIAN_FILTER
    assert result.metadata.steps[2]["technique"] == ProcessingTechnique.CANNY_EDGE_DETECTION

def test_image_processor_builder_pattern(lenna_image):
    processor = (
        ImageProcessor()
        .add_grayscale()
        .add_gaussian_filter(sigma=1.0)
    )
    
    result = processor.process(lenna_image)
    assert len(processor.steps) == 2
    assert result.is_grayscale
