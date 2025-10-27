"""Tests for the ffi Canny edge detection filter."""

import numpy as np
import pytest
import cv2
import os

from dgst.ffi.wrapper import canny_edge_detection, gaussian_filter

TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images/lenna.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# ---------------------------------------------------------------
# Tests for canny_edge_detection.
# ---------------------------------------------------------------

def test_canny_filter_wrong_dtype():
    img = np.ones((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        canny_edge_detection(img, low_threshold=50, high_threshold=150)


def test_canny_filter_wrong_shape():
    img = np.ones((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        canny_edge_detection(img, low_threshold=50, high_threshold=150)


def test_canny_filter_invalid_thresholds():
    """Test that high_threshold < low_threshold returns error."""
    img = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        canny_edge_detection(img, low_threshold=150, high_threshold=50)


def test_canny_filter_output_binary():
    """Test that output is binary (only 0 and 255)."""
    img = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "images/lenna.png not found"
    
    # Apply Canny (expects pre-smoothed input for best results)
    edges = canny_edge_detection(img, low_threshold=50, high_threshold=150)
    
    # Check output is binary
    unique_values = np.unique(edges)
    assert all(v in [0, 255] for v in unique_values), "Output should only contain 0 and 255"


def test_canny_filter_visual_output():
    """Test Canny filter with different threshold combinations."""
    img = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "images/lenna.png not found"
    
    # Pre-smooth for better results
    smoothed = gaussian_filter(img, sigma=1.4)
    
    # Test with different threshold combinations on smoothed image
    thresholds = [
        (30, 90),   # More edges (low thresholds)
        (50, 150),  # Medium (typical values)
        (100, 200), # Fewer edges (high thresholds)
    ]
    
    for low, high in thresholds:
        edges = canny_edge_detection(smoothed, low_threshold=low, high_threshold=high)
        
        # Check that some edges were detected
        edge_pixels = np.sum(edges == 255)
        total_pixels = edges.size
        edge_percentage = (edge_pixels / total_pixels) * 100
        assert edge_pixels > 0, f"Should detect at least some edges (L={low}, H={high})"


def test_canny_filter_opencv_comparison():
    """Compare with OpenCV Canny (for reference, not exact match expected)."""
    img = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "images/lenna.png not found"
    
    # Pre-smooth for fair comparison (OpenCV Canny does internal smoothing)
    smoothed = gaussian_filter(img, sigma=1.4)
    
    # OpenCV Canny on pre-smoothed image
    edges_cv = cv2.Canny(smoothed, threshold1=50, threshold2=150)
    
    # Our implementation on pre-smoothed image
    edges_custom = canny_edge_detection(smoothed, low_threshold=50, high_threshold=150)
    
    # Calculate similarity (IoU of edge pixels)
    intersection = np.sum((edges_cv == 255) & (edges_custom == 255))
    union = np.sum((edges_cv == 255) | (edges_custom == 255))
    iou = intersection / union if union > 0 else 0
    
    # We don't expect exact match due to implementation differences,
    # but there should be reasonable overlap
    assert iou > 0.3, f"Edge detection should have reasonable overlap with OpenCV (IoU={iou:.3f})"


def test_canny_filter_different_sigmas():
    """Test Canny with different Gaussian smoothing levels."""
    img = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "images/lenna.png not found"
    
    sigmas = [0.5, 1.0, 1.4, 2.0, 3.0]
    
    for sigma in sigmas:
        smoothed = gaussian_filter(img, sigma=sigma)
        edges = canny_edge_detection(smoothed, low_threshold=50, high_threshold=150)
        
        edge_pixels = np.sum(edges == 255)
        # More smoothing should generally result in fewer detected edges
        assert edge_pixels > 0, f"Should detect edges with sigma={sigma:.1f}"
