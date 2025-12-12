"""Tests for BT3a: Local descriptor extraction and comparison."""

import numpy as np
import pytest
import cv2
import os

from dgst.utils.loader import Image
from dgst.utils.processor import ImageProcessor
from dgst.utils.descriptors import (
    LocalDescriptorExtractor,
    DescriptorMethod,
    MatchStats,
    KeyPoint,
    ExtractionResult,
)


TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images/lenna.png")


@pytest.fixture
def lenna_image():
    """Load test image."""
    data = cv2.imread(TEST_IMAGE_PATH)
    if data is None:
        pytest.skip("lenna.png not found")
    return Image(data=data, rois=[])


@pytest.fixture
def extractor():
    """Create extractor instance."""
    return LocalDescriptorExtractor()


class TestLocalDescriptorExtractor:
    """Tests for LocalDescriptorExtractor class."""

    def test_extract_sift(self, extractor, lenna_image):
        """Test SIFT extraction returns keypoints and descriptors."""
        result = extractor.extract(lenna_image, DescriptorMethod.SIFT)

        assert result.num_keypoints > 0, "Should detect keypoints"
        assert result.has_descriptors(), "Should compute descriptors"
        assert result.descriptor_dimension == 128, "SIFT descriptor is 128-dimensional"
        assert not result.is_binary_descriptor, "SIFT uses float descriptors"
        assert result.method == DescriptorMethod.SIFT

    def test_extract_akaze(self, extractor, lenna_image):
        """Test AKAZE extraction returns keypoints and descriptors."""
        result = extractor.extract(lenna_image, DescriptorMethod.AKAZE)

        assert result.num_keypoints > 0, "Should detect keypoints"
        assert result.has_descriptors(), "Should compute descriptors"
        assert result.is_binary_descriptor, "AKAZE uses binary descriptors"

    def test_extract_orb(self, extractor, lenna_image):
        """Test ORB extraction returns keypoints and descriptors."""
        result = extractor.extract(lenna_image, DescriptorMethod.ORB)

        assert result.num_keypoints > 0, "Should detect keypoints"
        assert result.has_descriptors(), "Should compute descriptors"
        assert result.descriptor_dimension == 32, "ORB descriptor is 32 bytes"
        assert result.is_binary_descriptor, "ORB uses binary descriptors"

    def test_extract_from_grayscale(self, extractor, lenna_image):
        """Test extraction works on grayscale images."""
        lenna_image.to_grayscale()
        assert lenna_image.is_grayscale

        result = extractor.extract(lenna_image, DescriptorMethod.SIFT)
        assert result.num_keypoints > 0

    def test_extraction_result_properties(self, extractor, lenna_image):
        """Test ExtractionResult properties."""
        result = extractor.extract(lenna_image, DescriptorMethod.SIFT)

        assert result.image_width == lenna_image.data.shape[1]
        assert result.image_height == lenna_image.data.shape[0]
        assert result.extraction_time_ms > 0

    def test_keypoint_properties(self, extractor, lenna_image):
        """Test KeyPoint wrapper properties."""
        result = extractor.extract(lenna_image, DescriptorMethod.SIFT)
        
        assert len(result.keypoints) > 0
        kp = result.keypoints[0]
        
        assert isinstance(kp, KeyPoint)
        assert 0 <= kp.x <= lenna_image.data.shape[1]
        assert 0 <= kp.y <= lenna_image.data.shape[0]
        assert kp.size >= 0
        assert kp.position == (kp.x, kp.y)

    def test_extract_invalid_image(self, extractor):
        """Test extraction raises on invalid image."""
        invalid_image = Image(data=None, rois=[])

        with pytest.raises(ValueError, match="Image data is None"):
            extractor.extract(invalid_image, DescriptorMethod.SIFT)


class TestDescriptorComparison:
    """Tests for method comparison with ImageProcessor transforms."""

    def test_compare_rotation_invariance(self, extractor, lenna_image):
        """Test comparison under rotation using ImageProcessor."""
        transform = ImageProcessor().add_rotation(45)
        results = extractor.compare_methods(lenna_image, transform)

        # All methods should be tested
        assert DescriptorMethod.SIFT in results
        assert DescriptorMethod.AKAZE in results
        assert DescriptorMethod.ORB in results

        # All should find some matches
        for method, stats in results.items():
            assert isinstance(stats, MatchStats)
            assert stats.num_keypoints_original > 0
            assert stats.extraction_time_ms > 0

    def test_compare_scale_invariance(self, extractor, lenna_image):
        """Test comparison under scale change using ImageProcessor."""
        transform = ImageProcessor().add_scale_transform(0.75, restore_size=True)
        results = extractor.compare_methods(lenna_image, transform)

        # SIFT should have good repeatability under scale
        sift_stats = results[DescriptorMethod.SIFT]
        assert sift_stats.num_good_matches > 0, "SIFT should find matches under scale"

    def test_compare_noise_robustness(self, extractor, lenna_image):
        """Test comparison under noise using ImageProcessor."""
        transform = ImageProcessor().add_gaussian_noise(sigma=30)
        results = extractor.compare_methods(lenna_image, transform)

        # Should still find some matches despite noise
        for method, stats in results.items():
            assert stats.num_keypoints_original > 0

    def test_compare_combined_transforms(self, extractor, lenna_image):
        """Test comparison with multiple chained transforms."""
        transform = (
            ImageProcessor()
            .add_rotation(15)
            .add_gaussian_noise(sigma=10)
        )
        results = extractor.compare_methods(lenna_image, transform)

        for method, stats in results.items():
            assert stats.num_keypoints_original > 0
            assert stats.num_keypoints_transformed > 0


class TestVisualization:
    """Tests for visualization functions."""

    def test_visualize_keypoints(self, extractor, lenna_image):
        """Test keypoint visualization."""
        result = extractor.extract(lenna_image, DescriptorMethod.SIFT)
        vis_image = extractor.visualize_keypoints(lenna_image, result)

        assert vis_image.is_color, "Visualization should be color"
        assert vis_image.data.shape[:2] == lenna_image.data.shape[:2]
        assert len(vis_image.metadata.steps) > 0

    def test_visualize_comparison(self, extractor, lenna_image):
        """Test comparison visualization."""
        results = extractor.visualize_comparison(lenna_image)

        assert len(results) == 3, "Should visualize all 3 methods"
        for method, vis_image in results.items():
            assert vis_image.is_color
