import numpy as np
import pytest
from dgst.utils.loader import Image
from dgst.utils.features import FeatureExtractor, RansacLineParams

def test_feature_extractor_init():
    # Create a dummy boolean edge image
    data = np.zeros((100, 100), dtype=np.bool_)
    image = Image(data=data, rois=[])
    
    extractor = FeatureExtractor(image)
    assert extractor.image.data.dtype == np.bool_
    assert extractor.binary_image.shape == (100, 100)

def test_feature_extractor_invalid_input():
    # Test with non-boolean image
    data = np.zeros((100, 100), dtype=np.uint8)
    image = Image(data=data, rois=[])
    
    with pytest.raises(ValueError, match="Edge image must be of type bool"):
        FeatureExtractor(image)

def test_ransac_line_fitting_empty():
    # Test on empty image
    data = np.zeros((100, 100), dtype=np.bool_)
    image = Image(data=data, rois=[])
    extractor = FeatureExtractor(image)
    
    params = RansacLineParams(
        max_iterations=100,
        distance_threshold=2.0,
        min_inliers=10
    )
    
    # Should return None as there are no points
    result = extractor.ransac_line_fitting(params)
    assert result is None

def test_cloning_behavior():
    # Verify that original image is not modified when erasing features
    data = np.zeros((100, 100), dtype=np.bool_)
    # Add a line
    for i in range(100):
        data[i, i] = True
    
    image = Image(data=data, rois=[])
    extractor = FeatureExtractor(image)
    
    # Manually modify extractor's internal image
    extractor._edge_image.data[:] = False
    
    # Check that original image is intact
    assert np.sum(image.data) == 100
    assert np.sum(extractor.image.data) == 0
    assert np.sum(extractor._original_edge_image.data) == 100
