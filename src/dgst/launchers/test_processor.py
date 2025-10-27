#!/usr/bin/env python3
"""Test script for ImageProcessor with box and Gaussian filters."""

import os
import sys
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dgst.utils.processor import ImageProcessor


def test_single_filters():
    """Test individual filters."""
    print("=" * 60)
    print("Testing Individual Filters")
    print("=" * 60)
    
    # Create a simple test image
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # Test box filter
    print("\n1. Testing Box Filter (size=5)...")
    processor = ImageProcessor()
    processor.add_box_filter(filter_size=5)
    result = processor.process(image, keep_intermediate=True)
    print(f"   Input shape: {image.shape}, Output shape: {result.shape}")
    print(f"   Pipeline: {processor.get_pipeline_info()}")
    
    # Test Gaussian filter
    print("\n2. Testing Gaussian Filter (sigma=1.5)...")
    processor = ImageProcessor()
    processor.add_gaussian_filter(sigma=1.5)
    result = processor.process(image, keep_intermediate=True)
    print(f"   Input shape: {image.shape}, Output shape: {result.shape}")
    print(f"   Pipeline: {processor.get_pipeline_info()}")


def test_filter_chain():
    """Test chaining multiple filters."""
    print("\n" + "=" * 60)
    print("Testing Filter Chain")
    print("=" * 60)
    
    # Create test image
    image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    
    # Chain multiple filters
    print("\nChaining: Box(3) -> Gaussian(1.0) -> Box(5) -> Gaussian(2.0)")
    processor = (ImageProcessor()
                 .add_box_filter(filter_size=3)
                 .add_gaussian_filter(sigma=1.0)
                 .add_box_filter(filter_size=5)
                 .add_gaussian_filter(sigma=2.0))
    
    result = processor.process(image, keep_intermediate=True)
    
    print(f"\nInput shape: {image.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Number of steps: {len(processor)}")
    print("\nPipeline details:")
    for i, step_info in enumerate(processor.get_pipeline_info(), 1):
        print(f"  Step {i}: {step_info}")
    
    # Check intermediate results
    intermediates = processor.get_intermediate_results()
    print(f"\nIntermediate results: {len(intermediates)} images")
    for i, img in enumerate(intermediates, 1):
        print(f"  After step {i}: shape={img.shape}, dtype={img.dtype}")


def test_with_real_image():
    """Test with a real image if available."""
    print("\n" + "=" * 60)
    print("Testing With Real Image")
    print("=" * 60)
    
    # Try to find test images
    test_image_paths = [
        "/home/oscar/Work/vision/src/bench/images/lenna.png",
        "/home/oscar/Work/vision/src/bench/images/denis.jpg",
        "/home/oscar/Work/vision/data/single_frames/000000/camera_front_blur"
    ]
    
    test_image = None
    image_path = None
    
    for path in test_image_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                # Get first image in directory
                files = os.listdir(path)
                if files:
                    image_path = os.path.join(path, files[0])
                    test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image_path = path
                test_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            if test_image is not None:
                break
    
    if test_image is None:
        print("\n⚠ No test image found. Skipping real image test.")
        return
    
    print(f"\n✓ Loaded test image from: {image_path}")
    print(f"  Image shape: {test_image.shape}")
    print(f"  Image dtype: {test_image.dtype}")
    
    # Apply processing pipeline
    print("\nApplying pipeline: Grayscale -> Box(7) -> Gaussian(2.5)")
    processor = (ImageProcessor()
                 .add_grayscale()
                 .add_box_filter(filter_size=7)
                 .add_gaussian_filter(sigma=2.5))
    
    result = processor.process(test_image, keep_intermediate=True)
    
    print(f"\n✓ Processing complete!")
    print(f"  Output shape: {result.shape}")
    print(f"  Output dtype: {result.dtype}")
    
    # Save results if possible
    output_dir = "/home/oscar/Work/vision/src/dgst/launchers/output"
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original
        cv2.imwrite(os.path.join(output_dir, "original.png"), test_image)
        
        # Save intermediate results
        intermediates = processor.get_intermediate_results()
        for i, img in enumerate(intermediates, 1):
            cv2.imwrite(os.path.join(output_dir, f"step_{i}.png"), img)
        
        # Save final result
        cv2.imwrite(os.path.join(output_dir, "final.png"), result)
        
        print(f"\n✓ Results saved to: {output_dir}")
        print(f"  - original.png")
        for i in range(len(intermediates)):
            print(f"  - step_{i+1}.png")
        print(f"  - final.png")
    except Exception as e:
        print(f"\n⚠ Could not save results: {e}")


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    processor = ImageProcessor()
    
    # Test invalid box filter size (even number)
    print("\n1. Testing invalid box filter size (even number)...")
    try:
        processor.add_box_filter(filter_size=4)
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")
    
    # Test invalid Gaussian sigma (negative)
    print("\n2. Testing invalid Gaussian sigma (negative)...")
    try:
        processor.add_gaussian_filter(sigma=-1.0)
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")
    
    # Test invalid Gaussian sigma (zero)
    print("\n3. Testing invalid Gaussian sigma (zero)...")
    try:
        processor.add_gaussian_filter(sigma=0.0)
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")


def test_color_to_grayscale():
    """Test automatic color to grayscale conversion."""
    print("\n" + "=" * 60)
    print("Testing Color to Grayscale Conversion")
    print("=" * 60)
    
    # Create color image
    color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    print(f"\nInput: Color image with shape {color_image.shape}")
    
    # Process with grayscale step first, then box filter
    processor = ImageProcessor().add_grayscale().add_box_filter(filter_size=5)
    result = processor.process(color_image)
    
    print(f"Output: Grayscale image with shape {result.shape}")
    print("✓ Grayscale conversion + box filter successful")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ImageProcessor Test Suite")
    print("=" * 60)
    
    try:
        test_single_filters()
        test_filter_chain()
        test_color_to_grayscale()
        test_error_handling()
        test_with_real_image()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
