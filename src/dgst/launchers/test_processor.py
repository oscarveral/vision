#!/usr/bin/env python3
"""Test script for ImageProcessor with box and Gaussian filters."""

import os
import sys
import cv2
import numpy as np

from dgst.utils.loader import DataLoader
from dgst import DATA_ROOT

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dgst.utils.processor import ImageProcessor

def main():
   
    loader = DataLoader(DATA_ROOT + "/single_frames")
    image = loader.load(31)

    processor = (ImageProcessor()
        .add_kannala_brandt_undistortion(image.calibration)
    )

    image = processor.process_image(image)

    image.data = cv2.resize(image.data, (0,0), fx=0.3, fy=0.3)


    cv2.imshow("Original Image", image.data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
