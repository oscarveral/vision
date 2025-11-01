import cv2
import numpy as np

def into_hsv_channels(image: np.ndarray) -> np.ndarray:
    """Convert the image data into HSV channels and store them.

    Returns:
        np.ndarray: The HSV channels of the image.
    """
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image