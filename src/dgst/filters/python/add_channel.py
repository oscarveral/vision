import cv2 as cv
import numpy as np

def add_channel_weight(ch1: np.ndarray,
                       ch2: np.ndarray,
                       weight: float) -> np.ndarray:
    """
    Adds two channels together with a specified weight for the second channel.

    Parameters:
    ch1 (np.ndarray): The first channel.
    ch2 (np.ndarray): The second channel.
    weight (float): The weight to apply to the second channel.

    Returns:
    np.ndarray: The resulting channel after addition.
    """
    hs_channel = cv.addWeighted(ch1, 1-weight, ch2, weight, 0)
    return hs_channel