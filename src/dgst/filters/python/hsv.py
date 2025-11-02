import cv2
import numpy as np

def into_hsv_channels(image: np.ndarray) -> np.ndarray:
    """Convert the image data into HSV channels and store them.

    Returns:
        np.ndarray: The HSV channels of the image.
    """
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image


def fuse_hsv_channels(channels: list) -> np.ndarray:
    """Fuse HSV channels into a single HSV image.

    Args:
        channels: List of 3 numpy arrays representing H, S, V channels

    Returns:
        np.ndarray: Fused HSV image with shape (H, W, 3)
    """
    if len(channels) != 3:
        raise ValueError(f"Expected 3 channels (H, S, V), got {len(channels)}")
    
    # Validate all channels have the same shape
    shape = channels[0].shape
    for idx, ch in enumerate(channels):
        if ch.shape != shape:
            raise ValueError(
                f"All channels must have the same shape. "
                f"Channel 0 has shape {shape}, but channel {idx} has shape {ch.shape}"
            )
        if ch.ndim != 2:
            raise ValueError(
                f"All channels must be 2D arrays. Channel {idx} has {ch.ndim} dimensions"
            )
    
    # Stack the channels into an HSV image
    hsv_image = np.stack(channels, axis=2).astype(np.uint8)
    return hsv_image