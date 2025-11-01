import numpy as np
import cv2


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Compute Otsu threshold on a 2D image and return uint8 mask (0/255).

    Args:
        image: 2D image, uint8 or float. If float, values are assumed in [0,1] or [0,255].

    Returns:
        mask: uint8 array with values 0 or 255.
    """
    if image is None:
        raise ValueError("image must not be None")
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("otsu_threshold expects a 2D grayscale image")

    # Convert floats to uint8 if necessary
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # assume normalized [0,1] or [0,255]; scale if max <= 1.0
        if arr.max() <= 1.0:
            img_u8 = (arr * 255.0).astype(np.uint8)
        else:
            img_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype == np.uint8:
        img_u8 = arr
    else:
        # best effort conversion
        img_u8 = np.clip(arr, 0, 255).astype(np.uint8)

    # Use OpenCV Otsu thresholding
    _, mask = cv2.threshold(
        img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return mask
