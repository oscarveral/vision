import numpy as np
import cv2


def clahe_filter(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Works on grayscale (2D) or color BGR images (3-channel). For color
    images the algorithm is applied on the L channel in LAB colorspace and
    the result is converted back to BGR to preserve color balance.

    Args:
        image: Input image as numpy array (uint8 or float). If float, values
            are assumed to be in [0,1] or [0,255] and will be scaled/clipped
            to uint8.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization (width, height).

    Returns:
        result: uint8 image with same number of channels as input.
    """
    if image is None:
        raise ValueError("image must not be None")

    arr = np.asarray(image)

    # Prepare uint8 image
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        if arr.max() <= 1.0:
            img_u8 = (arr * 255.0).astype(np.uint8)
        else:
            img_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype == np.uint8:
        img_u8 = arr
    else:
        img_u8 = np.clip(arr, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])))

    # Grayscale
    if img_u8.ndim == 2:
        return clahe.apply(img_u8)

    # Color image: convert to LAB, apply to L channel
    if img_u8.ndim == 3 and img_u8.shape[2] == 3:
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        return result

    # Unsupported number of channels: attempt per-channel CLAHE
    if img_u8.ndim == 3:
        channels = []
        for c in range(img_u8.shape[2]):
            channels.append(clahe.apply(img_u8[:, :, c]))
        return np.stack(channels, axis=2)

    raise ValueError("Unsupported image shape for clahe_filter")
