import numpy as np
import cv2 as cv

# Eliminamos las componentes conexas pequeñas en las máscaras de señales para quitar ruido.
def filter_connected_components(mask, min_size=20):
    num_labels, labels_im = cv.connectedComponents(mask.astype(np.uint8))
    sizes = np.bincount(labels_im.flatten())
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    for j in range(1, num_labels):
        if sizes[j] >= min_size:
            filtered_mask[labels_im == j] = 255
    return filtered_mask