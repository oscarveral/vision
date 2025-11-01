
import numpy as np
import cv2

def scale_inter_area(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Escala una imagen usando el método de interpolación 'area'.

    Args:
        image (np.ndarray): Imagen de entrada.
        scale_factor (float): Factor de escala. Valores mayores a 1 aumentan el tamaño,
                              valores menores a 1 lo reducen.

    Returns:
        np.ndarray: Imagen escalada.
    """

    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return scaled_image