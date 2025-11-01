import cv2
import numpy as np

def median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Aplica un filtro de mediana a una imagen.

    Args:
        image (np.ndarray): Imagen de entrada.
        kernel_size (int): Tamaño del kernel. Debe ser un número impar mayor que 1.

    Returns:
        np.ndarray: Imagen filtrada.
    """

    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image