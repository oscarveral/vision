import cv2 as cv
import numpy as np

def filtro_rojo_azul(img: np.ndarray) -> np.ndarray:
    
    # Red color.
    lower_red1 = (0, 90, 50)
    upper_red1 = (10, 255, 210)
    lower_red2 = (165, 90, 50)
    upper_red2 = (179, 255, 210)

    # Blue color.
    lower_blue = (90, 90, 50)
    upper_blue = (130, 255, 210)
    
    mask_red1 = cv.inRange(img, lower_red1, upper_red1)
    mask_red2 = cv.inRange(img, lower_red2, upper_red2)
    mask_blue = cv.inRange(img, lower_blue, upper_blue)
    
    mask_combined = mask_red1 | mask_red2 | mask_blue

    return mask_combined * 255
