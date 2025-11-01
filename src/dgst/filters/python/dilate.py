import numpy as np
import scipy.ndimage as ndi

# Función para hacer más gruesos (visibles) los bordes detectados.
def dilate_edges(edge_image, kernel_size=3, iterations=1):
    # Usamos binary dilation para engrosar los puntos blancos (bordes) con binary dilation.
    dilated = np.array(ndi.binary_dilation(edge_image, structure=np.ones((kernel_size, kernel_size)), iterations=iterations))*255
    return dilated
