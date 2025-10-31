from dgst.utils.features import FeatureExtractor
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

extractor = FeatureExtractor()

# Cargamos la imagen
image = np.load('./borders_0.npy')

# Extraemos una línea usando RANSAC
line = extractor.ransac_line_fitting(
    edge_image=image.astype(bool), 
    max_iterations=1000, 
    distance_threshold=2.0, 
    min_inliers=30, 
    max_lsq_iterations=10,
    erase=True
)

# Visualizamos la línea sobre la imagen original (solo los puntos que quedan dentro de la imagen)
a, b, c = line
height, width = image.shape

if (b < 1e-6 and b > -1e-6):
    # Hay linea vertical
    y_vals = np.array([0, height - 1])
    x_vals = np.array([-c / a, -c / a])
elif (a < 1e-6 and a > -1e-6):
    # Hay linea horizontal
    x_vals = np.array([0, width - 1])
    y_vals = np.array([-c / b, -c / b])
else:
    # Calculamos dos puntos en los extremos de la imagen
    x_vals = np.array([0, width - 1])
    y_vals = (-a * x_vals - c) / b
    if (y_vals[0] < 0):
        y_vals[0] = 0
        x_vals[0] = (-b * y_vals[0] - c) / a
    elif (y_vals[0] >= height):
        y_vals[0] = height - 1
        x_vals[0] = (-b * y_vals[0] - c) / a
    if (y_vals[1] < 0):
        y_vals[1] = 0
        x_vals[1] = (-b * y_vals[1] - c) / a
    elif (y_vals[1] >= height):
        y_vals[1] = height - 1
        x_vals[1] = (-b * y_vals[1] - c) / a

# Plotteamos la imagen y la línea
plt.imshow(image, cmap='gray')
plt.plot(x_vals, y_vals, color='red')
plt.xlim(0, width - 1)
plt.ylim(height - 1, 0)
plt.title('RANSAC Line Fitting')
plt.show()

# Mostramos la imagen resultante tras borrar los inliers

edge_image = extractor.remove_line(
    edge_image=image.astype(bool),
    line=line,
    distance_threshold=2.0
)
edge_image = edge_image.astype(np.uint8) * 255

plt.imshow(edge_image, cmap='gray')
plt.title('Edge Image after Inlier Removal')
plt.xlim(0, width - 1)
plt.ylim(height - 1, 0)
plt.show()

