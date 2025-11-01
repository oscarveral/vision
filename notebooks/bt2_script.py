from dgst.utils.features import FeatureExtractor
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

# Cargamos la imagen
image = np.load('./borders_0.npy')

# Instanciamos el extractor de características con la imagen de bordes (binaria)
feature_extractor = FeatureExtractor(edge_image=image.astype(np.bool))

# Almacenamos los segmentos detectados
lines = []
segments = []

# Parámetros de iteración
min_inliers = 100
max_iterations = 500
min_segment_length = 50.0
iterations = 0

# Iteramos para encontrar más líneas
while iterations < max_iterations:
    line, segment = feature_extractor.ransac_segment_fitting(
        max_iterations=200,
        distance_threshold=0.7,
        density_threshold=0.2,
        min_inliers=min_inliers,
        max_lsq_iterations=0,
        min_segment_length=min_segment_length,
        erase=True
    )
    if (line is not None):
        lines.append(line)
    if (segment is not None):
        segments.append(segment)
    iterations += 1
    if (iterations % 10) == 0:
        min_inliers = min_inliers - 1  # Reducimos el umbral de inliers para encontrar líneas más débiles
        min_segment_length = max(10.0, min_segment_length - 5.0)  # Reducimos la longitud mínima del segmento

print(len(lines), "lines detected.")
print(len(segments), "segments detected.")
image = (image * 255).astype(np.uint8)
image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

# for line in lines:
#     a, b, c = line
#     if b != 0:
#         x0, y0 = 0, int(-c / b)
#         x1, y1 = image.shape[1], int((-a * image.shape[1] - c) / b)
#     else:
#         x0, y0 = int(-c / a), 0
#         x1, y1 = int(-c / a), image.shape[0]
#     cv.line(image, (x0, y0), (x1, y1), (255, 0, 0), 1)

for s in segments:
    (x_start, y_start), (x_end, y_end) = s
    cv.line(image, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 255, 0), 2)


# Plotteamos la imagen y la línea
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray')
plt.title('Detected Lines using RANSAC')
plt.axis('off')
plt.show()

