from dgst.utils.features import FeatureExtractor
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

# Cargamos la imagen
image = np.load('./borders_0.npy')

# Instanciamos el extractor de características con la imagen de bordes (binaria)
feature_extractor = FeatureExtractor(edge_image=image.astype(np.bool))

# Almacenamos los circulos detectados
circles = []

# Parámetros de iteración
min_inliers_ratio = 3.0
max_iterations = 500
min_radius = 10.0
max_radius = 50.0
iterations = 0

# Iteramos para encontrar más líneas
while iterations < max_iterations:
    circle, _ = feature_extractor.ransac_circle_fitting(
        distance_threshold=1.0,
        min_inlier_ratio=min_inliers_ratio,
        max_iterations=100,
        min_radius=min_radius,
        max_radius=max_radius,
        erase=True
    )
    if circle is None:
        break
    circles.append(circle)

print(f"{len(circles)} circles detected using RANSAC.")

image = feature_extractor._edge_image.copy()
image = cv.cvtColor(image.astype(np.uint8) * 255, cv.COLOR_GRAY2RGB)

for circle in circles:
    cv.circle(image, 
              center=(int(circle[0]), int(circle[1])), 
              radius=int(circle[2]), 
              color=(0, 255, 0), 
              thickness=2)



# Plotteamos la imagen y la línea
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray')
plt.title('Detected Lines using RANSAC')
plt.axis('off')
plt.show()

