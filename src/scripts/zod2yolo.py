import os
import json
import shutil
import numpy as np
import random
from glob import glob
from tqdm import tqdm
from pathlib import Path

PATH_CARPETA_1 = Path(__file__).parent # Annotations
PATH_CARPETA_2 = Path(__file__).parent # Images
PATH_CARPETA_3 = Path(__file__).parent # Infos

if not os.path.exists(PATH_CARPETA_1) or not os.path.exists(PATH_CARPETA_2) or not os.path.exists(PATH_CARPETA_3):
    raise ValueError(f"La ruta {PATH_CARPETA_1} no existe.")

OUTPUT_DIR = "bt4"
TOTAL_IMAGENES = 2000  
TRAIN_RATIO = 0.8      
MIN_PIXELS = 15         # Tamaño mínimo de caja en píxeles. 
BACKGROUND_RATIO = 0.1  # % de imágenes vacías permitidas (sin señales válidas).

CLASS_MAPPING = {
    "TrafficSign": 0
}

def crear_estructura_directorios(base_dir):
    subdirs = ['train', 'val']
    types = ['images', 'labels', 'calibration', 'raw_annotations']
    for t in types:
        for s in subdirs:
            os.makedirs(os.path.join(base_dir, t, s), exist_ok=True)
            
def convertir_bbox_yolo(coords, img_w, img_h):
    points = np.array(coords)
    if len(points.shape) == 3: points = points[0]
    
    # Clampeo de puntos fuera de imagen
    points[:, 0] = np.clip(points[:, 0], 0, img_w)
    points[:, 1] = np.clip(points[:, 1], 0, img_h)
    
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    
    w_pixel = x_max - x_min
    h_pixel = y_max - y_min
    
    # Eliminación de cajas muy pequeñas
    if w_pixel < MIN_PIXELS or h_pixel < MIN_PIXELS:
        return None
    
    # Normalización
    x_c = (x_min + w_pixel / 2) / img_w
    y_c = (y_min + h_pixel / 2) / img_h
    w = w_pixel / img_w
    h = h_pixel / img_h
    
    return x_c, y_c, w, h

def encontrar_imagen(base_path, frame_id):
    # Buscar imagen en la estructura dada
    pattern = os.path.join(base_path, "images_blur_*", "single_frames", frame_id, "camera_front_blur", "*.jpg")
    candidates = glob(pattern)
    return candidates[0] if candidates else None

# --- PROCESO PRINCIPAL ---

print("Escaneando IDs de anotaciones disponibles...")
search_path = os.path.join(PATH_CARPETA_1, "annotations", "single_frames", "*")
all_folders = sorted(glob(search_path))
all_ids = [os.path.basename(f) for f in all_folders]

print(f"Encontrados {len(all_ids)} frames totales.")

# Selección Aleatoria (si hay suficientes)
if len(all_ids) > TOTAL_IMAGENES:
    ids_seleccionados = random.sample(all_ids, TOTAL_IMAGENES)
else:
    ids_seleccionados = all_ids

crear_estructura_directorios(OUTPUT_DIR)

count_train = 0
count_val = 0
skipped_no_image = 0
skipped_no_objects = 0 # No tienen señales o solo tienen señales muy pequeñas
skipped_small_boxes = 0 # Contador de cajas individuales ignoradas

print("Procesando...")
for frame_id in tqdm(ids_seleccionados):
    
    json_path = os.path.join(PATH_CARPETA_1, "annotations", "single_frames", frame_id, "annotations", "object_detection.json")
    
    img_src_path = encontrar_imagen(PATH_CARPETA_2, frame_id)
    calib_path = os.path.join(PATH_CARPETA_3, "infos", "single_frames", frame_id, "calibration.json")
    
    if not img_src_path:
        skipped_no_image += 1
        continue
    if not os.path.exists(json_path):
        continue

    # Procesar Detecciones
    try:
        with open(json_path, 'r') as f:
            detections = json.load(f)
            
        yolo_lines = []
        IMG_W, IMG_H = 3848, 2168 
        
        for det in detections:
            if det['properties']['class'] in CLASS_MAPPING:
                cls_id = CLASS_MAPPING[det['properties']['class']]
                coords = det['geometry']['coordinates']
                
                # Convertir y filtrar
                res = convertir_bbox_yolo(coords, IMG_W, IMG_H)
                
                if res is not None:
                    xc, yc, w, h = res
                    yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                else:
                    skipped_small_boxes += 1
        
        # Guardar
        tiene_senales = len(yolo_lines) > 0
        es_background_elegido = (not tiene_senales) and (random.random() < BACKGROUND_RATIO)
        
        if tiene_senales or es_background_elegido:
            # Decidir split Train/Test
            split = "train" if random.random() < TRAIN_RATIO else "val"
            if split == "train": count_train += 1
            else: count_val += 1
            
            base_name = str(frame_id)
            
            # Guardar TXT
            dst_txt = os.path.join(OUTPUT_DIR, "labels", split, base_name + ".txt")
            with open(dst_txt, 'w') as f:
                f.write("\n".join(yolo_lines))
                
            # Copiar Imagen
            dst_img = os.path.join(OUTPUT_DIR, "images", split, base_name + ".jpg")
            shutil.copy2(img_src_path, dst_img)
            
            # Copiar Extras
            if os.path.exists(calib_path):
                shutil.copy2(calib_path, os.path.join(OUTPUT_DIR, "calibration", split, base_name + "_calib.json"))
            shutil.copy2(json_path, os.path.join(OUTPUT_DIR, "raw_annotations", split, base_name + "_det.json"))
            
        else:
            skipped_no_objects += 1

    except Exception as e:
        print(f"Error en {frame_id}: {e}")

print(f"\n--- Resumen ---")
print(f"Train: {count_train}")
print(f"Val: {count_val}")
print(f"Imágenes descartadas (Background excedente o sin objetos): {skipped_no_objects}")
print(f"Señales individuales ignoradas por ser < {MIN_PIXELS}px: {skipped_small_boxes}")
print(f"Dataset generado en: {os.path.abspath(OUTPUT_DIR)}")