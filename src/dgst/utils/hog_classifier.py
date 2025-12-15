"""
HOG Traffic Sign Classifier - BT3 Implementation.

This module implements a HOG-based classifier for traffic sign detection
with an abstraction layer to support multiple HOG implementations.

Uses lazy loading for memory efficiency - images are loaded on-demand
and can be unloaded when not needed.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator
import hashlib
import json
import os
import pickle
import random

import cv2
import numpy as np
from skimage.feature import hog as skimage_hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dgst.utils.loader import Image, RegionOfInterest, DataLoader


class HOGMethod(Enum):
    """Métodos de extracción HOG disponibles."""
    SKIMAGE = "skimage"
    CUSTOM = "custom"  # Para implementación propia futura


class HOGExtractor(ABC):
    """Interfaz abstracta para extractores HOG."""

    @abstractmethod
    def compute(self, patch: np.ndarray) -> np.ndarray:
        """Calcula descriptor HOG para un parche en escala de grises.
        
        Args:
            patch: Imagen en escala de grises (H, W) normalizada al tamaño esperado.
            
        Returns:
            Vector de características HOG.
        """
        pass

    @property
    @abstractmethod
    def feature_length(self) -> int:
        """Longitud del vector de características HOG."""
        pass

    @property
    @abstractmethod
    def patch_size(self) -> tuple[int, int]:
        """Tamaño esperado del parche (height, width)."""
        pass


class SkimageHOGExtractor(HOGExtractor):
    """Implementación HOG usando skimage.feature.hog."""

    def __init__(
        self,
        patch_size: tuple[int, int] = (64, 64),
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
    ):
        """Inicializa el extractor HOG de skimage.
        
        Args:
            patch_size: Tamaño del parche (height, width).
            orientations: Número de bins de orientación.
            pixels_per_cell: Tamaño de celda en píxeles.
            cells_per_block: Tamaño de bloque en celdas.
        """
        self._patch_size = patch_size
        self._orientations = orientations
        self._pixels_per_cell = pixels_per_cell
        self._cells_per_block = cells_per_block
        
        # Calcular longitud del feature vector
        h, w = patch_size
        cells_y = h // pixels_per_cell[0]
        cells_x = w // pixels_per_cell[1]
        blocks_y = cells_y - cells_per_block[0] + 1
        blocks_x = cells_x - cells_per_block[1] + 1
        self._feature_length = (
            blocks_y * blocks_x * 
            cells_per_block[0] * cells_per_block[1] * 
            orientations
        )

    def compute(self, patch: np.ndarray) -> np.ndarray:
        """Calcula descriptor HOG para un parche."""
        if patch.shape[:2] != self._patch_size:
            patch = cv2.resize(patch, (self._patch_size[1], self._patch_size[0]))
        
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            
        features = skimage_hog(
            patch,
            orientations=self._orientations,
            pixels_per_cell=self._pixels_per_cell,
            cells_per_block=self._cells_per_block,
            block_norm='L2-Hys',
            feature_vector=True,
        )
        return features

    @property
    def feature_length(self) -> int:
        return self._feature_length

    @property
    def patch_size(self) -> tuple[int, int]:
        return self._patch_size


# Implementación HOG propia (BT3kp)
class CustomHOGExtractor(HOGExtractor):
    """Implementación HOG propia desde cero (BT3kp = 10%).
    
    Implementa el algoritmo HOG siguiendo los pasos:
    1. Calcular gradientes (Sobel)
    2. Calcular magnitud y orientación
    3. Crear histogramas por celda
    4. Normalizar por bloques
    5. Concatenar en vector final
    """

    def __init__(
        self,
        patch_size: tuple[int, int] = (64, 64),
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
    ):
        """Inicializa el extractor HOG propio.
        
        Args:
            patch_size: Tamaño del parche (height, width).
            orientations: Número de bins de orientación (0-180°).
            pixels_per_cell: Tamaño de celda en píxeles.
            cells_per_block: Tamaño de bloque en celdas.
        """
        self._patch_size = patch_size
        self._orientations = orientations
        self._pixels_per_cell = pixels_per_cell
        self._cells_per_block = cells_per_block
        
        # Calcular longitud del feature vector
        h, w = patch_size
        self._cells_y = h // pixels_per_cell[0]
        self._cells_x = w // pixels_per_cell[1]
        self._blocks_y = self._cells_y - cells_per_block[0] + 1
        self._blocks_x = self._cells_x - cells_per_block[1] + 1
        self._feature_length = (
            self._blocks_y * self._blocks_x * 
            cells_per_block[0] * cells_per_block[1] * 
            orientations
        )

    def _compute_gradients(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calcula gradientes usando Sobel."""
        # Gradientes en X e Y
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
        
        # Magnitud y orientación
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi)  # En grados
        
        # Convertir a rango [0, 180) (unsigned gradients)
        orientation = orientation % 180
        
        return magnitude, orientation

    def _compute_cell_histogram(
        self,
        magnitude: np.ndarray,
        orientation: np.ndarray,
    ) -> np.ndarray:
        """Calcula histograma de orientaciones para una celda."""
        histogram = np.zeros(self._orientations)
        bin_width = 180.0 / self._orientations
        
        for i in range(magnitude.shape[0]):
            for j in range(magnitude.shape[1]):
                mag = magnitude[i, j]
                ori = orientation[i, j]
                
                # Bilinear interpolation entre bins
                bin_idx = ori / bin_width
                lower_bin = int(bin_idx) % self._orientations
                upper_bin = (lower_bin + 1) % self._orientations
                
                # Peso para interpolación
                upper_weight = bin_idx - int(bin_idx)
                lower_weight = 1.0 - upper_weight
                
                histogram[lower_bin] += mag * lower_weight
                histogram[upper_bin] += mag * upper_weight
        
        return histogram

    def _compute_cell_histograms(
        self,
        magnitude: np.ndarray,
        orientation: np.ndarray,
    ) -> np.ndarray:
        """Calcula histogramas para todas las celdas."""
        cell_h, cell_w = self._pixels_per_cell
        histograms = np.zeros((self._cells_y, self._cells_x, self._orientations))
        
        for cy in range(self._cells_y):
            for cx in range(self._cells_x):
                y_start = cy * cell_h
                y_end = y_start + cell_h
                x_start = cx * cell_w
                x_end = x_start + cell_w
                
                cell_mag = magnitude[y_start:y_end, x_start:x_end]
                cell_ori = orientation[y_start:y_end, x_start:x_end]
                
                histograms[cy, cx] = self._compute_cell_histogram(cell_mag, cell_ori)
        
        return histograms

    def _normalize_blocks(self, cell_histograms: np.ndarray) -> np.ndarray:
        """Normaliza histogramas por bloques (L2-Hys norm)."""
        block_h, block_w = self._cells_per_block
        features = []
        eps = 1e-5
        
        for by in range(self._blocks_y):
            for bx in range(self._blocks_x):
                # Extraer bloque de celdas
                block = cell_histograms[
                    by:by + block_h,
                    bx:bx + block_w
                ].flatten()
                
                # L2-Hys normalization
                # 1. L2 normalize
                norm = np.sqrt(np.sum(block**2) + eps)
                block = block / norm
                
                # 2. Clip values
                block = np.clip(block, 0, 0.2)
                
                # 3. L2 normalize again
                norm = np.sqrt(np.sum(block**2) + eps)
                block = block / norm
                
                features.extend(block)
        
        return np.array(features)

    def compute(self, patch: np.ndarray) -> np.ndarray:
        """Calcula descriptor HOG para un parche."""
        # Redimensionar si es necesario
        if patch.shape[:2] != self._patch_size:
            patch = cv2.resize(patch, (self._patch_size[1], self._patch_size[0]))
        
        # Convertir a escala de grises
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # Asegurar float64 para cálculos precisos
        patch = patch.astype(np.float64)
        
        # 1. Calcular gradientes
        magnitude, orientation = self._compute_gradients(patch)
        
        # 2. Calcular histogramas por celda
        cell_histograms = self._compute_cell_histograms(magnitude, orientation)
        
        # 3. Normalizar por bloques
        features = self._normalize_blocks(cell_histograms)
        
        return features

    @property
    def feature_length(self) -> int:
        return self._feature_length

    @property
    def patch_size(self) -> tuple[int, int]:
        return self._patch_size



@dataclass
class ImageReference:
    """Referencia ligera a una imagen sin cargar los datos.
    
    Permite lazy loading: solo carga la imagen cuando se necesita.
    """
    image_id: int  # Número de imagen en formato dgst (ej: 14 para 000014)
    data_path: Path  # Ruta base del directorio de datos
    rois: list[RegionOfInterest] = field(default_factory=list)
    
    def load(self) -> Image:
        """Carga la imagen completa desde disco."""
        loader = DataLoader(str(self.data_path))
        return loader.load(self.image_id)
    
    def load_metadata_only(self) -> list[RegionOfInterest]:
        """Carga solo los metadatos (ROIs) sin cargar la imagen."""
        if self.rois:
            return self.rois
        loader = DataLoader(str(self.data_path))
        self.rois = loader.load_metadata(self.image_id)
        return self.rois


@dataclass
class PatchSample:
    """Muestra de parche con su etiqueta."""
    patch: np.ndarray
    label: int  # 1 = señal, 0 = no señal
    bbox: tuple[int, int, int, int]  # (x, y, w, h) en imagen original
    source_image_idx: int


@dataclass
class ClassificationResult:
    """Resultado de clasificación para una imagen."""
    predictions: list[tuple[int, int, int, int, float]]  # (x, y, w, h, score)
    ground_truth: list[RegionOfInterest]
    

@dataclass
class PatchMetrics:
    """Métricas de evaluación a nivel de PARCHES (clasificación).
    
    Mide si el clasificador SVM distingue correctamente parches de 
    señal vs no-señal en datos de entrenamiento/test.
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    num_samples: int


# Alias para compatibilidad hacia atrás
EvaluationMetrics = PatchMetrics


@dataclass
class DetectionMetrics:
    """Métricas de evaluación a nivel de VENTANA (clasificación binaria).
    
    Cada ventana del sliding window se evalúa como clasificación binaria:
    - ¿La ventana está sobre un GT? (ground truth)
    - ¿El clasificador predice señal? (score >= threshold)
    
    Definiciones:
    - TP (True Positive): Ventana sobre GT con score alto (detectó correctamente)
    - FP (False Positive): Ventana sin GT con score alto (falsa alarma)
    - TN (True Negative): Ventana sin GT con score bajo (correcto rechazo)
    - FN (False Negative): Ventana sobre GT con score bajo (no detectó)
    
    Métricas adicionales:
    - hit_rate: (TP + TN) / total - % de ventanas clasificadas correctamente
    - gt_coverage: % de objetos GT que tienen al menos una detección TP
    
    Usa intersection ratio (fracción del window que solapa con GT)
    para determinar si una ventana "está sobre" un GT.
    """
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float   # (TP + TN) / Total ventanas
    precision: float  # TP / (TP + FP)
    recall: float     # TP / (TP + FN)
    f1: float
    hit_rate: float   # (TP + TN) / total - % de predicciones correctas
    gt_coverage: float  # % de GTs con al menos una detección
    num_images: int
    num_ground_truth: int
    num_gt_detected: int  # Número de GTs con al menos un TP
    num_detections: int  # Windows con score >= threshold
    num_windows: int     # Total de windows evaluadas
    iou_threshold: float



class HOGClassifier:
    """Clasificador de señales de tráfico basado en HOG + SVM.
    
    Usa lazy loading para eficiencia de memoria: las imágenes se cargan
    solo cuando se necesitan y se descargan automáticamente.
    
    Flujo de trabajo:
    1. Cargar referencias a imágenes (sin cargar datos)
    2. Particionar en train/test
    3. Extraer parches (carga imagen -> extrae -> descarga)
    4. Calcular descriptores HOG
    5. Entrenar SVM lineal
    6. Predecir en imágenes de test con ventana deslizante
    7. Visualizar resultados
    """

    def __init__(
        self,
        data_path: str,
        train_ratio: float = 0.8,
        patch_size: tuple[int, int] = (64, 64),
        hog_method: HOGMethod = HOGMethod.SKIMAGE,
        random_seed: int = 42,
        preprocess_fn: callable | None = None,
        **hog_params,
    ):
        """Inicializa el clasificador HOG.
        
        Args:
            data_path: Ruta al directorio con imágenes en formato dgst.
            train_ratio: Proporción de imágenes para entrenamiento.
            patch_size: Tamaño de los parches (height, width).
            hog_method: Método de extracción HOG a usar.
            random_seed: Semilla para reproducibilidad.
            preprocess_fn: Función opcional de preprocesado que toma un objeto
                Image y devuelve un objeto Image procesado. Se aplica antes
                de extraer cualquier parche (positivo o negativo).
            **hog_params: Parámetros adicionales para el extractor HOG.
        """
        self._data_path = Path(data_path)
        self._train_ratio = train_ratio
        self._patch_size = patch_size
        self._hog_method = hog_method
        self._random_seed = random_seed
        self._hog_params = hog_params
        self._preprocess_fn = preprocess_fn
        
        # Crear extractor HOG según método seleccionado
        self._extractor = self._create_extractor(hog_method, patch_size, **hog_params)
        
        # Estado interno - solo referencias, no imágenes cargadas
        self._train_refs: list[ImageReference] = []
        self._test_refs: list[ImageReference] = []
        self._classifier: LinearSVC | None = None
        self._scaler: StandardScaler | None = None
        self._is_trained = False
        self._negatives_per_image: int = 10  # Para persistencia

    def _create_extractor(
        self, 
        method: HOGMethod, 
        patch_size: tuple[int, int],
        **params
    ) -> HOGExtractor:
        """Crea el extractor HOG según el método especificado."""
        if method == HOGMethod.SKIMAGE:
            return SkimageHOGExtractor(patch_size=patch_size, **params)
        elif method == HOGMethod.CUSTOM:
            return CustomHOGExtractor(patch_size=patch_size, **params)
        else:
            raise ValueError(f"Método HOG no soportado: {method}")

    def _discover_image_ids(self) -> list[int]:
        """Descubre los IDs de imágenes disponibles en el directorio."""
        image_ids = []
        for entry in os.listdir(self._data_path):
            entry_path = self._data_path / entry
            if entry_path.is_dir() and entry.isdigit():
                image_ids.append(int(entry))
        return sorted(image_ids)

    def load_and_split(self) -> tuple[int, int]:
        """Descubre imágenes y las divide en train/test sin cargar datos.
        
        Returns:
            Tupla (num_train, num_test) con cantidad de imágenes en cada conjunto.
        """
        image_ids = self._discover_image_ids()
        
        if len(image_ids) == 0:
            raise ValueError(f"No se encontraron imágenes en {self._data_path}")
        
        # Mezclar y dividir
        random.seed(self._random_seed)
        random.shuffle(image_ids)
        
        split_idx = int(len(image_ids) * self._train_ratio)
        train_ids = image_ids[:split_idx]
        test_ids = image_ids[split_idx:]
        
        # Crear referencias (no carga imágenes, solo guarda IDs)
        self._train_refs = [
            ImageReference(image_id=img_id, data_path=self._data_path)
            for img_id in train_ids
        ]
        self._test_refs = [
            ImageReference(image_id=img_id, data_path=self._data_path)
            for img_id in test_ids
        ]
        
        return len(self._train_refs), len(self._test_refs)

    def _roi_to_bbox(self, roi: RegionOfInterest) -> tuple[int, int, int, int]:
        """Convierte ROI de 4 puntos a bounding box (x, y, w, h)."""
        points = [roi.p1, roi.p2, roi.p3, roi.p4]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x = int(min(xs))
        y = int(min(ys))
        w = int(max(xs) - x)
        h = int(max(ys) - y)
        return (x, y, max(w, 1), max(h, 1))

    def _compute_iou(
        self, 
        box1: tuple[int, int, int, int], 
        box2: tuple[int, int, int, int]
    ) -> float:
        """Calcula Intersection over Union entre dos bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calcular intersección
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _compute_intersection_ratio(
        self, 
        window: tuple[int, int, int, int], 
        gt: tuple[int, int, int, int]
    ) -> float:
        """Calcula qué fracción del window intersecta con el GT.
        
        Más apropiado que IoU para sliding window donde el tamaño del
        window es fijo pero el GT puede ser de cualquier tamaño.
        
        Returns:
            Fracción del área del window que intersecta con GT [0, 1].
        """
        wx, wy, ww, wh = window
        gx, gy, gw, gh = gt
        
        # Calcular intersección
        xi1 = max(wx, gx)
        yi1 = max(wy, gy)
        xi2 = min(wx + ww, gx + gw)
        yi2 = min(wy + wh, gy + gh)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        window_area = ww * wh
        
        return inter_area / window_area if window_area > 0 else 0.0

    def _extract_patches_from_image(
        self,
        image: Image,
        img_idx: int,
        negatives_per_image: int,
        iou_threshold: float,
    ) -> list[PatchSample]:
        """Extrae parches de una sola imagen.
        
        Para ROIs grandes (mayores que patch_size), los particiona en múltiples
        sub-parches en lugar de escalarlos. Esto hace que el entrenamiento sea
        consistente con el sliding window de inferencia.
        """
        samples = []
        
        # Aplicar preprocesado si está definido
        if self._preprocess_fn is not None:
            image = self._preprocess_fn(image)
        
        if image.data is None:
            return samples
            
        h, w = image.data.shape[:2]
        roi_bboxes = [self._roi_to_bbox(roi) for roi in image.rois]
        patch_h, patch_w = self._patch_size
        
        # Parches positivos: extraer de cada ROI
        for roi in image.rois:
            bbox = self._roi_to_bbox(roi)
            x, y, bw, bh = bbox
            
            # Validar límites
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            x2 = min(x + bw, w)
            y2 = min(y + bh, h)
            roi_w = x2 - x
            roi_h = y2 - y
            
            if roi_w < 10 or roi_h < 10:
                continue  # ROI demasiado pequeño
            
            # Estrategia: particionar ROIs grandes, escalar ROIs pequeños
            if roi_w <= patch_w and roi_h <= patch_h:
                # ROI pequeño: escalar al tamaño del parche
                patch = image.data[y:y2, x:x2]
                patch_resized = cv2.resize(patch, (patch_w, patch_h))
                samples.append(PatchSample(
                    patch=patch_resized,
                    label=1,
                    bbox=(x, y, roi_w, roi_h),
                    source_image_idx=img_idx,
                ))
            else:
                # ROI grande: particionar en sub-parches con overlap
                # Usar stride de 50% para overlap
                stride_x = max(patch_w // 2, 1)
                stride_y = max(patch_h // 2, 1)
                
                for py in range(y, y2 - patch_h + 1, stride_y):
                    for px in range(x, x2 - patch_w + 1, stride_x):
                        patch = image.data[py:py + patch_h, px:px + patch_w]
                        if patch.shape[0] == patch_h and patch.shape[1] == patch_w:
                            samples.append(PatchSample(
                                patch=patch.copy(),
                                label=1,
                                bbox=(px, py, patch_w, patch_h),
                                source_image_idx=img_idx,
                            ))
                
                # Asegurar al menos un parche del ROI (centrado)
                if len([s for s in samples if s.source_image_idx == img_idx and s.label == 1]) == 0:
                    cx = x + (roi_w - patch_w) // 2
                    cy = y + (roi_h - patch_h) // 2
                    cx = max(0, min(cx, w - patch_w))
                    cy = max(0, min(cy, h - patch_h))
                    patch = image.data[cy:cy + patch_h, cx:cx + patch_w]
                    if patch.shape[0] == patch_h and patch.shape[1] == patch_w:
                        samples.append(PatchSample(
                            patch=patch.copy(),
                            label=1,
                            bbox=(cx, cy, patch_w, patch_h),
                            source_image_idx=img_idx,
                        ))
        
        # Parches negativos: subdivisiones sin intersección significativa
        neg_count = 0
        max_attempts = negatives_per_image * 10
        attempts = 0
        
        while neg_count < negatives_per_image and attempts < max_attempts:
            attempts += 1
            
            # Posición aleatoria
            rand_x = random.randint(0, max(0, w - patch_w))
            rand_y = random.randint(0, max(0, h - patch_h))
            candidate_bbox = (rand_x, rand_y, patch_w, patch_h)
            
            # Verificar que no intersecta con ningún ROI
            is_negative = True
            for roi_bbox in roi_bboxes:
                if self._compute_iou(candidate_bbox, roi_bbox) > iou_threshold:
                    is_negative = False
                    break
            
            if is_negative:
                patch = image.data[rand_y:rand_y + patch_h, rand_x:rand_x + patch_w]
                if patch.shape[0] == patch_h and patch.shape[1] == patch_w:
                    samples.append(PatchSample(
                        patch=patch.copy(),
                        label=0,
                        bbox=candidate_bbox,
                        source_image_idx=img_idx,
                    ))
                    neg_count += 1
        
        return samples

    def extract_patches_lazy(
        self, 
        image_refs: list[ImageReference], 
        negatives_per_image: int = 10,
        iou_threshold: float = 0.1,
    ) -> list[PatchSample]:
        """Extrae parches con lazy loading - carga y descarga cada imagen.
        
        Args:
            image_refs: Lista de referencias a imágenes.
            negatives_per_image: Número de parches negativos por imagen.
            iou_threshold: Umbral máximo de IoU para considerar un parche como negativo.
            
        Returns:
            Lista de PatchSample con parches y etiquetas.
        """
        samples = []
        random.seed(self._random_seed)
        
        for img_idx, ref in enumerate(image_refs):
            # Cargar imagen
            image = ref.load()
            
            # Extraer parches
            img_samples = self._extract_patches_from_image(
                image, img_idx, negatives_per_image, iou_threshold
            )
            samples.extend(img_samples)
            
            # Imagen se descarga automáticamente al salir del scope
            # Forzamos liberación de memoria
            del image
        
        return samples

    def extract_patches(
        self, 
        images: list[Image], 
        negatives_per_image: int = 10,
        iou_threshold: float = 0.1,
    ) -> list[PatchSample]:
        """Extrae parches de imágenes ya cargadas (compatibilidad hacia atrás).
        
        Args:
            images: Lista de imágenes a procesar.
            negatives_per_image: Número de parches negativos por imagen.
            iou_threshold: Umbral máximo de IoU para considerar un parche como negativo.
            
        Returns:
            Lista de PatchSample con parches y etiquetas.
        """
        samples = []
        random.seed(self._random_seed)
        
        for img_idx, image in enumerate(images):
            img_samples = self._extract_patches_from_image(
                image, img_idx, negatives_per_image, iou_threshold
            )
            samples.extend(img_samples)
        
        return samples

    def _compute_features(self, patches: list[PatchSample]) -> np.ndarray:
        """Calcula descriptores HOG para todos los parches."""
        features = []
        for sample in patches:
            feat = self._extractor.compute(sample.patch)
            features.append(feat)
        return np.array(features)

    def _extract_sliding_windows(
        self,
        image_data: np.ndarray,
        stride: int,
    ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
        """Extrae todos los parches de sliding window de una imagen.
        
        Returns:
            Tuple de (lista de parches, lista de bboxes (x, y, w, h))
        """
        patch_h, patch_w = self._patch_size
        h, w = image_data.shape[:2]
        
        patches = []
        bboxes = []
        
        for y in range(0, h - patch_h + 1, stride):
            for x in range(0, w - patch_w + 1, stride):
                patch = image_data[y:y + patch_h, x:x + patch_w]
                patches.append(patch)
                bboxes.append((x, y, patch_w, patch_h))
        
        return patches, bboxes

    def _compute_features_batch(self, patches: list[np.ndarray]) -> np.ndarray:
        """Calcula descriptores HOG para una lista de parches (más eficiente que uno a uno)."""
        if len(patches) == 0:
            return np.array([])
        
        features = np.zeros((len(patches), self._extractor.feature_length))
        for i, patch in enumerate(patches):
            features[i] = self._extractor.compute(patch)
        return features

    def _get_model_filename(self, negatives_per_image: int) -> str:
        """Genera nombre de archivo de modelo basado en parámetros."""
        params = {
            "train_ratio": self._train_ratio,
            "patch_size": self._patch_size,
            "hog_method": self._hog_method.value,
            "random_seed": self._random_seed,
            "negatives_per_image": negatives_per_image,
            **self._hog_params,
        }
        # Crear hash de parámetros para nombre único
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:12]
        
        return f"hog_model_{self._hog_method.value}_ps{self._patch_size[0]}x{self._patch_size[1]}_neg{negatives_per_image}_{params_hash}.pkl"

    def _get_model_path(self, negatives_per_image: int) -> Path:
        """Obtiene ruta completa al archivo de modelo."""
        return self._data_path / self._get_model_filename(negatives_per_image)

    def save_model(self, negatives_per_image: int | None = None) -> Path:
        """Guarda el modelo entrenado en el directorio de datos.
        
        Args:
            negatives_per_image: Número de negativos usados (para nombre de archivo).
            
        Returns:
            Ruta al archivo guardado.
        """
        if not self._is_trained:
            raise RuntimeError("El clasificador no está entrenado.")
        
        negs = negatives_per_image or self._negatives_per_image
        model_path = self._get_model_path(negs)
        
        model_data = {
            "classifier": self._classifier,
            "scaler": self._scaler,
            "patch_size": self._patch_size,
            "hog_method": self._hog_method.value,
            "hog_params": self._hog_params,
            "random_seed": self._random_seed,
            "train_ratio": self._train_ratio,
            "negatives_per_image": negs,
        }
        
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        return model_path

    def load_model(self, negatives_per_image: int) -> bool:
        """Carga un modelo guardado si existe.
        
        Args:
            negatives_per_image: Número de negativos usados (para buscar archivo).
            
        Returns:
            True si se cargó el modelo, False si no existe.
        """
        model_path = self._get_model_path(negatives_per_image)
        
        if not model_path.exists():
            return False
        
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            self._classifier = model_data["classifier"]
            self._scaler = model_data["scaler"]
            self._negatives_per_image = model_data["negatives_per_image"]
            self._is_trained = True
            
            return True
        except (pickle.PickleError, KeyError, EOFError):
            return False

    def _estimate_positives_per_image(self) -> float:
        """Estima el número promedio de parches positivos por imagen.
        
        Útil para balancear automáticamente el número de negativos.
        Solo carga metadatos, no las imágenes completas.
        """
        total_patches = 0
        patch_h, patch_w = self._patch_size
        
        for ref in self._train_refs:
            rois = ref.load_metadata_only()
            for roi in rois:
                bbox = self._roi_to_bbox(roi)
                _, _, roi_w, roi_h = bbox
                
                if roi_w <= patch_w and roi_h <= patch_h:
                    # ROI pequeño: genera 1 parche
                    total_patches += 1
                else:
                    # ROI grande: estimar sub-parches con 50% stride
                    stride_x = max(patch_w // 2, 1)
                    stride_y = max(patch_h // 2, 1)
                    n_x = max(1, (roi_w - patch_w) // stride_x + 1)
                    n_y = max(1, (roi_h - patch_h) // stride_y + 1)
                    total_patches += n_x * n_y
        
        return total_patches / len(self._train_refs) if self._train_refs else 0

    def train(
        self, 
        negatives_per_image: int = 10, 
        force_retrain: bool = False,
        auto_balance_negatives: bool = False,
        balance_ratio: float = 1.5,
    ) -> EvaluationMetrics:
        """Entrena el clasificador SVM con lazy loading de imágenes.
        
        Si existe un modelo guardado con los mismos parámetros, lo carga
        en lugar de reentrenar (a menos que force_retrain=True).
        
        Args:
            negatives_per_image: Parches negativos por imagen.
            force_retrain: Si True, reentrena aunque exista modelo guardado.
            auto_balance_negatives: Si True, calcula automáticamente el número
                de negativos para balancear con los positivos generados por
                particionamiento de ROIs.
            balance_ratio: Ratio de negativos/positivos cuando auto_balance=True.
                Por defecto 1.5 (50% más negativos que positivos).
            
        Returns:
            Métricas de evaluación en conjunto de entrenamiento.
        """
        if len(self._train_refs) == 0:
            raise RuntimeError("Primero llama a load_and_split()")
        
        # Auto-balance: calcular negativos basado en positivos estimados
        if auto_balance_negatives:
            avg_positives = self._estimate_positives_per_image()
            negatives_per_image = max(10, int(avg_positives * balance_ratio))
        
        self._negatives_per_image = negatives_per_image
        
        # Intentar cargar modelo existente
        if not force_retrain and self.load_model(negatives_per_image):
            # Modelo cargado, evaluar en datos de entrenamiento
            samples = self.extract_patches_lazy(self._train_refs, negatives_per_image)
            if len(samples) == 0:
                raise ValueError("No se extrajeron parches de entrenamiento")
            
            X = self._compute_features(samples)
            y = np.array([s.label for s in samples])
            del samples
            
            X_scaled = self._scaler.transform(X)
            y_pred = self._classifier.predict(X_scaled)
            
            return EvaluationMetrics(
                accuracy=accuracy_score(y, y_pred),
                precision=precision_score(y, y_pred, zero_division=0),
                recall=recall_score(y, y_pred, zero_division=0),
                f1=f1_score(y, y_pred, zero_division=0),
                num_samples=len(y),
            )
        
        # Extraer parches con lazy loading
        samples = self.extract_patches_lazy(self._train_refs, negatives_per_image)
        
        if len(samples) == 0:
            raise ValueError("No se extrajeron parches de entrenamiento")
        
        # Calcular features
        X = self._compute_features(samples)
        y = np.array([s.label for s in samples])
        
        # Liberar parches de memoria (ya tenemos los features)
        del samples
        
        # Normalizar
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Entrenar SVM
        self._classifier = LinearSVC(random_state=self._random_seed, max_iter=10000)
        self._classifier.fit(X_scaled, y)
        self._is_trained = True
        
        # Guardar modelo
        self.save_model(negatives_per_image)
        
        # Métricas en training
        y_pred = self._classifier.predict(X_scaled)
        
        return EvaluationMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred, zero_division=0),
            f1=f1_score(y, y_pred, zero_division=0),
            num_samples=len(y),
        )

    def predict_sliding_window(
        self, 
        image: Image, 
        stride: int = 16,
        scales: list[float] | None = None,
        score_threshold: float = 0.0,
    ) -> list[tuple[int, int, int, int, float]]:
        """Aplica ventana deslizante y clasifica cada parche.
        
        Args:
            image: Imagen a procesar.
            stride: Paso de la ventana deslizante.
            scales: Escalas a evaluar. Por defecto [1.0, 0.75, 0.5].
            score_threshold: Umbral mínimo de score para reportar detección.
            
        Returns:
            Lista de (x, y, w, h, score) para cada detección positiva.
        """
        if not self._is_trained:
            raise RuntimeError("El clasificador no está entrenado. Llama a train() primero.")
        
        if scales is None:
            scales = [1.0, 0.75, 0.5]
        
        detections = []
        patch_h, patch_w = self._patch_size
        
        # Aplicar preprocesado si está definido
        if self._preprocess_fn is not None:
            image = self._preprocess_fn(image)
        
        for scale in scales:
            # Redimensionar imagen
            if scale != 1.0:
                new_w = int(image.data.shape[1] * scale)
                new_h = int(image.data.shape[0] * scale)
                scaled_img = cv2.resize(image.data, (new_w, new_h))
            else:
                scaled_img = image.data
                
            h, w = scaled_img.shape[:2]
            
            # Ventana deslizante
            for y in range(0, h - patch_h + 1, stride):
                for x in range(0, w - patch_w + 1, stride):
                    patch = scaled_img[y:y + patch_h, x:x + patch_w]
                    
                    # Calcular HOG y predecir
                    feat = self._extractor.compute(patch).reshape(1, -1)
                    feat_scaled = self._scaler.transform(feat)
                    
                    # Usar decision_function para obtener score
                    score = self._classifier.decision_function(feat_scaled)[0]
                    
                    if score > score_threshold:
                        # Convertir coordenadas a escala original
                        orig_x = int(x / scale)
                        orig_y = int(y / scale)
                        orig_w = int(patch_w / scale)
                        orig_h = int(patch_h / scale)
                        
                        detections.append((orig_x, orig_y, orig_w, orig_h, score))
        
        return detections

    def _non_max_suppression(
        self, 
        detections: list[tuple[int, int, int, int, float]], 
        iou_threshold: float = 0.3
    ) -> list[tuple[int, int, int, int, float]]:
        """Aplica Non-Maximum Suppression para eliminar detecciones redundantes."""
        if len(detections) == 0:
            return []
        
        # Ordenar por score descendente
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                d for d in detections
                if self._compute_iou(
                    (best[0], best[1], best[2], best[3]),
                    (d[0], d[1], d[2], d[3])
                ) < iou_threshold
            ]
        
        return keep

    def visualize_predictions(
        self, 
        image: Image,
        stride: int = 16,
        scales: list[float] | None = None,
        score_threshold: float = 0.0,
        nms_threshold: float = 0.3,
        show_ground_truth: bool = True,
    ) -> Image:
        """Visualiza predicciones como mapa de calor sobre la imagen.
        
        Cada ventana del sliding window se colorea en verde con opacidad
        proporcional al score. Ground truth se dibuja en rojo.
        
        Args:
            image: Imagen a procesar.
            stride: Paso de ventana deslizante.
            scales: Escalas a evaluar (solo usa la primera para visualización).
            score_threshold: Umbral mínimo de score para colorear (default 0).
            nms_threshold: No usado en esta visualización.
            show_ground_truth: Si True, dibuja los ROIs reales en rojo.
            
        Returns:
            Imagen con overlay de mapa de calor.
        """
        if not self._is_trained:
            raise RuntimeError("El clasificador no está entrenado. Llama a train() primero.")
        
        # Aplicar preprocesado si está definido
        if self._preprocess_fn is not None:
            image = self._preprocess_fn(image)
        
        # Preparar imagen para visualización
        result = image.clone()
        if result.is_grayscale:
            result.data = cv2.cvtColor(result.data, cv2.COLOR_GRAY2BGR)
        else:
            result.data = result.data.copy().astype(np.float32)
        
        # Crear capa de overlay
        overlay = np.zeros_like(result.data, dtype=np.float32)
        count_map = np.zeros((result.data.shape[0], result.data.shape[1]), dtype=np.float32)
        
        patch_h, patch_w = self._patch_size
        
        h, w = result.data.shape[:2]
        
        # Extraer todas las ventanas en batch (usar imagen procesada)
        image_uint8 = result.data.astype(np.uint8) if result.data.dtype != np.uint8 else result.data
        patches, bboxes = self._extract_sliding_windows(image_uint8, stride)
        
        if len(patches) > 0:
            # Calcular HOG features en batch
            features = self._compute_features_batch(patches)
            features_scaled = self._scaler.transform(features)
            scores = self._classifier.decision_function(features_scaled)
            
            # Aplicar scores al overlay
            for (x, y, pw, ph), score in zip(bboxes, scores):
                if score > score_threshold:
                    # Normalizar score a [0, 1] para opacidad
                    # Usar función cuadrática para gradiente más pronunciado:
                    # - Scores bajos (0-0.5): casi invisibles
                    # - Scores altos (>1.0): muy visibles
                    normalized = max(0.0, min(1.0, score / 1.5))
                    # Aplicar gamma para que solo scores altos sean prominentes
                    alpha = normalized ** 2.5
                    
                    # Añadir verde con opacidad al overlay
                    overlay[y:y + patch_h, x:x + patch_w, 1] += alpha * 255  # Canal verde
                    count_map[y:y + patch_h, x:x + patch_w] += 1
        
        # Normalizar overlay por número de contribuciones
        count_map = np.maximum(count_map, 1)  # Evitar división por cero
        for c in range(3):
            overlay[:, :, c] /= count_map
        
        # Mezclar overlay con imagen original
        result.data = result.data.astype(np.float32)
        alpha_blend = 0.6  # Transparencia global del overlay (más opaco)
        result.data = result.data * (1 - alpha_blend) + overlay * alpha_blend
        result.data = np.clip(result.data, 0, 255).astype(np.uint8)
        
        # Dibujar ground truth en ROJO (encima del heatmap)
        if show_ground_truth:
            for roi in image.rois:
                pts = np.array([roi.p1, roi.p2, roi.p3, roi.p4], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(result.data, [pts], True, (0, 0, 255), 3)  # ROJO grueso
        
        # Actualizar metadata
        result.metadata.add_step({
            "technique": "hog_heatmap",
            "num_ground_truth": len(image.rois),
            "stride": stride,
        })
        
        return result

    def visualize_predictions_by_ref(
        self,
        image_ref: ImageReference,
        stride: int = 16,
        scales: list[float] | None = None,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        show_ground_truth: bool = True,
    ) -> Image:
        """Visualiza predicciones cargando imagen desde referencia.
        
        Args:
            image_ref: Referencia a la imagen a procesar.
            stride: Paso de ventana deslizante.
            scales: Escalas a evaluar.
            score_threshold: Umbral de score para detecciones.
            nms_threshold: Umbral IoU para NMS.
            show_ground_truth: Si True, dibuja también los ROIs reales.
            
        Returns:
            Imagen con bounding boxes dibujados.
        """
        image = image_ref.load()
        result = self.visualize_predictions(
            image, stride, scales, score_threshold, nms_threshold, show_ground_truth
        )
        del image  # Liberar memoria de imagen original
        return result

    def evaluate(self, negatives_per_image: int = 10) -> EvaluationMetrics:
        """Evalúa el clasificador en el conjunto de test con lazy loading.
        
        Args:
            negatives_per_image: Parches negativos por imagen para evaluación.
            
        Returns:
            Métricas de evaluación en test.
        """
        if not self._is_trained:
            raise RuntimeError("El clasificador no está entrenado. Llama a train() primero.")
        
        if len(self._test_refs) == 0:
            raise RuntimeError("No hay imágenes de test. Llama a load_and_split() primero.")
        
        # Extraer parches de test con lazy loading
        samples = self.extract_patches_lazy(self._test_refs, negatives_per_image)
        
        if len(samples) == 0:
            raise ValueError("No se extrajeron parches de test")
        
        # Calcular features y predecir
        X = self._compute_features(samples)
        y = np.array([s.label for s in samples])
        
        # Liberar parches
        del samples
        
        X_scaled = self._scaler.transform(X)
        y_pred = self._classifier.predict(X_scaled)
        
        return EvaluationMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred, zero_division=0),
            f1=f1_score(y, y_pred, zero_division=0),
            num_samples=len(y),
        )

    def evaluate_detections(
        self,
        stride: int = 16,
        scales: list[float] | None = None,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        intersection_threshold: float = 0.3,
        max_images: int | None = None,
        n_jobs: int = 4,
    ) -> DetectionMetrics:
        """Evalúa detecciones iterando ventana por ventana (paralelizado).
        
        Para cada ventana con score >= threshold, comprueba si intersecta
        significativamente con algún ground truth usando intersection ratio
        (fracción del window que solapa con GT).
        
        Args:
            stride: Paso de ventana deslizante.
            scales: No usado (se usa escala 1.0).
            score_threshold: Umbral de score para considerar una ventana como detección.
            nms_threshold: No usado en esta evaluación.
            intersection_threshold: Umbral mínimo de intersection ratio para TP.
                Un valor de 0.3 significa que al menos 30% del window debe
                solapar con el GT para considerarse un match.
            max_images: Máximo de imágenes a evaluar (None = todas).
            n_jobs: Número de threads para procesamiento paralelo.
            
        Returns:
            DetectionMetrics con TP, FP, TN, FN, accuracy, precision, recall, F1.
        """
        if not self._is_trained:
            raise RuntimeError("El clasificador no está entrenado. Llama a train() primero.")
        
        if len(self._test_refs) == 0:
            raise RuntimeError("No hay imágenes de test. Llama a load_and_split() primero.")
        
        # Seleccionar subconjunto si se especifica max_images
        test_refs = self._test_refs
        if max_images is not None and max_images < len(test_refs):
            random.seed(self._random_seed)
            test_refs = random.sample(test_refs, max_images)
        
        def process_single_image(ref: ImageReference) -> dict:
            """Procesa una imagen y retorna contadores locales."""
            image = ref.load()
            
            # Aplicar preprocesado si está definido
            if self._preprocess_fn is not None:
                image = self._preprocess_fn(image)
            
            # Obtener ground truth bboxes
            gt_bboxes = [self._roi_to_bbox(roi) for roi in image.rois]
            local_gt = len(gt_bboxes)
            
            local_tp = 0
            local_fp = 0
            local_tn = 0
            local_fn = 0
            local_windows = 0
            local_windows_positive = 0
            
            # Track which GTs have been detected (at least one TP)
            gt_detected = [False] * local_gt
            
            # Extraer todas las ventanas en batch
            patches, bboxes = self._extract_sliding_windows(image.data, stride)
            local_windows = len(patches)
            
            if len(patches) == 0:
                del image
                return {
                    "tp": 0, "fp": 0, "tn": 0, "fn": 0, "gt": local_gt,
                    "gt_detected": 0, "windows": 0, "windows_positive": 0
                }
            
            # Calcular HOG features en batch
            features = self._compute_features_batch(patches)
            features_scaled = self._scaler.transform(features)
            scores = self._classifier.decision_function(features_scaled)
            
            # Evaluar cada ventana como clasificación binaria
            for window_bbox, score in zip(bboxes, scores):
                # Calcular mejor intersección con GT y track which GT
                best_intersection = 0.0
                best_gt_idx = -1
                for gt_idx, gt_bbox in enumerate(gt_bboxes):
                    inter_ratio = self._compute_intersection_ratio(window_bbox, gt_bbox)
                    if inter_ratio > best_intersection:
                        best_intersection = inter_ratio
                        best_gt_idx = gt_idx
                
                is_over_gt = best_intersection >= intersection_threshold
                is_positive_prediction = score >= score_threshold
                
                if is_positive_prediction:
                    local_windows_positive += 1
                    if is_over_gt:
                        # TP: ventana sobre GT con score alto (detectó correctamente)
                        local_tp += 1
                        # Mark this GT as detected
                        if best_gt_idx >= 0:
                            gt_detected[best_gt_idx] = True
                    else:
                        # FP: ventana sin GT con score alto (falsa alarma)
                        local_fp += 1
                else:
                    # Score bajo
                    if is_over_gt:
                        # FN: ventana sobre GT con score bajo (no detectó)
                        local_fn += 1
                    else:
                        # TN: ventana sin GT con score bajo (correcto rechazo)
                        local_tn += 1
            
            local_gt_detected = sum(gt_detected)
            
            del image
            return {
                "tp": local_tp, "fp": local_fp, "tn": local_tn, "fn": local_fn,
                "gt": local_gt, "gt_detected": local_gt_detected,
                "windows": local_windows, "windows_positive": local_windows_positive
            }
        
        # Procesar imágenes en paralelo
        if n_jobs > 1 and len(test_refs) > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(process_single_image, test_refs))
        else:
            results = [process_single_image(ref) for ref in test_refs]
        
        # Agregar resultados
        total_tp = sum(r["tp"] for r in results)
        total_fp = sum(r["fp"] for r in results)
        total_tn = sum(r["tn"] for r in results)
        total_fn = sum(r["fn"] for r in results)
        total_gt = sum(r["gt"] for r in results)
        total_gt_detected = sum(r["gt_detected"] for r in results)
        total_windows = sum(r["windows"] for r in results)
        total_windows_positive = sum(r["windows_positive"] for r in results)
        
        # Calcular métricas a nivel de ventana
        accuracy = (total_tp + total_tn) / total_windows if total_windows > 0 else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Nuevas métricas más intuitivas
        # hit_rate: % de ventanas clasificadas correctamente
        hit_rate = (total_tp + total_tn) / total_windows if total_windows > 0 else 0.0
        
        # gt_coverage: % de objetos GT que fueron detectados al menos una vez
        gt_coverage = total_gt_detected / total_gt if total_gt > 0 else 0.0
        
        return DetectionMetrics(
            true_positives=total_tp,
            false_positives=total_fp,
            true_negatives=total_tn,
            false_negatives=total_fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            hit_rate=hit_rate,
            gt_coverage=gt_coverage,
            num_images=len(test_refs),
            num_ground_truth=total_gt,
            num_gt_detected=total_gt_detected,
            num_detections=total_windows_positive,
            num_windows=total_windows,
            iou_threshold=intersection_threshold,  # Mantener nombre por compatibilidad
        )

    def iterate_train_images(self) -> Iterator[Image]:
        """Itera sobre imágenes de entrenamiento con lazy loading.
        
        Yields:
            Image: Cada imagen de entrenamiento, cargada bajo demanda.
        """
        for ref in self._train_refs:
            yield ref.load()

    def iterate_test_images(self) -> Iterator[Image]:
        """Itera sobre imágenes de test con lazy loading.
        
        Yields:
            Image: Cada imagen de test, cargada bajo demanda.
        """
        for ref in self._test_refs:
            yield ref.load()

    def load_train_image(self, index: int) -> Image:
        """Carga una imagen de entrenamiento específica.
        
        Args:
            index: Índice de la imagen en el conjunto de entrenamiento.
            
        Returns:
            Imagen cargada.
        """
        if index < 0 or index >= len(self._train_refs):
            raise IndexError(f"Índice {index} fuera de rango [0, {len(self._train_refs)})")
        return self._train_refs[index].load()

    def load_test_image(self, index: int) -> Image:
        """Carga una imagen de test específica.
        
        Args:
            index: Índice de la imagen en el conjunto de test.
            
        Returns:
            Imagen cargada.
        """
        if index < 0 or index >= len(self._test_refs):
            raise IndexError(f"Índice {index} fuera de rango [0, {len(self._test_refs)})")
        return self._test_refs[index].load()

    @property
    def train_refs(self) -> list[ImageReference]:
        """Referencias a imágenes de entrenamiento (sin datos cargados)."""
        return self._train_refs

    @property
    def test_refs(self) -> list[ImageReference]:
        """Referencias a imágenes de test (sin datos cargados)."""
        return self._test_refs

    @property
    def num_train_images(self) -> int:
        """Número de imágenes de entrenamiento."""
        return len(self._train_refs)

    @property
    def num_test_images(self) -> int:
        """Número de imágenes de test."""
        return len(self._test_refs)

    # Propiedades de compatibilidad hacia atrás
    @property
    def train_images(self) -> list[Image]:
        """Carga y retorna todas las imágenes de entrenamiento.
        
        ADVERTENCIA: Esto carga todas las imágenes en memoria.
        Usa iterate_train_images() o load_train_image() para lazy loading.
        """
        return [ref.load() for ref in self._train_refs]

    @property
    def test_images(self) -> list[Image]:
        """Carga y retorna todas las imágenes de test.
        
        ADVERTENCIA: Esto carga todas las imágenes en memoria.
        Usa iterate_test_images() o load_test_image() para lazy loading.
        """
        return [ref.load() for ref in self._test_refs]

    @property
    def is_trained(self) -> bool:
        """Indica si el clasificador está entrenado."""
        return self._is_trained

    @property
    def hog_method(self) -> HOGMethod:
        """Método HOG en uso."""
        return self._hog_method

    @property
    def feature_length(self) -> int:
        """Longitud del vector de características HOG."""
        return self._extractor.feature_length
