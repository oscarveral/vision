"""
BT3a: Comparación de detectores y extractores de descriptores locales.

Este módulo implementa la extracción y comparación de features locales invariantes
(SIFT, AKAZE, ORB) para el reconocimiento de señales de tráfico.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
import time

import cv2
import numpy as np

from dgst.utils.loader import Image

if TYPE_CHECKING:
    from dgst.utils.processor import ImageProcessor


class DescriptorMethod(Enum):
    """Métodos de extracción de descriptores locales disponibles."""

    SIFT = "sift"
    AKAZE = "akaze"
    ORB = "orb"


class KeyPoint:
    """Punto de interés detectado en una imagen."""
    
    __slots__ = ("_cv_kp",)

    def __init__(self, cv_keypoint: cv2.KeyPoint):
        """Inicializa desde un cv2.KeyPoint (uso interno)."""
        self._cv_kp = cv_keypoint

    @property
    def x(self) -> float:
        """Coordenada x del keypoint."""
        return self._cv_kp.pt[0]

    @property
    def y(self) -> float:
        """Coordenada y del keypoint."""
        return self._cv_kp.pt[1]

    @property
    def position(self) -> tuple[float, float]:
        """Posición (x, y) del keypoint."""
        return self._cv_kp.pt

    @property
    def size(self) -> float:
        """Diámetro de la región del keypoint."""
        return self._cv_kp.size

    @property
    def angle(self) -> float:
        """Orientación del keypoint en grados [0, 360)."""
        return self._cv_kp.angle

    @property
    def response(self) -> float:
        """Respuesta del detector (calidad del keypoint)."""
        return self._cv_kp.response

    @property
    def octave(self) -> int:
        """Octava de la pirámide donde se detectó."""
        return self._cv_kp.octave

    def __repr__(self) -> str:
        return f"KeyPoint(x={self.x:.1f}, y={self.y:.1f}, size={self.size:.1f})"


class ExtractionResult:
    """Resultado de la extracción de keypoints y descriptores."""
    
    __slots__ = ("_keypoints", "_descriptors", "_method", "_extraction_time_ms", "_image_shape")

    def __init__(
        self,
        cv_keypoints: list,
        cv_descriptors: np.ndarray | None,
        method: DescriptorMethod,
        extraction_time_ms: float,
        image_shape: tuple[int, int],
    ):
        """Inicializa desde resultados de OpenCV (uso interno)."""
        self._keypoints = [KeyPoint(kp) for kp in cv_keypoints]
        self._descriptors = cv_descriptors
        self._method = method
        self._extraction_time_ms = extraction_time_ms
        self._image_shape = image_shape

    @property
    def keypoints(self) -> list[KeyPoint]:
        """Lista de keypoints detectados."""
        return self._keypoints

    @property
    def method(self) -> DescriptorMethod:
        """Método usado para la extracción."""
        return self._method

    @property
    def extraction_time_ms(self) -> float:
        """Tiempo de extracción en milisegundos."""
        return self._extraction_time_ms

    @property
    def image_width(self) -> int:
        """Ancho de la imagen procesada."""
        return self._image_shape[1]

    @property
    def image_height(self) -> int:
        """Alto de la imagen procesada."""
        return self._image_shape[0]

    @property
    def num_keypoints(self) -> int:
        """Número de keypoints detectados."""
        return len(self._keypoints)

    @property
    def descriptor_dimension(self) -> int:
        """Dimensión del descriptor (ej: 128 para SIFT)."""
        if self._descriptors is None or len(self._descriptors) == 0:
            return 0
        return self._descriptors.shape[1]

    @property
    def is_binary_descriptor(self) -> bool:
        """True si el descriptor es binario (ORB, AKAZE)."""
        return self._method in (DescriptorMethod.ORB, DescriptorMethod.AKAZE)

    def has_descriptors(self) -> bool:
        """Indica si se extrajeron descriptores."""
        return self._descriptors is not None and len(self._descriptors) > 0

    def __repr__(self) -> str:
        return (
            f"ExtractionResult(method={self._method.value}, "
            f"keypoints={self.num_keypoints}, time={self._extraction_time_ms:.1f}ms)"
        )


@dataclass
class MatchStats:
    """Estadísticas de matching entre imagen original y transformada."""

    method: DescriptorMethod
    num_keypoints_original: int = 0
    num_keypoints_transformed: int = 0
    num_matches: int = 0
    num_good_matches: int = 0
    repeatability: float = 0.0
    extraction_time_ms: float = 0.0


class LocalDescriptorExtractor:
    """Extractor de descriptores locales para comparación de métodos."""

    def __init__(self):
        """Inicializa los detectores/descriptores de OpenCV."""
        self._detectors = {
            DescriptorMethod.SIFT: cv2.SIFT_create(),
            DescriptorMethod.AKAZE: cv2.AKAZE_create(),
            DescriptorMethod.ORB: cv2.ORB_create(nfeatures=1000),
        }

    def extract(
        self,
        image: Image,
        method: DescriptorMethod,
    ) -> ExtractionResult:
        """Extrae keypoints y descriptores de una imagen."""
        if image.data is None:
            raise ValueError("Image data is None")

        start = time.perf_counter()

        # Convertir a escala de grises si es necesario.
        if image.is_color:
            gray = cv2.cvtColor(image.data, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.data.copy()

        # Asegurar que sea uint8
        if gray.dtype != np.uint8:
            if gray.dtype in (np.float32, np.float64):
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)

        detector = self._detectors[method]
        cv_keypoints, cv_descriptors = detector.detectAndCompute(gray, None)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return ExtractionResult(
            cv_keypoints=cv_keypoints,
            cv_descriptors=cv_descriptors,
            method=method,
            extraction_time_ms=elapsed_ms,
            image_shape=gray.shape[:2],
        )

    def compare_methods(
        self,
        image: Image,
        transform: ImageProcessor,
        methods: list[DescriptorMethod] | None = None,
    ) -> dict[DescriptorMethod, MatchStats]:
        """Compara rendimiento de diferentes métodos bajo una transformación."""
        if methods is None:
            methods = list(DescriptorMethod)

        # Aplicar transformación usando el ImageProcessor.
        transformed_image = transform.process(image.clone())

        results = {}

        for method in methods:
            # Extraer de imagen original
            result1 = self.extract(image, method)

            # Extraer de imagen transformada
            result2 = self.extract(transformed_image, method)

            stats = MatchStats(
                method=method,
                num_keypoints_original=result1.num_keypoints,
                num_keypoints_transformed=result2.num_keypoints,
                extraction_time_ms=(result1.extraction_time_ms + result2.extraction_time_ms) / 2,
            )

            # Hacer matching si hay descriptores
            if result1.has_descriptors() and result2.has_descriptors():
                matches, good_matches = self._match_results(result1, result2, method)
                stats.num_matches = len(matches)
                stats.num_good_matches = len(good_matches)

                # Calcular repetibilidad aproximada
                if result1.num_keypoints > 0:
                    stats.repeatability = len(good_matches) / result1.num_keypoints

            results[method] = stats

        return results

    def _match_results(
        self,
        result1: ExtractionResult,
        result2: ExtractionResult,
        method: DescriptorMethod,
    ) -> tuple[list, list]:
        """Hace matching entre dos conjuntos de descriptores (interno).
        
        Retorna matches únicos: cada keypoint de result1 matchea con a lo sumo
        un keypoint de result2, y viceversa.
        """
        desc1 = result1._descriptors
        desc2 = result2._descriptors

        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return [], []

        # Seleccionar norma según tipo de descriptor
        if method == DescriptorMethod.SIFT:
            norm_type = cv2.NORM_L2
        else:
            norm_type = cv2.NORM_HAMMING

        bf = cv2.BFMatcher(norm_type)
        matches = bf.knnMatch(desc1, desc2, k=2)

        # Aplicar ratio test y recopilar candidatos
        candidates = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                is_good = m.distance < 0.75 * n.distance
                candidates.append((m, is_good))
            elif len(match_pair) == 1:
                candidates.append((match_pair[0], False))

        # Filtrar para obtener matches únicos por queryIdx y trainIdx
        # Primero ordenar por distancia para quedarnos con los mejores
        candidates.sort(key=lambda x: x[0].distance)
        
        used_query = set()  # keypoints usados de imagen 1
        used_train = set()  # keypoints usados de imagen 2
        
        unique_all = []
        unique_good = []
        
        for match, is_good in candidates:
            if match.queryIdx not in used_query and match.trainIdx not in used_train:
                used_query.add(match.queryIdx)
                used_train.add(match.trainIdx)
                unique_all.append(match)
                if is_good:
                    unique_good.append(match)

        return unique_all, unique_good

    def visualize_keypoints(
        self,
        image: Image,
        result: ExtractionResult,
        color: tuple[int, int, int] = (0, 255, 0),
        rich_keypoints: bool = True,
    ) -> Image:
        """Visualiza keypoints sobre la imagen.

        Args:
            image: Imagen original.
            result: Resultado de extracción con keypoints.
            color: Color BGR para los keypoints.
            rich_keypoints: Si True, dibuja tamaño y orientación.

        Returns:
            Nueva imagen con keypoints dibujados.
        """
        if image.is_grayscale:
            vis_data = cv2.cvtColor(image.data, cv2.COLOR_GRAY2BGR)
        else:
            vis_data = image.data.copy()

        # Obtener cv2 keypoints internos para visualización
        cv_keypoints = [kp._cv_kp for kp in result.keypoints]

        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich_keypoints else 0
        result_data = cv2.drawKeypoints(
            vis_data, cv_keypoints, None, color=color, flags=flags
        )

        result_image = image.clone()
        result_image.data = result_data
        result_image.metadata.add_step({
            "technique": "visualize_keypoints",
            "method": result.method.value,
            "num_keypoints": result.num_keypoints,
        })

        return result_image

    def visualize_comparison(
        self,
        image: Image,
        methods: list[DescriptorMethod] | None = None,
    ) -> dict[DescriptorMethod, Image]:
        """Genera visualizaciones de keypoints para cada método.

        Args:
            image: Imagen de entrada.
            methods: Métodos a visualizar. Si None, todos.

        Returns:
            Diccionario con imagen visualizada para cada método.
        """
        if methods is None:
            methods = list(DescriptorMethod)

        results = {}
        colors = {
            DescriptorMethod.SIFT: (0, 255, 0),
            DescriptorMethod.AKAZE: (255, 0, 0),
            DescriptorMethod.ORB: (0, 0, 255),
        }

        for method in methods:
            extraction_result = self.extract(image, method)
            results[method] = self.visualize_keypoints(
                image, extraction_result, color=colors[method]
            )

        return results
