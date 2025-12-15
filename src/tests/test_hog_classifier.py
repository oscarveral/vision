"""Tests for HOGClassifier (BT3 - HOG Traffic Sign Classification)."""

import numpy as np
import pytest
import cv2
import os
from pathlib import Path

from dgst.utils.loader import Image, RegionOfInterest
from dgst.utils.hog_classifier import (
    HOGClassifier,
    HOGMethod,
    HOGExtractor,
    SkimageHOGExtractor,
    CustomHOGExtractor,
    PatchSample,
    EvaluationMetrics,
)


# Directorio de imágenes de test (bt3)
BT3_TEST_PATH = Path(__file__).parent.parent.parent / "notebooks" / "images" / "bt3" / "test"
BT3_REF_PATH = Path(__file__).parent.parent.parent / "notebooks" / "images" / "bt3" / "reference"


def create_synthetic_image_with_roi() -> Image:
    """Crea una imagen sintética con un ROI para testing."""
    # Imagen 200x200 con fondo gris
    data = np.full((200, 200, 3), 128, dtype=np.uint8)
    
    # Añadir un "círculo rojo" simulando señal en (100, 100)
    cv2.circle(data, (100, 100), 30, (0, 0, 255), -1)
    
    # ROI que encierra el círculo
    roi = RegionOfInterest(
        p1=(70.0, 70.0),
        p2=(130.0, 70.0),
        p3=(130.0, 130.0),
        p4=(70.0, 130.0),
    )
    
    return Image(data=data, rois=[roi])


class TestSkimageHOGExtractor:
    """Tests para el extractor HOG de skimage."""

    def test_compute_returns_correct_shape(self):
        """Test que compute devuelve un vector de la longitud correcta."""
        extractor = SkimageHOGExtractor(
            patch_size=(64, 64),
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
        )
        
        patch = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        features = extractor.compute(patch)
        
        assert features.shape == (extractor.feature_length,)
        assert extractor.feature_length == 1764  # 7*7*2*2*9

    def test_compute_with_color_image(self):
        """Test que compute funciona con imágenes a color."""
        extractor = SkimageHOGExtractor(patch_size=(64, 64))
        
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        features = extractor.compute(patch)
        
        assert features.shape == (extractor.feature_length,)

    def test_compute_with_resize(self):
        """Test que compute redimensiona parches incorrectos."""
        extractor = SkimageHOGExtractor(patch_size=(64, 64))
        
        # Parche de tamaño diferente
        patch = np.random.randint(0, 255, (100, 80), dtype=np.uint8)
        features = extractor.compute(patch)
        
        assert features.shape == (extractor.feature_length,)

    def test_patch_size_property(self):
        """Test propiedad patch_size."""
        extractor = SkimageHOGExtractor(patch_size=(128, 64))
        assert extractor.patch_size == (128, 64)


class TestCustomHOGExtractor:
    """Tests para el extractor HOG propio."""

    def test_compute_returns_correct_shape(self):
        """Test que compute devuelve un vector de la longitud correcta."""
        extractor = CustomHOGExtractor(
            patch_size=(64, 64),
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
        )
        
        patch = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        features = extractor.compute(patch)
        
        assert features.shape == (extractor.feature_length,)
        assert extractor.feature_length == 1764  # 7*7*2*2*9

    def test_compute_with_color_image(self):
        """Test que compute funciona con imágenes a color."""
        extractor = CustomHOGExtractor(patch_size=(64, 64))
        
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        features = extractor.compute(patch)
        
        assert features.shape == (extractor.feature_length,)

    def test_same_length_as_skimage(self):
        """Test que CustomHOG produce misma longitud que SkimageHOG."""
        custom = CustomHOGExtractor(patch_size=(64, 64))
        skimage = SkimageHOGExtractor(patch_size=(64, 64))
        
        assert custom.feature_length == skimage.feature_length

    def test_classifier_with_custom_hog(self):
        """Test que el clasificador funciona con HOG propio."""
        clf = HOGClassifier(
            data_path="/tmp/fake_path",
            hog_method=HOGMethod.CUSTOM,
            patch_size=(64, 64),
        )
        
        assert clf.hog_method == HOGMethod.CUSTOM
        assert clf.feature_length == 1764


class TestHOGClassifier:
    """Tests para el clasificador HOG completo."""

    def test_init_with_skimage_method(self):
        """Test inicialización con método skimage."""
        clf = HOGClassifier(
            data_path="/tmp/fake_path",
            hog_method=HOGMethod.SKIMAGE,
            patch_size=(64, 64),
        )
        
        assert clf.hog_method == HOGMethod.SKIMAGE
        assert clf.feature_length == 1764
        assert not clf.is_trained

    def test_roi_to_bbox(self):
        """Test conversión de ROI a bounding box."""
        clf = HOGClassifier(data_path="/tmp/fake")
        
        roi = RegionOfInterest(
            p1=(10.0, 20.0),
            p2=(50.0, 20.0),
            p3=(50.0, 60.0),
            p4=(10.0, 60.0),
        )
        
        bbox = clf._roi_to_bbox(roi)
        assert bbox == (10, 20, 40, 40)  # (x, y, w, h)

    def test_compute_iou(self):
        """Test cálculo de IoU."""
        clf = HOGClassifier(data_path="/tmp/fake")
        
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 10, 10)
        
        iou = clf._compute_iou(box1, box2)
        # Intersección: 5x5 = 25, Unión: 100 + 100 - 25 = 175
        assert abs(iou - 25/175) < 0.01

    def test_compute_iou_no_overlap(self):
        """Test IoU sin solapamiento."""
        clf = HOGClassifier(data_path="/tmp/fake")
        
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 10, 10)
        
        iou = clf._compute_iou(box1, box2)
        assert iou == 0.0

    def test_extract_patches_synthetic(self):
        """Test extracción de parches con imagen sintética.
        
        Con ROIs grandes (mayores que patch_size), se particionan en
        múltiples sub-parches con 50% overlap. El ROI de 60x60 con
        patch_size de 32x32 genera 4 parches: 2x2 con stride 16.
        """
        clf = HOGClassifier(data_path="/tmp/fake", patch_size=(32, 32))
        
        image = create_synthetic_image_with_roi()
        
        samples = clf.extract_patches([image], negatives_per_image=5)
        
        # ROI de 60x60 > patch 32x32, se particiona en múltiples parches
        # Con stride 16 (50% overlap): 2 posiciones en X * 2 en Y = 4 parches
        positives = [s for s in samples if s.label == 1]
        negatives = [s for s in samples if s.label == 0]
        
        assert len(positives) >= 1  # Al menos 1 parche positivo (puede ser más si es grande)
        assert len(negatives) == 5

    def test_compute_features(self):
        """Test cálculo de features HOG."""
        clf = HOGClassifier(data_path="/tmp/fake", patch_size=(64, 64))
        
        image = create_synthetic_image_with_roi()
        samples = clf.extract_patches([image], negatives_per_image=3)
        
        features = clf._compute_features(samples)
        
        assert features.shape == (len(samples), clf.feature_length)


class TestHOGClassifierWithRealData:
    """Tests con datos reales si están disponibles."""

    @pytest.fixture
    def classifier_with_data(self):
        """Fixture que carga clasificador con datos reales."""
        # Intentar con directorio de test primero
        if BT3_TEST_PATH.exists():
            return HOGClassifier(
                data_path=str(BT3_TEST_PATH),
                train_ratio=0.7,
                patch_size=(64, 64),
            )
        elif BT3_REF_PATH.exists():
            return HOGClassifier(
                data_path=str(BT3_REF_PATH),
                train_ratio=0.7,
                patch_size=(64, 64),
            )
        else:
            pytest.skip("No se encontró directorio bt3 de imágenes")

    def test_load_and_split(self, classifier_with_data):
        """Test carga y división de datos."""
        clf = classifier_with_data
        
        n_train, n_test = clf.load_and_split()
        
        assert n_train > 0
        assert n_test >= 0
        assert n_train + n_test > 0
        assert len(clf.train_images) == n_train
        assert len(clf.test_images) == n_test

    def test_train(self, classifier_with_data):
        """Test entrenamiento del clasificador."""
        clf = classifier_with_data
        clf.load_and_split()
        
        metrics = clf.train(negatives_per_image=5)
        
        assert clf.is_trained
        assert isinstance(metrics, EvaluationMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.num_samples > 0

    def test_predict_sliding_window(self, classifier_with_data):
        """Test predicción con ventana deslizante."""
        clf = classifier_with_data
        clf.load_and_split()
        clf.train(negatives_per_image=5)
        
        if len(clf.test_images) > 0:
            detections = clf.predict_sliding_window(
                clf.test_images[0],
                stride=32,
                scales=[1.0],
                score_threshold=-1.0,  # Obtener todas las detecciones
            )
            
            # Debe devolver una lista de detecciones
            assert isinstance(detections, list)

    def test_visualize_predictions(self, classifier_with_data):
        """Test visualización de predicciones."""
        clf = classifier_with_data
        clf.load_and_split()
        clf.train(negatives_per_image=5)
        
        if len(clf.test_images) > 0:
            result = clf.visualize_predictions(
                clf.test_images[0],
                stride=32,
                score_threshold=0.0,
            )
            
            assert result.is_color
            assert len(result.metadata.steps) > 0

    def test_evaluate(self, classifier_with_data):
        """Test evaluación en conjunto de test."""
        clf = classifier_with_data
        clf.load_and_split()
        clf.train(negatives_per_image=5)
        
        if len(clf.test_images) > 0:
            metrics = clf.evaluate(negatives_per_image=3)
            
            assert isinstance(metrics, EvaluationMetrics)
            assert 0.0 <= metrics.accuracy <= 1.0
