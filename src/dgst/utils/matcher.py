"""
BT3b: Sistema de matching de señales contra banco de referencias.

Este módulo implementa el matching de imágenes/ROIs detectados contra
un banco de referencias de señales de tráfico usando descriptores locales.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable
import os

import cv2
import numpy as np

from dgst.utils.loader import Image, RegionOfInterest, DataLoader
from dgst.utils.descriptors import (
    DescriptorMethod,
    LocalDescriptorExtractor,
    ExtractionResult,
)


class MatcherType(Enum):
    """Tipo de matcher a usar."""
    BRUTE_FORCE = "brute_force"
    FLANN = "flann"


@dataclass
class MatchResult:
    """Resultado del matching contra una referencia."""
    reference_name: str
    score: float
    num_matches: int
    num_good_matches: int
    
    def __repr__(self) -> str:
        return f"MatchResult({self.reference_name}: {self.score:.1%}, {self.num_good_matches} matches)"


@dataclass
class LocalizationResult:
    """Resultado de localizar un template en una imagen."""
    found: bool
    score: float
    bounding_box: tuple[int, int, int, int] | None  # (x, y, w, h)
    num_matches: int
    
    def __repr__(self) -> str:
        if self.found and self.bounding_box:
            return f"LocalizationResult(found, score={self.score:.1%}, bbox={self.bounding_box})"
        return f"LocalizationResult(not found)"


def extract_roi_image(image: Image, roi: RegionOfInterest) -> Image:
    """Extrae un ROI como imagen recortada."""
    pts = np.array([roi.p1, roi.p2, roi.p3, roi.p4], dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    
    h_img, w_img = image.data.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, w_img - x), min(h, h_img - y)
    
    if w <= 0 or h <= 0:
        raise ValueError("ROI fuera de los límites de la imagen")
    
    return Image(data=image.data[y:y+h, x:x+w].copy(), rois=[])


class SignMatcher:
    """Matcher de señales contra banco de referencias.
    
    Carga referencias desde un directorio con estructura de dataset
    y permite matchear imágenes o ROIs contra las referencias.
    """
    
    def __init__(
        self,
        references_path: str,
        method: DescriptorMethod = DescriptorMethod.SIFT,
        matcher_type: MatcherType = MatcherType.BRUTE_FORCE,
        ratio_threshold: float = 0.75,
        preprocessor: Callable[[Image], Image] | None = None,
    ):
        """Inicializa el matcher y carga las referencias.
        
        Args:
            references_path: Directorio con referencias (misma estructura que dataset).
            method: Método de extracción de descriptores.
            matcher_type: Tipo de matcher (BruteForce o FLANN).
            ratio_threshold: Umbral para ratio test de Lowe.
            preprocessor: Función opcional que preprocesa imágenes antes de extraer
                features. Recibe Image y devuelve Image procesada.
        """
        self._references_path = Path(references_path)
        self._method = method
        self._matcher_type = matcher_type
        self._ratio_threshold = ratio_threshold
        self._preprocessor = preprocessor
        self._extractor = LocalDescriptorExtractor()
        self._references: list[tuple[str, Image, ExtractionResult]] = []
        self._matcher = self._create_matcher()
        
        self._load_references()
    
    def _create_matcher(self) -> cv2.DescriptorMatcher:
        """Crea el matcher según el tipo configurado."""
        if self._method == DescriptorMethod.SIFT:
            if self._matcher_type == MatcherType.FLANN:
                return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            return cv2.BFMatcher(cv2.NORM_L2)
        else:
            if self._matcher_type == MatcherType.FLANN:
                return cv2.FlannBasedMatcher(
                    dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1),
                    dict(checks=50)
                )
            return cv2.BFMatcher(cv2.NORM_HAMMING)
    
    def _load_references(self) -> None:
        """Carga referencias desde el directorio."""
        if not self._references_path.exists():
            raise ValueError(f"References path not found: {self._references_path}")
        
        loader = DataLoader(path=str(self._references_path))
        
        for item in sorted(self._references_path.iterdir()):
            if not item.is_dir() or not item.name.isdigit():
                continue
            
            ref_id = int(item.name)
            try:
                ref_image = loader.load(ref_id)
                
                if ref_image.rois:
                    for i, roi in enumerate(ref_image.rois):
                        try:
                            roi_img = extract_roi_image(ref_image, roi)
                            self._add_reference(f"{ref_id:06d}_{i}", roi_img)
                        except ValueError:
                            continue
                else:
                    self._add_reference(f"{ref_id:06d}", ref_image)
            except Exception:
                continue
    
    def _add_reference(self, name: str, image: Image) -> None:
        """Añade una referencia al banco."""
        processed = self._preprocess(image)
        extraction = self._extractor.extract(processed, self._method)
        if extraction.has_descriptors():
            self._references.append((name, image, extraction))
    
    def _preprocess(self, image: Image) -> Image:
        """Aplica preprocesamiento si está configurado."""
        if self._preprocessor is None:
            return image
        return self._preprocessor(image)
    
    @property
    def num_references(self) -> int:
        return len(self._references)
    
    @property
    def reference_names(self) -> list[str]:
        return [r[0] for r in self._references]
    
    def match(self, image: Image, top_k: int = 3) -> list[MatchResult]:
        """Matchea una imagen contra todas las referencias."""
        if not self._references:
            return []
        
        processed = self._preprocess(image)
        query_ext = self._extractor.extract(processed, self._method)
        if not query_ext.has_descriptors():
            return []
        
        results = []
        for name, _, ref_ext in self._references:
            score, n_matches, n_good = self._compute_match_score(query_ext, ref_ext)
            results.append(MatchResult(name, score, n_matches, n_good))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    def match_roi(self, image: Image, roi: RegionOfInterest, top_k: int = 3) -> list[MatchResult]:
        """Matchea un ROI específico contra las referencias."""
        return self.match(extract_roi_image(image, roi), top_k)
    
    def _compute_match_score(
        self, query: ExtractionResult, ref: ExtractionResult
    ) -> tuple[float, int, int]:
        """Calcula score de matching entre query y referencia."""
        desc1, desc2 = query._descriptors, ref._descriptors
        
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return 0.0, 0, 0
        
        try:
            matches = self._matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return 0.0, 0, 0
        
        # Ratio test con filtro de unicidad
        candidates = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                is_good = m.distance < self._ratio_threshold * n.distance
                candidates.append((m, is_good))
        
        candidates.sort(key=lambda x: x[0].distance)
        
        used_q, used_t = set(), set()
        n_all, n_good = 0, 0
        
        for match, is_good in candidates:
            if match.queryIdx not in used_q and match.trainIdx not in used_t:
                used_q.add(match.queryIdx)
                used_t.add(match.trainIdx)
                n_all += 1
                if is_good:
                    n_good += 1
        
        score = min(1.0, n_good / query.num_keypoints) if query.num_keypoints > 0 else 0.0
        return score, n_all, n_good
    
    def locate_template(
        self,
        template: Image,
        image: Image,
        min_inliers: int = 5,
        ransac_threshold: float = 10.0,
        max_detections: int = 10,
        ratio_threshold: float | None = None,
    ) -> list[LocalizationResult]:
        """Localiza múltiples instancias de un template en una imagen usando RANSAC.
        
        Usa RANSAC iterativo: encuentra una instancia, elimina los inliers usados,
        y repite hasta que no se encuentren más matches válidos.
        
        Args:
            template: Imagen del template a buscar.
            image: Imagen donde buscar.
            min_inliers: Mínimo de inliers para considerar detección válida.
            ransac_threshold: Umbral de reproyección para RANSAC (píxeles).
            max_detections: Máximo de detecciones a encontrar.
            ratio_threshold: Umbral para ratio test de Lowe. Si None, usa el del matcher.
                Valores más altos (0.8-0.9) son más permisivos.
            
        Returns:
            Lista de LocalizationResult con todas las localizaciones encontradas.
        """
        template_processed = self._preprocess(template)
        template_ext = self._extractor.extract(template_processed, self._method)
        if not template_ext.has_descriptors():
            return []
        
        image_processed = self._preprocess(image)
        image_ext = self._extractor.extract(image_processed, self._method)
        if not image_ext.has_descriptors():
            return []
        
        # Usar ratio_threshold pasado o el del matcher
        effective_ratio = ratio_threshold if ratio_threshold is not None else self._ratio_threshold
        
        h_t, w_t = template.data.shape[:2]
        h_img, w_img = image.data.shape[:2]
        
        desc_t = template_ext._descriptors
        desc_i = image_ext._descriptors.copy()
        
        # Mantener track de keypoints usados en la imagen
        used_image_kp = set()
        results: list[LocalizationResult] = []
        
        for _ in range(max_detections):
            # Matching
            try:
                matches = self._matcher.knnMatch(desc_t, desc_i, k=2)
            except cv2.error:
                break
            
            # Ratio test, excluyendo keypoints ya usados
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.trainIdx not in used_image_kp:
                        if m.distance < effective_ratio * n.distance:
                            good_matches.append(m)
            
            if len(good_matches) < 4:
                break
            
            # Obtener puntos correspondientes
            pts_template = np.float32([
                template_ext.keypoints[m.queryIdx].position for m in good_matches
            ]).reshape(-1, 1, 2)
            pts_image = np.float32([
                image_ext.keypoints[m.trainIdx].position for m in good_matches
            ]).reshape(-1, 1, 2)
            
            # Estimar homografía con RANSAC
            try:
                H, mask = cv2.findHomography(pts_template, pts_image, cv2.RANSAC, ransac_threshold)
            except cv2.error:
                break
            
            if H is None or mask is None:
                break
            
            num_inliers = int(np.sum(mask))
            
            if num_inliers < min_inliers:
                break
            
            # Proyectar esquinas del template
            template_corners = np.float32([
                [0, 0], [w_t, 0], [w_t, h_t], [0, h_t]
            ]).reshape(-1, 1, 2)
            
            try:
                projected = cv2.perspectiveTransform(template_corners, H)
            except cv2.error:
                break
            
            # Calcular bounding box
            projected = projected.reshape(-1, 2)
            x_min = int(max(0, np.min(projected[:, 0])))
            y_min = int(max(0, np.min(projected[:, 1])))
            x_max = int(min(w_img, np.max(projected[:, 0])))
            y_max = int(min(h_img, np.max(projected[:, 1])))
            
            if x_max <= x_min or y_max <= y_min:
                break
            
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            score = min(1.0, num_inliers / template_ext.num_keypoints) if template_ext.num_keypoints > 0 else 0.0
            
            results.append(LocalizationResult(True, score, bbox, num_inliers))
            
            # Marcar inliers como usados para la siguiente iteración
            mask_flat = mask.flatten()
            for i, m in enumerate(good_matches):
                if mask_flat[i]:
                    used_image_kp.add(m.trainIdx)
        
        return results
    
    def locate_template_batch(
        self,
        template: Image,
        images: list[Image],
        min_inliers: int = 5,
        ransac_threshold: float = 10.0,
        max_detections: int = 10,
        ratio_threshold: float | None = None,
    ) -> list[list[LocalizationResult]]:
        """Localiza un template en múltiples imágenes.
        
        Returns:
            Lista de listas. Cada sublista contiene las localizaciones para esa imagen.
        """
        return [self.locate_template(template, img, min_inliers, ransac_threshold, max_detections, ratio_threshold) for img in images]
    
    def visualize_match(
        self,
        query: Image,
        reference_name: str,
        max_matches: int = 50,
    ) -> Image:
        """Visualiza matches entre query y una referencia."""
        ref = None
        for name, img, ext in self._references:
            if name == reference_name:
                ref = (img, ext)
                break
        
        if ref is None:
            raise ValueError(f"Reference not found: {reference_name}")
        
        ref_img, ref_ext = ref
        query_ext = self._extractor.extract(query, self._method)
        
        img1 = query.data if query.is_color else cv2.cvtColor(query.data, cv2.COLOR_GRAY2BGR)
        img2 = ref_img.data if ref_img.is_color else cv2.cvtColor(ref_img.data, cv2.COLOR_GRAY2BGR)
        
        if not query_ext.has_descriptors() or not ref_ext.has_descriptors():
            h = max(img1.shape[0], img2.shape[0])
            result = np.zeros((h, img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
            result[:img1.shape[0], :img1.shape[1]] = img1
            result[:img2.shape[0], img1.shape[1]:] = img2
            return Image(data=result, rois=[])
        
        try:
            matches = self._matcher.knnMatch(query_ext._descriptors, ref_ext._descriptors, k=2)
        except cv2.error:
            matches = []
        
        good_matches = []
        for mp in matches:
            if len(mp) == 2 and mp[0].distance < self._ratio_threshold * mp[1].distance:
                good_matches.append(mp[0])
        
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]
        
        kp1 = [kp._cv_kp for kp in query_ext.keypoints]
        kp2 = [kp._cv_kp for kp in ref_ext.keypoints]
        
        result_data = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches, None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        
        result = Image(data=result_data, rois=[])
        result.metadata.add_step({
            "technique": "visualize_match",
            "reference": reference_name,
            "num_matches": len(good_matches),
        })
        return result
    
    def visualize_localization(
        self,
        template: Image,
        image: Image,
        localizations: list[LocalizationResult],
        bbox_color: tuple[int, int, int] = (0, 255, 0),
    ) -> Image:
        """Visualiza localizaciones de un template en una imagen.
        
        Args:
            template: Imagen del template.
            image: Imagen donde se buscó.
            localizations: Lista de resultados de locate_template.
            bbox_color: Color del bounding box.
            
        Returns:
            Imagen con template a la izquierda, imagen con bboxes a la derecha.
        """
        img1 = template.data if template.is_color else cv2.cvtColor(template.data, cv2.COLOR_GRAY2BGR)
        img2 = image.data.copy() if image.is_color else cv2.cvtColor(image.data, cv2.COLOR_GRAY2BGR)
        
        # Dibujar todos los bboxes
        for i, loc in enumerate(localizations):
            if loc.found and loc.bounding_box:
                x, y, w, h = loc.bounding_box
                cv2.rectangle(img2, (x, y), (x + w, y + h), bbox_color, 3)
                label = f"#{i+1} {loc.score:.0%} ({loc.num_matches})"
                cv2.putText(img2, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
        
        # Concatenar imágenes
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        
        result_data = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
        result_data[:h1, :w1] = img1
        result_data[:h2, w1:w1+w2] = img2
        
        result = Image(data=result_data, rois=[])
        result.metadata.add_step({
            "technique": "visualize_localization",
            "num_detections": len([l for l in localizations if l.found]),
        })
        return result
    
    def get_reference_image(self, name: str) -> Image | None:
        """Obtiene la imagen de una referencia por nombre."""
        for n, img, _ in self._references:
            if n == name:
                return img
        return None
    
    def visualize_localization_batch(
        self,
        template: Image,
        images: list[Image],
        localizations: list[list[LocalizationResult]],
        bbox_color: tuple[int, int, int] = (0, 255, 0),
    ) -> list[Image | None]:
        """Visualiza localizaciones de template en múltiples imágenes.
        
        Args:
            template: Imagen del template.
            images: Lista de imágenes donde se buscó.
            localizations: Resultados de locate_template_batch (lista de listas).
            bbox_color: Color del bounding box.
            
        Returns:
            Lista de Image con visualizaciones. None para imágenes sin detecciones.
        """
        if len(images) != len(localizations):
            raise ValueError("images and localizations must have same length")
        
        results: list[Image | None] = []
        
        for image, locs in zip(images, localizations):
            if not locs:  # Lista vacía
                results.append(None)
            else:
                results.append(self.visualize_localization(template, image, locs, bbox_color))
        
        return results
