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
        load_hole: bool = False,
        ratio_threshold: float = 0.75,
        duplicate_threshold: float = 0.7,
        preprocessor: Callable[[Image], Image] | None = None,
    ):
        """Inicializa el matcher y carga las referencias.
        
        Args:
            references_path: Directorio con referencias (misma estructura que dataset).
            method: Método de extracción de descriptores.
            matcher_type: Tipo de matcher (BruteForce o FLANN).
            ratio_threshold: Umbral para ratio test de Lowe.
            duplicate_threshold: Umbral de similitud para rechazar duplicados (0.0-1.0).
                Si una nueva referencia tiene score >= este umbral con una existente,
                no se añade. Usar 1.0 para desactivar.
            preprocessor: Función opcional que preprocesa imágenes antes de extraer
                features. Recibe Image y devuelve Image procesada.
        """
        self._references_path = Path(references_path)
        self._method = method
        self._matcher_type = matcher_type
        self._ratio_threshold = ratio_threshold
        self._duplicate_threshold = duplicate_threshold
        self._preprocessor = preprocessor
        self._extractor = LocalDescriptorExtractor()
        self._references: list[tuple[str, Image, ExtractionResult]] = []
        self._load_hole = load_hole
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
                
                if self._load_hole:
                    # Usar la imagen completa como referencia
                    self._add_reference(f"{ref_id:06d}", ref_image)
                elif ref_image.rois:
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
        """Añade una referencia al banco si no es muy similar a una existente."""
        # No preprocesar referencias - usar imagen original
        extraction = self._extractor.extract(image, self._method)
        if not extraction.has_descriptors():
            return
        
        # Verificar si ya existe una referencia muy similar
        if self._duplicate_threshold < 1.0:
            for _, _, ref_ext in self._references:
                score, _, _ = self._compute_match_score(extraction, ref_ext)
                if score >= self._duplicate_threshold:
                    return  # Duplicado detectado, no añadir
        
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
        min_template_size: int = 100,
    ) -> list[LocalizationResult]:
        """Localiza múltiples instancias de un template en una imagen usando RANSAC.
        
        Usa RANSAC iterativo: encuentra una instancia, elimina los inliers usados,
        y repite hasta que no se encuentren más matches válidos.
        
        Si el template es pequeño (menor que min_template_size en cualquier dimensión),
        se escala automáticamente para mejorar la detección de keypoints.
        
        Args:
            template: Imagen del template a buscar.
            image: Imagen donde buscar.
            min_inliers: Mínimo de inliers para considerar detección válida.
            ransac_threshold: Umbral de reproyección para RANSAC (píxeles).
            max_detections: Máximo de detecciones a encontrar.
            ratio_threshold: Umbral para ratio test de Lowe. Si None, usa el del matcher.
                Valores más altos (0.8-0.9) son más permisivos.
            min_template_size: Tamaño mínimo del template. Si el template es menor,
                se escala automáticamente para mejorar la extracción de features.
            
        Returns:
            Lista de LocalizationResult con todas las localizaciones encontradas.
        """
        h_t_orig, w_t_orig = template.data.shape[:2]
        h_img, w_img = image.data.shape[:2]
        
        # Escalar template si es muy pequeño para obtener mejores keypoints
        scale_factor = 1.0
        if min(h_t_orig, w_t_orig) < min_template_size:
            scale_factor = min_template_size / min(h_t_orig, w_t_orig)
            new_w = int(w_t_orig * scale_factor)
            new_h = int(h_t_orig * scale_factor)
            scaled_data = cv2.resize(template.data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            template_scaled = Image(data=scaled_data, rois=[])
        else:
            template_scaled = template
        
        h_t_scaled, w_t_scaled = template_scaled.data.shape[:2]
        
        template_processed = self._preprocess(template_scaled)
        template_ext = self._extractor.extract(template_processed, self._method)
        if not template_ext.has_descriptors():
            return []
        
        image_processed = self._preprocess(image)
        image_ext = self._extractor.extract(image_processed, self._method)
        if not image_ext.has_descriptors():
            return []
        
        # Usar ratio_threshold pasado o el del matcher
        effective_ratio = ratio_threshold if ratio_threshold is not None else self._ratio_threshold
        
        desc_t = template_ext._descriptors
        desc_i = image_ext._descriptors
        
        # Mantener track de keypoints usados en la imagen
        used_image_kp = set()
        results: list[LocalizationResult] = []
        
        for _ in range(max_detections):
            # Matching: template -> scene (estándar para localización)
            try:
                matches = self._matcher.knnMatch(desc_t, desc_i, k=2)
            except cv2.error:
                break
            
            # Ratio test, excluyendo keypoints ya usados
            # m.queryIdx = índice en template, m.trainIdx = índice en imagen
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
            # Los keypoints del template están en coordenadas escaladas,
            # los de la imagen están en coordenadas originales
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
            
            # Proyectar esquinas del template (en coordenadas escaladas)
            template_corners = np.float32([
                [0, 0], [w_t_scaled, 0], [w_t_scaled, h_t_scaled], [0, h_t_scaled]
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
    
    def locate_template_sliding(
        self,
        template: Image,
        image: Image,
        window_scale: float = 2.0,
        overlap: float = 0.5,
        top_k_candidates: int = 5,
        min_inliers: int = 5,
        ransac_threshold: float = 10.0,
        ratio_threshold: float | None = None,
        refine_margin: float = 0.5,
    ) -> list[LocalizationResult]:
        """Localiza un template pequeño en una imagen grande usando ventana deslizante.
        
        Estrategia coarse-to-fine:
        1. Divide la imagen en patches con tamaño proporcional al template
        2. Hace matching rápido para encontrar regiones candidatas
        3. Refina la búsqueda en las regiones más prometedoras
        
        Args:
            template: Imagen del template a buscar.
            image: Imagen donde buscar.
            window_scale: Factor de escala para el tamaño de ventana respecto al template.
                Por ejemplo, 2.0 significa ventanas de 2x el tamaño del template.
            overlap: Proporción de solapamiento entre ventanas (0.0-0.9).
            top_k_candidates: Número de regiones candidatas a refinar.
            min_inliers: Mínimo de inliers para considerar detección válida.
            ransac_threshold: Umbral de reproyección para RANSAC.
            ratio_threshold: Umbral para ratio test. Si None, usa el del matcher.
            refine_margin: Margen adicional alrededor del candidato para refinamiento
                (como proporción del tamaño de ventana).
            
        Returns:
            Lista de LocalizationResult ordenadas por score.
        """
        h_t, w_t = template.data.shape[:2]
        h_img, w_img = image.data.shape[:2]
        
        # Calcular tamaño de ventana
        window_h = int(h_t * window_scale)
        window_w = int(w_t * window_scale)
        
        # Si la ventana es muy grande respecto a la imagen, usar método directo
        if window_h >= h_img * 0.8 or window_w >= w_img * 0.8:
            return self.locate_template(
                template, image, min_inliers, ransac_threshold, 
                max_detections=1, ratio_threshold=ratio_threshold
            )
        
        # Extraer features del template una sola vez
        template_processed = self._preprocess(template)
        template_ext = self._extractor.extract(template_processed, self._method)
        if not template_ext.has_descriptors():
            return []
        
        effective_ratio = ratio_threshold if ratio_threshold is not None else self._ratio_threshold
        
        # Calcular paso de la ventana deslizante
        step_h = max(1, int(window_h * (1 - overlap)))
        step_w = max(1, int(window_w * (1 - overlap)))
        
        # Fase 1: Búsqueda gruesa - encontrar candidatos prometedores
        candidates: list[tuple[float, int, int, int, int]] = []  # (score, x, y, w, h)
        
        for y in range(0, h_img - window_h + 1, step_h):
            for x in range(0, w_img - window_w + 1, step_w):
                # Extraer patch
                patch_data = image.data[y:y+window_h, x:x+window_w]
                patch = Image(data=patch_data, rois=[])
                
                # Calcular score de matching rápido
                patch_processed = self._preprocess(patch)
                patch_ext = self._extractor.extract(patch_processed, self._method)
                
                if not patch_ext.has_descriptors():
                    continue
                
                # Matching rápido
                try:
                    matches = self._matcher.knnMatch(
                        template_ext._descriptors, 
                        patch_ext._descriptors, 
                        k=2
                    )
                except cv2.error:
                    continue
                
                # Contar buenos matches
                good_count = 0
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < effective_ratio * n.distance:
                            good_count += 1
                
                if good_count >= 4:  # Mínimo para homografía
                    score = good_count / max(1, template_ext.num_keypoints)
                    candidates.append((score, x, y, window_w, window_h))
        
        if not candidates:
            return []
        
        # Ordenar por score y tomar los mejores
        candidates.sort(reverse=True, key=lambda c: c[0])
        top_candidates = candidates[:top_k_candidates]
        
        # Fase 2: Refinamiento - buscar con más precisión en cada candidato
        results: list[LocalizationResult] = []
        used_regions: list[tuple[int, int, int, int]] = []
        
        for _, cx, cy, cw, ch in top_candidates:
            # Expandir región para refinamiento
            margin_x = int(cw * refine_margin)
            margin_y = int(ch * refine_margin)
            
            rx = max(0, cx - margin_x)
            ry = max(0, cy - margin_y)
            rw = min(w_img - rx, cw + 2 * margin_x)
            rh = min(h_img - ry, ch + 2 * margin_y)
            
            # Verificar que no se solape demasiado con regiones ya procesadas
            overlaps = False
            for ux, uy, uw, uh in used_regions:
                # Calcular IoU aproximado
                ix = max(rx, ux)
                iy = max(ry, uy)
                ix2 = min(rx + rw, ux + uw)
                iy2 = min(ry + rh, uy + uh)
                if ix < ix2 and iy < iy2:
                    inter_area = (ix2 - ix) * (iy2 - iy)
                    union_area = rw * rh + uw * uh - inter_area
                    if inter_area / union_area > 0.5:
                        overlaps = True
                        break
            
            if overlaps:
                continue
            
            # Extraer región para refinamiento
            region_data = image.data[ry:ry+rh, rx:rx+rw]
            region = Image(data=region_data, rois=[])
            
            # Localizar en la región
            region_results = self.locate_template(
                template, region, min_inliers, ransac_threshold,
                max_detections=1, ratio_threshold=ratio_threshold
            )
            
            for loc in region_results:
                if loc.found and loc.bounding_box:
                    # Ajustar coordenadas al sistema de la imagen completa
                    bx, by, bw, bh = loc.bounding_box
                    adjusted_bbox = (bx + rx, by + ry, bw, bh)
                    results.append(LocalizationResult(
                        found=True,
                        score=loc.score,
                        bounding_box=adjusted_bbox,
                        num_matches=loc.num_matches
                    ))
                    used_regions.append(adjusted_bbox)
        
        # Ordenar por score
        results.sort(key=lambda r: r.score, reverse=True)
        return results
    
    def locate_template_batch(
        self,
        template: Image,
        images: list[Image],
        min_inliers: int = 5,
        ransac_threshold: float = 10.0,
        max_detections: int = 10,
        ratio_threshold: float | None = None,
        min_template_size: int = 100,
    ) -> list[list[LocalizationResult]]:
        """Localiza un template en múltiples imágenes.
        
        Returns:
            Lista de listas. Cada sublista contiene las localizaciones para esa imagen.
        """
        return [self.locate_template(template, img, min_inliers, ransac_threshold, max_detections, ratio_threshold, min_template_size) for img in images]
    
    def locate_template_sliding_batch(
        self,
        template: Image,
        images: list[Image],
        window_scale: float = 2.0,
        overlap: float = 0.5,
        top_k_candidates: int = 5,
        min_inliers: int = 5,
        ransac_threshold: float = 10.0,
        ratio_threshold: float | None = None,
        refine_margin: float = 0.5,
    ) -> list[list[LocalizationResult]]:
        """Localiza un template en múltiples imágenes usando ventana deslizante.
        
        Returns:
            Lista de listas. Cada sublista contiene las localizaciones para esa imagen.
        """
        return [
            self.locate_template_sliding(
                template, img, window_scale, overlap, top_k_candidates,
                min_inliers, ransac_threshold, ratio_threshold, refine_margin
            ) 
            for img in images
        ]
    
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
        query_processed = self._preprocess(query)
        query_ext = self._extractor.extract(query_processed, self._method)
        
        img1, img2 = self._prepare_images_for_viz(query, ref_img)
        
        if not query_ext.has_descriptors() or not ref_ext.has_descriptors():
            return self._concat_images(img1, img2)
        
        good_matches = self._get_good_matches(
            query_ext._descriptors, ref_ext._descriptors, 
            self._ratio_threshold, max_matches
        )
        
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
    
    def visualize_template_match(
        self,
        template: Image,
        image: Image,
        max_matches: int = 100,
        ratio_threshold: float | None = None,
        draw_homography: bool = True,
        min_template_size: int = 100,
    ) -> Image:
        """Visualiza matches entre un template pequeño y una imagen de escena.
        
        Args:
            template: Imagen del template (pequeña).
            image: Imagen de escena (grande).
            max_matches: Máximo de matches a dibujar.
            ratio_threshold: Umbral para ratio test. Si None, usa el del matcher.
            draw_homography: Si True, dibuja el contorno proyectado del template.
            min_template_size: Tamaño mínimo para escalar el template.
            
        Returns:
            Imagen con template a la izquierda y escena con matches a la derecha.
        """
        template_scaled, scale_factor = self._scale_template(template, min_template_size)
        h_t_scaled, w_t_scaled = template_scaled.data.shape[:2]
        
        template_ext = self._extractor.extract(self._preprocess(template_scaled), self._method)
        image_ext = self._extractor.extract(self._preprocess(image), self._method)
        
        img1, img2 = self._prepare_images_for_viz(template_scaled, image)
        
        if not template_ext.has_descriptors() or not image_ext.has_descriptors():
            return self._concat_images(img1, img2)
        
        effective_ratio = ratio_threshold if ratio_threshold is not None else self._ratio_threshold
        good_matches = self._get_good_matches(
            template_ext._descriptors, image_ext._descriptors,
            effective_ratio, max_matches
        )
        
        # Dibujar homografía si hay suficientes matches
        H = None
        if draw_homography and len(good_matches) >= 4:
            H = self._draw_homography_on_image(
                template_ext, image_ext, good_matches,
                w_t_scaled, h_t_scaled, img2
            )
        
        kp1 = [kp._cv_kp for kp in template_ext.keypoints]
        kp2 = [kp._cv_kp for kp in image_ext.keypoints]
        
        result_data = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches, None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        
        result = Image(data=result_data, rois=[])
        result.metadata.add_step({
            "technique": "visualize_template_match",
            "num_matches": len(good_matches),
            "homography_found": H is not None,
            "scale_factor": scale_factor,
        })
        return result
    
    def visualize_template_match_batch(
        self,
        template: Image,
        images: list[Image],
        max_matches: int = 100,
        ratio_threshold: float | None = None,
        draw_homography: bool = True,
        min_template_size: int = 100,
    ) -> list[Image]:
        """Visualiza matches de un template contra múltiples imágenes.
        
        Returns:
            Lista de imágenes con visualizaciones de matches.
        """
        return [
            self.visualize_template_match(
                template, img, max_matches, ratio_threshold,
                draw_homography, min_template_size
            )
            for img in images
        ]
    
    def visualize_localization(
        self,
        template: Image,
        image: Image,
        localizations: list[LocalizationResult],
        bbox_color: tuple[int, int, int] = (0, 255, 0),
    ) -> Image:
        """Visualiza localizaciones de un template en una imagen."""
        img1, img2 = self._prepare_images_for_viz(template, image)
        
        for i, loc in enumerate(localizations):
            if loc.found and loc.bounding_box:
                x, y, w, h = loc.bounding_box
                cv2.rectangle(img2, (x, y), (x + w, y + h), bbox_color, 3)
                label = f"#{i+1} {loc.score:.0%} ({loc.num_matches})"
                cv2.putText(img2, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
        
        result = self._concat_images(img1, img2)
        result.metadata.add_step({
            "technique": "visualize_localization",
            "num_detections": len([l for l in localizations if l.found]),
        })
        return result
    
    def visualize_localization_batch(
        self,
        template: Image,
        images: list[Image],
        localizations: list[list[LocalizationResult]],
        bbox_color: tuple[int, int, int] = (0, 255, 0),
    ) -> list[Image | None]:
        """Visualiza localizaciones de template en múltiples imágenes."""
        if len(images) != len(localizations):
            raise ValueError("images and localizations must have same length")
        
        return [
            self.visualize_localization(template, img, locs, bbox_color) if locs else None
            for img, locs in zip(images, localizations)
        ]
    
    def get_reference_image(self, name: str) -> Image | None:
        """Obtiene la imagen de una referencia por nombre."""
        for n, img, _ in self._references:
            if n == name:
                return img
        return None
    
    # -------------------------------------------------------------------------
    # Métodos auxiliares privados
    # -------------------------------------------------------------------------
    
    def _scale_template(
        self, template: Image, min_size: int
    ) -> tuple[Image, float]:
        """Escala un template si es menor que min_size."""
        h, w = template.data.shape[:2]
        if min(h, w) >= min_size:
            return template, 1.0
        
        scale = min_size / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = cv2.resize(template.data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return Image(data=scaled, rois=[]), scale
    
    def _prepare_images_for_viz(
        self, img1: Image, img2: Image
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepara imágenes para visualización (convierte a BGR si es necesario)."""
        out1 = img1.data if img1.is_color else cv2.cvtColor(img1.data, cv2.COLOR_GRAY2BGR)
        out2 = img2.data.copy() if img2.is_color else cv2.cvtColor(img2.data, cv2.COLOR_GRAY2BGR)
        return out1, out2
    
    def _concat_images(self, img1: np.ndarray, img2: np.ndarray) -> Image:
        """Concatena dos imágenes horizontalmente."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        result = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
        result[:h1, :w1] = img1
        result[:h2, w1:w1+w2] = img2
        return Image(data=result, rois=[])
    
    def _get_good_matches(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float,
        max_matches: int | None = None,
    ) -> list:
        """Obtiene buenos matches aplicando ratio test."""
        try:
            matches = self._matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []
        
        good = []
        for mp in matches:
            if len(mp) == 2 and mp[0].distance < ratio_threshold * mp[1].distance:
                good.append(mp[0])
        
        good.sort(key=lambda x: x.distance)
        return good[:max_matches] if max_matches else good
    
    def _draw_homography_on_image(
        self,
        template_ext: ExtractionResult,
        image_ext: ExtractionResult,
        matches: list,
        w_template: int,
        h_template: int,
        img: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 255),
    ) -> np.ndarray | None:
        """Dibuja el contorno proyectado del template en la imagen."""
        pts_t = np.float32([
            template_ext.keypoints[m.queryIdx].position for m in matches
        ]).reshape(-1, 1, 2)
        pts_i = np.float32([
            image_ext.keypoints[m.trainIdx].position for m in matches
        ]).reshape(-1, 1, 2)
        
        try:
            H, _ = cv2.findHomography(pts_t, pts_i, cv2.RANSAC, 10.0)
            if H is not None:
                corners = np.float32([
                    [0, 0], [w_template, 0], [w_template, h_template], [0, h_template]
                ]).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(corners, H)
                cv2.polylines(img, [projected.reshape(-1, 2).astype(np.int32)], True, color, 3)
                return H
        except cv2.error:
            pass
        return None
