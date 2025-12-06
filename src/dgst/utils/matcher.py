"""
BT3b: Sistema de matching de señales contra banco de referencias.

Este módulo implementa el matching de imágenes/ROIs detectados contra
un banco de referencias de señales de tráfico usando descriptores locales.
Las referencias usan la misma estructura que el dataset principal.
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
class ReferenceSign:
    """Señal de referencia con descriptores pre-calculados."""
    
    name: str
    image: Image
    extraction: ExtractionResult
    source_id: int | None = None  # ID del directorio origen si aplica


@dataclass
class MatchResult:
    """Resultado del matching contra una referencia."""
    
    reference_name: str
    score: float
    num_matches: int
    num_good_matches: int
    
    def __repr__(self) -> str:
        return f"MatchResult({self.reference_name}: {self.score:.1%}, {self.num_good_matches} matches)"


def extract_roi_image(image: Image, roi: RegionOfInterest) -> Image:
    """Extrae un ROI como imagen recortada.
    
    Args:
        image: Imagen completa.
        roi: Región de interés.
        
    Returns:
        Imagen recortada del ROI.
    """
    pts = np.array([roi.p1, roi.p2, roi.p3, roi.p4], dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    
    h_img, w_img = image.data.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    
    if w <= 0 or h <= 0:
        raise ValueError("ROI fuera de los límites de la imagen")
    
    roi_data = image.data[y:y+h, x:x+w].copy()
    return Image(data=roi_data, rois=[])


@dataclass
class LocalizationResult:
    """Resultado de localizar un ROI en una imagen."""
    
    image_index: int
    score: float
    num_inliers: int
    bounding_box: tuple[int, int, int, int] | None  # (x, y, w, h)
    homography: np.ndarray | None  # Matriz de homografía 3x3
    
    def __repr__(self) -> str:
        bbox_str = f"({self.bounding_box[0]}, {self.bounding_box[1]})" if self.bounding_box else "None"
        return f"LocalizationResult(img={self.image_index}, score={self.score:.1%}, pos={bbox_str})"



class SignMatcher:
    """Matcher de señales contra banco de referencias.
    
    Las referencias se cargan desde un directorio que sigue la misma
    estructura que el dataset principal (con DataLoader).
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
            references_path: Ruta al directorio con referencias (misma estructura que dataset).
            method: Método de extracción de descriptores.
            matcher_type: Tipo de matcher (BruteForce o FLANN).
            ratio_threshold: Umbral para ratio test de Lowe.
            preprocessor: Función opcional que preprocesa imágenes antes de extraer
                features. Recibe una Image y devuelve una Image procesada.
                Útil para destacar características relevantes (ej: realzar bordes,
                normalizar iluminación, etc.).
        """
        self._references_path = Path(references_path)
        self._method = method
        self._matcher_type = matcher_type
        self._ratio_threshold = ratio_threshold
        self._preprocessor = preprocessor
        self._extractor = LocalDescriptorExtractor()
        self._references: list[ReferenceSign] = []
        self._matcher = self._create_matcher()
        self._loader = DataLoader(path=str(self._references_path))
        
        # Cargar referencias automáticamente
        self._load_references()
    
    def _create_matcher(self) -> cv2.DescriptorMatcher:
        """Crea el matcher según el tipo configurado."""
        if self._method == DescriptorMethod.SIFT:
            if self._matcher_type == MatcherType.FLANN:
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                return cv2.FlannBasedMatcher(index_params, search_params)
            else:
                return cv2.BFMatcher(cv2.NORM_L2)
        else:
            if self._matcher_type == MatcherType.FLANN:
                index_params = dict(
                    algorithm=6,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1,
                )
                search_params = dict(checks=50)
                return cv2.FlannBasedMatcher(index_params, search_params)
            else:
                return cv2.BFMatcher(cv2.NORM_HAMMING)
    
    def _load_references(self) -> None:
        """Carga referencias desde el directorio usando la estructura del dataset."""
        if not self._references_path.exists():
            raise ValueError(f"References path not found: {self._references_path}")
        
        # Buscar todos los subdirectorios numéricos (como en el dataset)
        ref_dirs = []
        for item in sorted(self._references_path.iterdir()):
            if item.is_dir() and item.name.isdigit():
                ref_dirs.append(int(item.name))
        
        for ref_id in ref_dirs:
            try:
                # Usar DataLoader para cargar imagen con ROIs
                ref_image = self._loader.load(ref_id)
                
                if ref_image.rois:
                    # Si tiene ROIs, extraer cada uno como referencia separada
                    for i, roi in enumerate(ref_image.rois):
                        try:
                            roi_image = extract_roi_image(ref_image, roi)
                            name = f"{ref_id:06d}_{i}"
                            self._add_reference(name, roi_image, source_id=ref_id)
                        except ValueError:
                            continue
                else:
                    # Si no tiene ROIs, usar la imagen completa
                    name = f"{ref_id:06d}"
                    self._add_reference(name, ref_image, source_id=ref_id)
                    
            except Exception as e:
                # Silenciosamente ignorar referencias que no se pueden cargar
                continue
    
    def _add_reference(self, name: str, image: Image, source_id: int | None = None) -> None:
        """Añade una referencia al banco (interno).
        
        Evita añadir duplicados comparando descriptores.
        """
        processed = self._preprocess(image)
        extraction = self._extractor.extract(processed, self._method)
        
        if not extraction.has_descriptors():
            return
        
        # Verificar si ya existe una referencia con descriptores muy similares
        new_desc = extraction._descriptors
        for existing_ref in self._references:
            existing_desc = existing_ref.extraction._descriptors
            
            # Comparar tamaño de descriptores
            if new_desc.shape != existing_desc.shape:
                continue
            
            # Calcular similitud (correlación)
            try:
                if self._method == DescriptorMethod.SIFT:
                    # Para descriptores flotantes, usar correlación
                    new_flat = new_desc.flatten()
                    exist_flat = existing_desc.flatten()
                    corr = np.corrcoef(new_flat, exist_flat)[0, 1]
                    if corr > 0.95:
                        return  # Duplicado, no añadir
                else:
                    # Para descriptores binarios, usar distancia de Hamming normalizada
                    diff = np.mean(new_desc != existing_desc)
                    if diff < 0.05:  # Menos del 5% de diferencia
                        return  # Duplicado, no añadir
            except Exception:
                continue
        
        ref = ReferenceSign(
            name=name,
            image=image,
            extraction=extraction,
            source_id=source_id,
        )
        self._references.append(ref)

    
    def _preprocess(self, image: Image) -> Image:
        """Aplica preprocesamiento si está configurado.
        
        Args:
            image: Imagen a procesar.
            
        Returns:
            Imagen procesada (o la original si no hay preprocessor).
        """
        if self._preprocessor is None:
            return image
        return self._preprocessor(image)
    
    @property
    def method(self) -> DescriptorMethod:
        return self._method
    
    @property
    def num_references(self) -> int:
        return len(self._references)
    
    @property
    def reference_names(self) -> list[str]:
        return [r.name for r in self._references]
    
    def match(self, image: Image, top_k: int = 3) -> list[MatchResult]:
        """Matchea una imagen contra todas las referencias.
        
        Args:
            image: Imagen a matchear.
            top_k: Número de mejores resultados a devolver.
            
        Returns:
            Lista de MatchResult ordenados por score descendente.
        """
        if not self._references:
            return []
        
        processed = self._preprocess(image)
        query_extraction = self._extractor.extract(processed, self._method)
        
        if not query_extraction.has_descriptors():
            return []
        
        results = []
        for ref in self._references:
            score, num_matches, num_good = self._compute_match_score(
                query_extraction, ref.extraction
            )
            
            results.append(MatchResult(
                reference_name=ref.name,
                score=score,
                num_matches=num_matches,
                num_good_matches=num_good,
            ))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    def match_roi(
        self,
        image: Image,
        roi: RegionOfInterest,
        top_k: int = 3,
    ) -> list[MatchResult]:
        """Matchea un ROI específico contra las referencias.
        
        Args:
            image: Imagen completa.
            roi: Región de interés a extraer y matchear.
            top_k: Número de mejores resultados.
            
        Returns:
            Lista de MatchResult ordenados por score.
        """
        roi_image = extract_roi_image(image, roi)
        return self.match(roi_image, top_k)
    
    def _compute_match_score(
        self,
        query: ExtractionResult,
        reference: ExtractionResult,
    ) -> tuple[float, int, int]:
        """Calcula score de matching entre query y referencia.
        
        Retorna matches únicos: cada keypoint matchea con a lo sumo
        un keypoint de la otra imagen.
        """
        desc1 = query._descriptors
        desc2 = reference._descriptors
        
        if desc1 is None or desc2 is None:
            return 0.0, 0, 0
        
        if len(desc1) < 2 or len(desc2) < 2:
            return 0.0, 0, 0
        
        try:
            matches = self._matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return 0.0, 0, 0
        
        # Aplicar ratio test y recopilar candidatos
        candidates = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                is_good = m.distance < self._ratio_threshold * n.distance
                candidates.append((m, is_good))
            elif len(match_pair) == 1:
                candidates.append((match_pair[0], False))
        
        # Filtrar para obtener matches únicos por queryIdx y trainIdx
        # Ordenar por distancia para quedarnos con los mejores
        candidates.sort(key=lambda x: x[0].distance)
        
        used_query = set()  # keypoints usados de query
        used_train = set()  # keypoints usados de reference
        
        unique_all = []
        unique_good = []
        
        for match, is_good in candidates:
            if match.queryIdx not in used_query and match.trainIdx not in used_train:
                used_query.add(match.queryIdx)
                used_train.add(match.trainIdx)
                unique_all.append(match)
                if is_good:
                    unique_good.append(match)
        
        num_matches = len(unique_all)
        num_good = len(unique_good)
        
        if query.num_keypoints > 0:
            score = num_good / query.num_keypoints
        else:
            score = 0.0
        
        score = min(1.0, score)
        
        return score, num_matches, num_good
    
    def visualize_match(
        self,
        query: Image,
        reference_name: str,
        max_matches: int = 50,
    ) -> Image:
        """Visualiza matches entre query y una referencia específica.
        
        Args:
            query: Imagen query.
            reference_name: Nombre de la referencia.
            max_matches: Máximo de matches a dibujar.
            
        Returns:
            Imagen con matches dibujados.
        """
        ref = None
        for r in self._references:
            if r.name == reference_name:
                ref = r
                break
        
        if ref is None:
            raise ValueError(f"Reference not found: {reference_name}")
        
        processed_query = self._preprocess(query)
        query_extraction = self._extractor.extract(processed_query, self._method)
        
        if not query_extraction.has_descriptors() or not ref.extraction.has_descriptors():
            h1, w1 = query.data.shape[:2]
            h2, w2 = ref.image.data.shape[:2]
            h = max(h1, h2)
            result = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
            q_img = query.data if query.is_color else cv2.cvtColor(query.data, cv2.COLOR_GRAY2BGR)
            r_img = ref.image.data if ref.image.is_color else cv2.cvtColor(ref.image.data, cv2.COLOR_GRAY2BGR)
            result[:h1, :w1] = q_img
            result[:h2, w1:w1+w2] = r_img
            return Image(data=result, rois=[])
        
        desc1 = query_extraction._descriptors
        desc2 = ref.extraction._descriptors
        
        try:
            matches = self._matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            matches = []
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self._ratio_threshold * n.distance:
                    good_matches.append(m)
        
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]
        
        kp1 = [kp._cv_kp for kp in query_extraction.keypoints]
        kp2 = [kp._cv_kp for kp in ref.extraction.keypoints]
        
        img1 = query.data if query.is_color else cv2.cvtColor(query.data, cv2.COLOR_GRAY2BGR)
        img2 = ref.image.data if ref.image.is_color else cv2.cvtColor(ref.image.data, cv2.COLOR_GRAY2BGR)
        
        result_data = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        
        result = Image(data=result_data, rois=[])
        result.metadata.add_step({
            "technique": "visualize_match",
            "reference": reference_name,
            "num_matches": len(good_matches),
        })
        
        return result
    
    def get_reference_image(self, name: str) -> Image | None:
        """Obtiene la imagen de una referencia por nombre."""
        for r in self._references:
            if r.name == name:
                return r.image
        return None
    
    def locate_template(
        self,
        template: Image,
        images: list[Image],
        min_inliers: int = 6,
        ransac_threshold: float = 10.0,
        ratio_threshold: float | None = None,
        use_affine: bool = True,
    ) -> list[LocalizationResult | None]:
        """Busca un template/ROI en una lista de imágenes.
        
        Encuentra dónde aparece el template en cada imagen usando matching
        de descriptores y verificación geométrica con RANSAC.
        Realiza búsqueda multi-escala y multi-rotación para mayor robustez.
        
        Args:
            template: Imagen del ROI/señal a buscar.
            images: Lista de imágenes donde buscar.
            min_inliers: Mínimo de inliers para considerar un match válido.
            ransac_threshold: Umbral de reproyección para RANSAC (píxeles).
            ratio_threshold: Umbral para ratio test (None usa el del matcher).
                Valores más altos (ej: 0.85) son más permisivos.
            use_affine: Si True, usa transformación afín parcial.
            scales: Lista de factores de escala a probar.
                Default: [0.5, 0.75, 1.0, 1.5, 2.0]
            rotations: Lista de ángulos en grados a probar.
                Default: [-15, -10, -5, 0, 5, 10, 15]
            
        Returns:
            Lista de LocalizationResult del mismo tamaño que images.
            Contiene None para imágenes donde no se encontró el template.
        """
        
        effective_ratio = ratio_threshold if ratio_threshold is not None else self._ratio_threshold
        
        # Preprocesar template base
        template_processed = self._preprocess(template)
        
        # Generar versiones transformadas del template
        template_variants = []
        h_orig, w_orig = template_processed.data.shape[:2]
        
        for scale in [1.0]:
            for angle in [0]:
                # Resize
                new_w = max(1, int(w_orig * scale))
                new_h = max(1, int(h_orig * scale))
                resized = cv2.resize(template_processed.data, (new_w, new_h))
                
                # Rotate
                if angle != 0:
                    center = (new_w // 2, new_h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    
                    # Calcular nuevo tamaño para no recortar
                    cos = abs(M[0, 0])
                    sin = abs(M[0, 1])
                    new_w_rot = int(new_h * sin + new_w * cos)
                    new_h_rot = int(new_h * cos + new_w * sin)
                    
                    M[0, 2] += (new_w_rot - new_w) / 2
                    M[1, 2] += (new_h_rot - new_h) / 2
                    
                    rotated = cv2.warpAffine(resized, M, (new_w_rot, new_h_rot))
                else:
                    rotated = resized
                    M = np.eye(2, 3, dtype=np.float32)
                
                variant_img = Image(data=rotated, rois=[])
                extraction = self._extractor.extract(variant_img, self._method)
                
                if extraction.has_descriptors():
                    template_variants.append({
                        'extraction': extraction,
                        'scale': scale,
                        'angle': angle,
                        'h': rotated.shape[0],
                        'w': rotated.shape[1],
                    })
        
        if not template_variants:
            return [None] * len(images)
        
        results: list[LocalizationResult | None] = [None] * len(images)
        
        for idx, image in enumerate(images):
            image_processed = self._preprocess(image)
            image_extraction = self._extractor.extract(image_processed, self._method)
            
            if not image_extraction.has_descriptors():
                continue
            
            best_result = None
            best_inliers = 0
            
            # Probar todas las variantes del template
            for variant in template_variants:
                template_extraction = variant['extraction']
                
                desc_template = template_extraction._descriptors
                desc_image = image_extraction._descriptors
                
                try:
                    matches = self._matcher.knnMatch(desc_template, desc_image, k=2)
                except cv2.error:
                    continue
                
                # Ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < effective_ratio * n.distance:
                            good_matches.append(m)
                    elif len(match_pair) == 1 and effective_ratio >= 0.9:
                        good_matches.append(match_pair[0])
                
                if len(good_matches) < 4:
                    continue
                
                kp_template = template_extraction.keypoints
                kp_image = image_extraction.keypoints
                
                pts_template = np.float32([
                    kp_template[m.queryIdx].position for m in good_matches
                ]).reshape(-1, 1, 2)
                pts_image = np.float32([
                    kp_image[m.trainIdx].position for m in good_matches
                ]).reshape(-1, 1, 2)
                
                try:
                    if use_affine:
                        M, mask = cv2.estimateAffinePartial2D(
                            pts_template, pts_image,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=ransac_threshold,
                        )
                        if M is not None:
                            H = np.vstack([M, [0, 0, 1]])
                        else:
                            continue
                    else:
                        H, mask = cv2.findHomography(
                            pts_template, pts_image, cv2.RANSAC, ransac_threshold
                        )
                except cv2.error:
                    continue
                
                if H is None or mask is None:
                    continue
                
                num_inliers = int(np.sum(mask))
                
                if num_inliers >= min_inliers and num_inliers > best_inliers:
                    # Calcular bounding box
                    h_v, w_v = variant['h'], variant['w']
                    template_corners = np.float32([
                        [0, 0], [w_v, 0], [w_v, h_v], [0, h_v]
                    ])
                    
                    bbox = None
                    try:
                        projected_corners = []
                        for pt in template_corners:
                            x, y = pt
                            w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
                            if abs(w) < 1e-10:
                                break
                            xp = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
                            yp = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
                            projected_corners.append([xp, yp])
                        
                        if len(projected_corners) == 4:
                            corners = np.array(projected_corners)
                            x_min = int(np.min(corners[:, 0]))
                            y_min = int(np.min(corners[:, 1]))
                            x_max = int(np.max(corners[:, 0]))
                            y_max = int(np.max(corners[:, 1]))
                            
                            h_img, w_img = image.data.shape[:2]
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            x_max = min(w_img, x_max)
                            y_max = min(h_img, y_max)
                            
                            if x_max > x_min and y_max > y_min:
                                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    except Exception:
                        pass
                    
                    # Score normalizado
                    score = min(1.0, num_inliers / max(1, template_extraction.num_keypoints))
                    
                    best_inliers = num_inliers
                    best_result = LocalizationResult(
                        image_index=idx,
                        score=score,
                        num_inliers=num_inliers,
                        bounding_box=bbox,
                        homography=H,
                    )
            
            results[idx] = best_result
        
        return results



    
    def visualize_localization(
        self,
        template: Image,
        images: list[Image],
        localizations: list[LocalizationResult | None],
        max_matches: int = 50,
        draw_bbox: bool = True,
        bbox_color: tuple[int, int, int] = (0, 255, 0),
        match_color: tuple[int, int, int] = (0, 255, 0),
    ) -> list[Image | None]:
        """Visualiza los matches entre un template y múltiples imágenes.
        
        Args:
            template: Imagen del template/ROI.
            images: Lista de imágenes (misma que se pasó a locate_template).
            localizations: Resultados de locate_template.
            max_matches: Máximo de matches a dibujar por imagen.
            draw_bbox: Si True, dibuja el bounding box proyectado.
            bbox_color: Color BGR del bounding box.
            match_color: Color BGR de las líneas de match.
            
        Returns:
            Lista de Image del mismo tamaño que images.
            Contiene None para imágenes donde no había localización.
        """
        if len(images) != len(localizations):
            raise ValueError("images and localizations must have the same length")
        
        results: list[Image | None] = []
        
        # Preprocesar y extraer descriptores del template una sola vez
        template_processed = self._preprocess(template)
        template_extraction = self._extractor.extract(template_processed, self._method)
        img1 = template.data if template.is_color else cv2.cvtColor(template.data, cv2.COLOR_GRAY2BGR)
        
        for image, localization in zip(images, localizations):
            if localization is None:
                results.append(None)
                continue
            
            # Preparar imagen
            img2 = image.data if image.is_color else cv2.cvtColor(image.data, cv2.COLOR_GRAY2BGR)
            
            if not template_extraction.has_descriptors():
                # Sin descriptores, solo concatenar imágenes
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                h = max(h1, h2)
                result_data = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
                result_data[:h1, :w1] = img1
                result_data[:h2, w1:w1+w2] = img2
                results.append(Image(data=result_data, rois=[]))
                continue
            
            image_processed = self._preprocess(image)
            image_extraction = self._extractor.extract(image_processed, self._method)
            
            if not image_extraction.has_descriptors():
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                h = max(h1, h2)
                result_data = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
                result_data[:h1, :w1] = img1
                result_data[:h2, w1:w1+w2] = img2
                results.append(Image(data=result_data, rois=[]))
                continue
            
            # Hacer matching
            desc_template = template_extraction._descriptors
            desc_image = image_extraction._descriptors
            
            try:
                matches = self._matcher.knnMatch(desc_template, desc_image, k=2)
            except cv2.error:
                matches = []
            
            # Ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self._ratio_threshold * n.distance:
                        good_matches.append(m)
            
            # Filtrar por inliers usando la homografía del resultado
            if localization.homography is not None and len(good_matches) >= 4:
                kp_template = template_extraction.keypoints
                kp_image = image_extraction.keypoints
                H = localization.homography
                
                inlier_matches = []
                for m in good_matches:
                    x, y = kp_template[m.queryIdx].position
                    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
                    if abs(w) < 1e-10:
                        continue
                    xp = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
                    yp = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
                    
                    x_real, y_real = kp_image[m.trainIdx].position
                    dist = np.sqrt((xp - x_real)**2 + (yp - y_real)**2)
                    
                    if dist < 10.0:
                        inlier_matches.append(m)
                
                good_matches = inlier_matches
            
            # Limitar número de matches
            good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]
            
            # Obtener keypoints de OpenCV
            kp1 = [kp._cv_kp for kp in template_extraction.keypoints]
            kp2 = [kp._cv_kp for kp in image_extraction.keypoints]
            
            # Dibujar matches
            result_data = cv2.drawMatches(
                img1, kp1, img2, kp2, good_matches, None,
                matchColor=match_color,
                singlePointColor=(255, 0, 0),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            
            # Dibujar bounding box proyectado
            if draw_bbox and localization.homography is not None:
                h_t, w_t = template.data.shape[:2]
                template_corners = np.float32([
                    [0, 0],
                    [w_t, 0],
                    [w_t, h_t],
                    [0, h_t],
                ])
                
                H = localization.homography
                projected_corners = []
                for pt in template_corners:
                    x, y = pt
                    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
                    if abs(w) < 1e-10:
                        continue
                    xp = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
                    yp = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
                    projected_corners.append([xp + img1.shape[1], yp])
                
                if len(projected_corners) == 4:
                    pts = np.array(projected_corners, dtype=np.int32)
                    cv2.polylines(result_data, [pts], True, bbox_color, 3)
            
            result = Image(data=result_data, rois=[])
            result.metadata.add_step({
                "technique": "visualize_localization",
                "num_matches": len(good_matches),
                "score": localization.score,
            })
            
            results.append(result)
        
        return results
