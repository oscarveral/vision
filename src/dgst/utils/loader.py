
import copy
import json
import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from dgst import DATA_ROOT


class RegionOfInterest:
    def __init__(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
    ):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def __repr__(self):
        return f"RegionOfInterest(p1={self.p1}, p2={self.p2}, p3={self.p3}, p4={self.p4})"

    def clone(self) -> "RegionOfInterest":
        """Return a deep copy of this RegionOfInterest."""
        # tuples are immutable but create new tuples for clarity
        return RegionOfInterest(
            p1=(float(self.p1[0]), float(self.p1[1])),
            p2=(float(self.p2[0]), float(self.p2[1])),
            p3=(float(self.p3[0]), float(self.p3[1])),
            p4=(float(self.p4[0]), float(self.p4[1])),
        )


class Calibration:
    def __init__(self, calibration_data: dict):
        self.data = calibration_data

        # Extract front camera calibration if available
        if "FC" in calibration_data:
            fc = calibration_data["FC"]
            self.camera_type = fc.get("camera_type")
            self.intrinsics = np.array(fc.get("intrinsics", [])) if fc.get("intrinsics") is not None else None
            self.extrinsics = np.array(fc.get("extrinsics", [])) if fc.get("extrinsics") is not None else None
            self.lidar_extrinsics = (
                np.array(fc.get("lidar_extrinsics", [])) 
                if fc.get("lidar_extrinsics") is not None 
                else None
            )
            self.image_dimensions = fc.get("image_dimensions")
            self.distortion = fc.get("distortion")
            self.field_of_view = fc.get("field_of_view")
            self.xi = fc.get("xi")
            self.undistortion = fc.get("undistortion")

    def __repr__(self):
        return (
            f"Calibration(camera_type={self.camera_type}, "
            f"image_dimensions={self.image_dimensions}, "
            f"intrinsics={self.intrinsics}, "
            f"distortion={self.distortion}, "
            f"undistortion={self.undistortion})"
        )

    def clone(self) -> "Calibration":
        """Return a deep copy of this Calibration."""
        # Create a shallow copy of the stored dict, but deep-copy numpy arrays where applicable
        data_copy = copy.deepcopy(self.data) if self.data is not None else None

        cloned = Calibration(data_copy if data_copy is not None else {})

        # Deep copy numpy arrays if present
        try:
            if isinstance(self.intrinsics, np.ndarray):
                cloned.intrinsics = np.copy(self.intrinsics)
            else:
                cloned.intrinsics = (
                    None
                    if self.intrinsics is None
                    else np.array(self.intrinsics)
                )

            if isinstance(self.extrinsics, np.ndarray):
                cloned.extrinsics = np.copy(self.extrinsics)
            else:
                cloned.extrinsics = (
                    None
                    if self.extrinsics is None
                    else np.array(self.extrinsics)
                )

            if isinstance(self.lidar_extrinsics, np.ndarray):
                cloned.lidar_extrinsics = np.copy(self.lidar_extrinsics)
            else:
                cloned.lidar_extrinsics = (
                    None
                    if self.lidar_extrinsics is None
                    else np.array(self.lidar_extrinsics)
                )
        except Exception:
            # Fallback to deepcopy for any unexpected structure
            cloned = Calibration(
                copy.deepcopy(self.data) if self.data is not None else {}
            )

        # Copy simple fields
        cloned.camera_type = copy.deepcopy(self.camera_type)
        cloned.image_dimensions = copy.deepcopy(self.image_dimensions)
        cloned.distortion = copy.deepcopy(self.distortion)
        cloned.field_of_view = copy.deepcopy(self.field_of_view)
        cloned.xi = copy.deepcopy(self.xi)
        cloned.undistortion = copy.deepcopy(self.undistortion)

        return cloned




class ProcessingMetadata:
    """Metadata tracking for image processing operations."""
    
    def __init__(self):
        self.steps: list[dict[str, Any]] = []
        self.original_shape: tuple | None = None
        self.creation_time: str = datetime.now().isoformat()
    
    def add_step(self, step_info: dict[str, Any]) -> None:
        """Add a processing step to the metadata."""
        step_info['timestamp'] = datetime.now().isoformat()
        self.steps.append(step_info)
    
    def get_step_count(self) -> int:
        """Get the total number of processing steps."""
        return len(self.steps)
    
    def get_last_step(self) -> dict[str, Any] | None:
        """Get information about the last processing step."""
        return self.steps[-1] if self.steps else None
    
    def clone(self) -> "ProcessingMetadata":
        """Create a deep copy of this metadata."""
        new_metadata = ProcessingMetadata()
        new_metadata.steps = copy.deepcopy(self.steps)
        new_metadata.original_shape = self.original_shape
        new_metadata.creation_time = self.creation_time
        return new_metadata
    
    def __repr__(self) -> str:
        return f"ProcessingMetadata(steps={len(self.steps)}, original_shape={self.original_shape})"


class Image: 
    def __init__(
        self, 
        data: np.ndarray, 
        rois: list[RegionOfInterest] | None = None, 
        calibration: Calibration | None = None, 
    ):
        self.data = data
        self.rois = rois if rois is not None else []
        self.calibration = calibration
        self.metadata: ProcessingMetadata = ProcessingMetadata()
        
        # Store original properties in metadata
        if data is not None:
            self.metadata.original_shape = data.shape

    @property
    def is_grayscale(self) -> bool:
        """Check if image is grayscale (2D or 3D with 1 channel)."""
        if self.data is None:
            return False
        return self.data.ndim == 2 or (self.data.ndim == 3 and self.data.shape[2] == 1)

    @property
    def is_color(self) -> bool:
        """Check if image is color (3D with 3 channels)."""
        if self.data is None:
            return False
        return self.data.ndim == 3 and self.data.shape[2] == 3

    @property
    def num_channels(self) -> int:
        """Get number of channels."""
        if self.data is None:
            return 0
        if self.data.ndim == 2:
            return 1
        return self.data.shape[2]

    def to_grayscale(self) -> "Image":
        """Convert image to grayscale in-place."""
        if self.data is None:
            return self
            
        if self.is_color:
            self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        return self

    def to_bgr(self) -> "Image":
        """Convert image to BGR in-place."""
        if self.data is None:
            return self
            
        if self.is_grayscale:
            self.data = cv2.cvtColor(self.data, cv2.COLOR_GRAY2BGR)
        return self

    def show_rois(self):
        if self.data is None or not self.rois:
            return
        
        # Ensure we can draw colors
        img_copy = self.data.copy()
        if self.is_grayscale:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

        for roi in self.rois:
            pts = np.array([roi.p1, roi.p2, roi.p3, roi.p4], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(
                img_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=2
            )
        self.data = img_copy

    def clone(self) -> "Image":
        """Return a deep copy of this Image."""
        data_copy = None
        if self.data is not None:
            data_copy = np.copy(self.data)

        rois_copy = [r.clone() for r in self.rois]
        calibration_copy = self.calibration.clone() if self.calibration is not None else None

        res = Image(data=data_copy, rois=rois_copy, calibration=calibration_copy)
        res.metadata = self.metadata.clone()

        return res
    
    def get_image(self) -> np.ndarray:
        """Get the image data as a numpy array."""
        return self.data

class DataLoader:
    def __init__(self, path=DATA_ROOT):
        self._path = path

    def load_image(self, number: int) -> np.ndarray:
        image_path = os.path.join(self._path, str(number).zfill(6))

        image_path = os.path.join(image_path, "camera_front_blur")
        image_path = os.path.join(image_path, os.listdir(image_path)[0])
        print(f"Loading image from {image_path}")
        image = cv2.imread(image_path)
        return image

    def load_metadata(self, number: int) -> list[RegionOfInterest]:
        image_path = os.path.join(self._path, str(number).zfill(6))
        metadata_path = os.path.join(
            image_path, "annotations/traffic_signs.json"
        )
        result = []
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
                for item in metadata:
                    coordinates = item["geometry"]["coordinates"]
                    roi = RegionOfInterest(
                        p1=(coordinates[0][0], coordinates[0][1]),
                        p2=(coordinates[1][0], coordinates[1][1]),
                        p3=(coordinates[2][0], coordinates[2][1]),
                        p4=(coordinates[3][0], coordinates[3][1]),
                    )
                    result.append(roi)

        metadata_path = os.path.join(
            image_path, "annotations/object_detection.json"
        )
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
                for item in metadata:
                    properties = item["properties"]
                    if properties["class"] == "TrafficSign":
                        coordinates = item["geometry"]["coordinates"]
                        roi = RegionOfInterest(
                            p1=(coordinates[0][0], coordinates[0][1]),
                            p2=(coordinates[1][0], coordinates[1][1]),
                            p3=(coordinates[2][0], coordinates[2][1]),
                            p4=(coordinates[3][0], coordinates[3][1]),
                        )
                        result.append(roi)
        return result

    def load_calibration(self, number: int) -> Calibration | None:
        image_path = os.path.join(self._path, str(number).zfill(6))
        calibration_path = os.path.join(image_path, "calibration.json")

        if os.path.exists(calibration_path):
            with open(calibration_path) as f:
                calibration_data = json.load(f)
                return Calibration(calibration_data)
        return None

    def load(self, number: int) -> Image:
        image = self.load_image(number)
        rois = self.load_metadata(number)
        calibration = self.load_calibration(number)
        return Image(data=image, rois=rois, calibration=calibration)
    
    def load_all(self) -> list[Image]:
        images = []
        for entry in os.listdir(self._path):
            entry_path = os.path.join(self._path, entry)
            if os.path.isdir(entry_path) and entry.isdigit():
                number = int(entry)
                img = self.load(number)
                images.append(img)
        return images
