
import os
from typing import List, Optional
import cv2
import json
import numpy as np
import enum
from dgst import DATA_ROOT
import copy


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
            self.intrinsics = np.array(fc.get("intrinsics", []))
            self.extrinsics = np.array(fc.get("extrinsics", []))
            self.lidar_extrinsics = np.array(fc.get("lidar_extrinsics", []))
            self.image_dimensions = fc.get("image_dimensions", [])
            self.distortion = fc.get("distortion", [])
            self.field_of_view = fc.get("field_of_view", [])
            self.xi = fc.get("xi")
            self.undistortion = fc.get("undistortion", [])
        else:
            self.camera_type = None
            self.intrinsics = None
            self.extrinsics = None
            self.lidar_extrinsics = None
            self.image_dimensions = None
            self.distortion = None
            self.field_of_view = None
            self.xi = None
            self.undistortion = None

    def __repr__(self):
        return f"Calibration(camera_type={self.camera_type}, image_dimensions={self.image_dimensions}, intrinsics={self.intrinsics}, distortion={self.distortion}, undistortion={self.undistortion})"

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

class ImageFormat(enum.Enum):
    BGR = 1
    GRAYSCALE = 2
    BOOLEAN = 3

class Image: 
    def __init__(self, data: np.ndarray, rois: list[RegionOfInterest], calibration: Calibration = None, format: ImageFormat = ImageFormat.BGR):
        self.data = data
        self.rois = rois
        self.calibration = calibration
        self.format = format
        self.hsv_channels: Optional[List[np.ndarray]] = None

    def show_rois(self):

        if self.data is None or self.rois is None:
            return
        
        if self.format != ImageFormat.BGR and self.format != ImageFormat.GRAYSCALE:
            raise ValueError("Image format must be BGR or GRAYSCALE to show ROIs.")

        img_copy = self.data.copy()
        for roi in self.rois:
            pts = np.array([roi.p1, roi.p2, roi.p3, roi.p4], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(
                img_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=2
            )
        self.data = img_copy

    def clone(self) -> "Image":
        """Return a deep copy of this Image.

        - numpy array for image data is copied via np.copy
        - ROI objects are cloned
        - Calibration is cloned if present
        """
        data_copy = None
        if self.data is not None:
            try:
                data_copy = np.copy(self.data)
            except Exception:
                # fallback to deepcopy
                data_copy = copy.deepcopy(self.data)

        rois_copy = [r.clone() for r in self.rois] if self.rois is not None else []
        calibration_copy = self.calibration.clone() if self.calibration is not None else None

        if self.hsv_channels is not None:
            try:
                hsv_copy = [np.copy(channel) for channel in self.hsv_channels]
            except Exception:
                hsv_copy = copy.deepcopy(self.hsv_channels)
        else:
            hsv_copy = None

        res = Image(data=data_copy, rois=rois_copy, calibration=calibration_copy, format=self.format)
        res.hsv_channels = hsv_copy

        return res
    
    def get_hsv_channel(self, channel: str) -> np.ndarray:
        """Get a specific HSV channel from the image.

        Args:
            channel (str): One of 'H', 'S', or 'V'.
        Returns:
            np.ndarray: The requested HSV channel.
        """

        if self.hsv_channels is None:
            raise ValueError("HSV channels not computed. Please convert the image to HSV first.")
        
        channel = channel.upper()
        if channel == 'H':
            return self.hsv_channels[0]
        elif channel == 'S':
            return self.hsv_channels[1]
        elif channel == 'V':
            return self.hsv_channels[2]
        else:
            raise ValueError("Invalid channel specified. Use 'H', 'S', or 'V'.")

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

    def load_metadata(self, number: str) -> list[RegionOfInterest]:
        image_path = os.path.join(self._path, str(number).zfill(6))
        metadata_path = os.path.join(
            image_path, "annotations/traffic_signs.json"
        )
        result = []
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
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
            with open(metadata_path, "r") as f:
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

    def load_calibration(self, number: int) -> Calibration:
        image_path = os.path.join(self._path, str(number).zfill(6))
        calibration_path = os.path.join(image_path, "calibration.json")

        if os.path.exists(calibration_path):
            with open(calibration_path, "r") as f:
                calibration_data = json.load(f)
                return Calibration(calibration_data)
        return None

    def load(self, number: int) -> Image:
        image = self.load_image(number)
        rois = self.load_metadata(number)
        calibration = self.load_calibration(number)
        return Image(data=image, rois=rois, calibration=calibration)
