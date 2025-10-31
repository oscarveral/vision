from email.mime import image
import os
import cv2
import json
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

class Calibration:
    def __init__(self, calibration_data: dict):
        self.data = calibration_data
        
        # Extract front camera calibration if available
        if 'FC' in calibration_data:
            fc = calibration_data['FC']
            self.camera_type = fc.get('camera_type')
            self.intrinsics = np.array(fc.get('intrinsics', []))
            self.extrinsics = np.array(fc.get('extrinsics', []))
            self.lidar_extrinsics = np.array(fc.get('lidar_extrinsics', []))
            self.image_dimensions = fc.get('image_dimensions', [])
            self.distortion = fc.get('distortion', [])
            self.field_of_view = fc.get('field_of_view', [])
            self.xi = fc.get('xi')
            self.undistortion = fc.get('undistortion', [])
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

class Image: 
    def __init__(self, data: np.ndarray, rois: list[RegionOfInterest], calibration: Calibration = None):
        self.data = data
        self.rois = rois
        self.calibration = calibration

    def show_rois(self):
        img_copy = self.data.copy()
        for roi in self.rois:
            pts = np.array([roi.p1, roi.p2, roi.p3, roi.p4], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        self.data = img_copy

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
        metadata_path = os.path.join(image_path, "annotations/traffic_signs.json")
        result = []
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                for item in metadata:
                    coordinates = item['geometry']['coordinates']
                    roi = RegionOfInterest(
                        p1=(coordinates[0][0], coordinates[0][1]),
                        p2=(coordinates[1][0], coordinates[1][1]),
                        p3=(coordinates[2][0], coordinates[2][1]),
                        p4=(coordinates[3][0], coordinates[3][1]),
                    )
                    result.append(roi)

        metadata_path = os.path.join(image_path, "annotations/object_detection.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                for item in metadata:
                    properties = item['properties']
                    if properties['class'] == 'TrafficSign':
                        coordinates = item['geometry']['coordinates']
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
            with open(calibration_path, 'r') as f:
                calibration_data = json.load(f)
                return Calibration(calibration_data)
        return None

    def load(self, number: int) -> Image:
        image = self.load_image(number)
        rois = self.load_metadata(number)
        calibration = self.load_calibration(number)
        return Image(data=image, rois=rois, calibration=calibration)
