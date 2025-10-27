import os

import cv2
import json
import numpy as np

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


class DataLoader:
    def __init__(self, path=None):
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


