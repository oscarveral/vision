import cv2
import sys, os
from abc import ABC, abstractmethod
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ffi.wrapper import box_filter

class Filter(ABC):
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def set_parameters(self, params: dict):
        pass

class ConvolutionFilter(Filter):
    def __init__(self, ksize: int = 5):
        self._ksize = ksize

    @property
    def ksize(self):
        return self._ksize
    
    @ksize.setter
    def ksize(self, ksize: int):
        assert ksize % 2 == 1, "Kernel size must be odd."
        self._ksize = ksize

class CVBoxFilter(ConvolutionFilter):
    def __init__(self, ksize: int = 5):
        super().__init__(ksize)

    def set_parameters(self, params: np.array):
        if 'ksize' in params:
            self.ksize = params['ksize']

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.boxFilter(image, -1, (self.ksize, self.ksize))
    
class DGSTBoxFilter(ConvolutionFilter):
    def __init__(self, ksize: int = 5):
        super().__init__(ksize)

    def set_parameters(self, params: dict):
        if 'ksize' in params:
            self.ksize = params['ksize']

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = box_filter(image, self.ksize)
        return image

str2filter = {"cv_box": CVBoxFilter, "dgst_box": DGSTBoxFilter}

def get_filter(filter_name: str) -> Filter:
    if filter_name not in str2filter:
        raise ValueError(f"Filter '{filter_name}' not recognized. Available filters: {list(str2filter.keys())}")
    return str2filter[filter_name]()


def main():
    if len(sys.argv) < 2:
        print("Usage: python filters.py <filter_name>")
        sys.exit(1)

    filter_name = sys.argv[1]
    filter_instance = get_filter(filter_name)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit(1)

    # Create a window and a trackbar (slider) for ksize
    window_name = 'Filtered Video'
    cv2.namedWindow(window_name)
    def on_trackbar(val):
        # Trackbar value is always odd (2n+1)
        ksize = val * 2 + 1
        filter_instance.ksize = ksize

    # Initial ksize value
    initial_val = (filter_instance.ksize - 1) // 2
    cv2.createTrackbar('ksize', window_name, initial_val, 32, on_trackbar)  # 1 to 65 (step 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get current ksize from trackbar
        trackbar_val = cv2.getTrackbarPos('ksize', window_name)
        ksize = trackbar_val * 2 + 1
        filter_instance.ksize = ksize

        start_time = cv2.getTickCount()
        filtered_frame = filter_instance.apply(frame)
        end_time = cv2.getTickCount()
        delay_ms = (end_time - start_time) / cv2.getTickFrequency() * 1000

        # Put delay on the frame
        cv2.putText(
            filtered_frame,
            f"Filter delay: {delay_ms:.2f} ms | ksize: {ksize}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),    
            2,
            cv2.LINE_AA
        )
        cv2.imshow(window_name, filtered_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()