import numpy as np

from dgst.ffi.wrapper import ransac_line_fitting

class FeatureExtractor:
    """Image features extractor.
    """

    def __init__(self, edge_image: np.ndarray):
        if edge_image.dtype != np.bool_:
            raise ValueError("Edge image must be of type bool.")
        if len(edge_image.shape) != 2:
            raise ValueError("Edge image must be a 2D array.")
        self._edge_image = edge_image

    def ransac_line_fitting(
            self,
            max_iterations: int,
            distance_threshold: float,
            min_inliers: int,
            max_lsq_iterations: int = 0,
            erase: bool = False
    ) -> tuple:
        """Fit a line to edge points using RANSAC.

        Args:
            max_iterations: Number of RANSAC iterations
            distance_threshold: Distance threshold to consider a point as an inlier
            min_inliers: Minimum number of inliers to accept a model
            max_lsq_iterations: Number of least squares refinement iterations. Set to 0 to skip refinement.
            erase: If True, remove the inliers of the detected line from the edge image.

        Returns:
            A tuple (a, b, c) representing the line equation ax + by + c = 0 
        """
        if min_inliers <= 0:
            raise ValueError("Minimum number of inliers must be positive.")
        if max_lsq_iterations < 0:
            raise ValueError("Maximum number of least squares iterations cannot be negative.") 
        if max_iterations <= 0:
            raise ValueError("Maximum number of iterations must be positive.")
        if distance_threshold <= 0:
            raise ValueError("Distance threshold must be positive.")

        result = ransac_line_fitting(
            edge_map=self._edge_image,
            max_iterations=max_iterations,
            max_lsq_iterations=max_lsq_iterations,
            distance_threshold=distance_threshold,
            min_inlier_count=min_inliers
        )

        if result is None:
            return None, self._edge_image

        if erase:
            self._remove_line(result, distance_threshold)
            self._edge_image = self._edge_image

        return result, self._edge_image

    def _remove_line(
            self,
            line: tuple,
            distance_threshold: float
    ):
        """Remove inliers of a given line from the edge image.

        Args:
            line: A tuple (a, b, c) representing the line equation ax + by + c = 0
            distance_threshold: Distance threshold to consider a point as an inlier

        Returns:
            The edge image with inliers removed.
        """

        if line is None or len(line) != 3:
            raise ValueError("Line must be a tuple of (a, b, c).")
        a, b, c = line
        if a == 0 and b == 0:
            raise ValueError("Invalid line parameters: a and b cannot both be zero.")
        if distance_threshold <= 0:
            raise ValueError("Distance threshold must be positive.")

        a, b, c = line
        height, width = self._edge_image.shape
        yy, xx = np.nonzero(self._edge_image)
        distances = np.abs(a * xx + b * yy + c) / np.sqrt(a ** 2 + b ** 2)
        inlier_mask = distances <= distance_threshold
        self._edge_image[yy[inlier_mask], xx[inlier_mask]] = False

    def windowed_ramsac_line_fitting(
            self,
            window_size: int,
            step: int,
            max_iterations: int,
            distance_threshold: float,
            min_inliers: int,
            max_lsq_iterations: int = 0,
            erase: bool = False
    ) -> list:
        """Apply RANSAC line fitting in a sliding window fashion.

        Args:
            window_size: Size of the sliding window (square).
            step: Step size for sliding the window.
            max_iterations: Number of RANSAC iterations per window.
            distance_threshold: Distance threshold to consider a point as an inlier.
            min_inliers: Minimum number of inliers to accept a model.
            max_lsq_iterations: Number of least squares refinement iterations. Set to 0 to skip refinement.
            erase: If True, remove the inliers of the detected lines from the edge image.

        Returns:
            A list of detected lines represented as tuples (a, b, c).
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive.")
        if step <= 0:
            raise ValueError("Step size must be positive.")
        if max_iterations <= 0:
            raise ValueError("Maximum number of iterations must be positive.")
        if distance_threshold <= 0:
            raise ValueError("Distance threshold must be positive.")
        if min_inliers <= 0:
            raise ValueError("Minimum number of inliers must be positive.")
        if max_lsq_iterations < 0:
            raise ValueError("Maximum number of least squares iterations cannot be negative.")

        detected_lines = []

        for y in range(0, self._edge_image.shape[0] - window_size + 1, step):
            for x in range(0, self._edge_image.shape[1] - window_size + 1, step):
                window = self._edge_image[y:y + window_size, x:x + window_size]
                feature_extractor = FeatureExtractor(edge_image=window.copy())
                line, _ = feature_extractor.ransac_line_fitting(
                    max_iterations=max_iterations,
                    distance_threshold=distance_threshold,
                    min_inliers=min_inliers,
                    max_lsq_iterations=max_lsq_iterations,
                    erase=False
                )
                if line is not None:
                    # Adjust line parameters to the original image coordinates
                    a, b, c = line
                    adjusted_c = c - a * x - b * y
                    adjusted_line = (a, b, adjusted_c)
                    detected_lines.append(adjusted_line)
                    if erase:
                        self._remove_line(adjusted_line, distance_threshold)

        return detected_lines, self._edge_image
