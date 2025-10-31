import numpy as np

from dgst.ffi.wrapper import ransac_line_fitting

class FeatureExtractor:
    """Image features extractor.
    """

    def __init__(self):
        pass

    def ransac_line_fitting(
            self,
            edge_image: np.ndarray,
            max_iterations: int,
            distance_threshold: float,
            min_inliers: int,
            max_lsq_iterations: int = 0,
            erase: bool = False
    ) -> tuple:
        """Fit a line to edge points using RANSAC.

        Args:
            edge_image: Binary edge map as a 2D numpy array of type bool
            max_iterations: Number of RANSAC iterations
            distance_threshold: Distance threshold to consider a point as an inlier
            min_inliers: Minimum number of inliers to accept a model
            max_lsq_iterations: Number of least squares refinement iterations. Set to 0 to skip refinement.

        Returns:
            A tuple (a, b, c) representing the line equation ax + by + c = 0
        """

        if edge_image.dtype != np.bool_:
            raise ValueError("Edge image must be of type bool.")
        if len(edge_image.shape) != 2:
            raise ValueError("Edge image must be a 2D array.")
        if not np.any(edge_image):
            raise ValueError("Edge image contains no edge points.")
        
        if min_inliers <= 0:
            raise ValueError("Minimum number of inliers must be positive.")
        if max_lsq_iterations < 0:
            raise ValueError("Maximum number of least squares iterations cannot be negative.") 
        if max_iterations <= 0:
            raise ValueError("Maximum number of iterations must be positive.")
        if distance_threshold <= 0:
            raise ValueError("Distance threshold must be positive.")

        result = ransac_line_fitting(
            edge_map=edge_image,
            max_iterations=max_iterations,
            max_lsq_iterations=max_lsq_iterations,
            distance_threshold=distance_threshold,
            min_inlier_count=min_inliers
        )

        if result is None:
            return None
        
        if erase:
            a, b, c = result
            height, width = edge_image.shape
            yy, xx = np.nonzero(edge_image)
            distances = np.abs(a * xx + b * yy + c) / np.sqrt(a ** 2 + b ** 2)
            inlier_mask = distances <= distance_threshold
            edge_image[yy[inlier_mask], xx[inlier_mask]] = False
        
        return result
    
    def remove_line(
            self,
            edge_image: np.ndarray,
            line: tuple,
            distance_threshold: float
    ) -> np.ndarray:
        """Remove inliers of a given line from the edge image.

        Args:
            edge_image: Binary edge map as a 2D numpy array of type bool
            line: A tuple (a, b, c) representing the line equation ax + by + c = 0
            distance_threshold: Distance threshold to consider a point as an inlier

        Returns:
            The edge image with inliers removed.
        """

        if edge_image.dtype != np.bool_:
            raise ValueError("Edge image must be of type bool.")
        if len(edge_image.shape) != 2:
            raise ValueError("Edge image must be a 2D array.")
        if line is None or len(line) != 3:
            raise ValueError("Line must be a tuple of (a, b, c).")
        a, b, c = line
        if a == 0 and b == 0:
            raise ValueError("Invalid line parameters: a and b cannot both be zero.")
        if distance_threshold <= 0:
            raise ValueError("Distance threshold must be positive.")

        a, b, c = line
        height, width = edge_image.shape
        yy, xx = np.nonzero(edge_image)
        distances = np.abs(a * xx + b * yy + c) / np.sqrt(a ** 2 + b ** 2)
        inlier_mask = distances <= distance_threshold
        edge_image[yy[inlier_mask], xx[inlier_mask]] = False

        return edge_image
