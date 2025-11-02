import numpy as np
import cv2
from pyparsing import Optional

from dgst.filters.ffi import ransac_line_fitting, ransac_circle_fitting
from dgst.utils.loader import Image, ImageFormat

class FeatureExtractor:
    """Image features extractor."""

    def __init__(self, edge_image: Image):
        if edge_image.format != ImageFormat.BOOLEAN:
            raise ValueError("Edge image must be of type bool.")
        if len(edge_image.data.shape) != 2:
            raise ValueError("Edge image must be a 2D array.")
        self._edge_image = edge_image.clone()
        self._original_edge_image = edge_image.clone()
        self._lines = []
        self._segments = []
        self._circles = []

    def ransac_line_fitting(
        self,
        max_iterations: int,
        distance_threshold: float,
        min_inliers: int,
        max_lsq_iterations: int = 0,
        erase: bool = False,
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
            raise ValueError(
                "Maximum number of least squares iterations cannot be negative."
            )
        if max_iterations <= 0:
            raise ValueError("Maximum number of iterations must be positive.")
        if distance_threshold <= 0:
            raise ValueError("Distance threshold must be positive.")

        result = ransac_line_fitting(
            edge_map=self._edge_image.data,
            max_iterations=max_iterations,
            max_lsq_iterations=max_lsq_iterations,
            distance_threshold=distance_threshold,
            min_inlier_count=min_inliers,
        )

        if result is None:
            return None

        self._lines.append(result)

        if erase:
            self._remove_line(result, distance_threshold)

        return result

    def _remove_line(self, line: tuple, distance_threshold: float):
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
            raise ValueError(
                "Invalid line parameters: a and b cannot both be zero."
            )
        if distance_threshold <= 0:
            raise ValueError("Distance threshold must be positive.")

        a, b, c = line
        yy, xx = np.nonzero(self._edge_image.data)
        distances = np.abs(a * xx + b * yy + c) / np.sqrt(a ** 2 + b ** 2)
        inlier_mask = distances <= distance_threshold
        self._edge_image.data[yy[inlier_mask], xx[inlier_mask]] = False

    def windowed_ransac_line_fitting(
        self,
        window_size: int,
        step: int,
        max_iterations: int,
        distance_threshold: float,
        min_inliers: int,
        max_lsq_iterations: int = 0,
        erase: bool = False,
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
            raise ValueError(
                "Maximum number of least squares iterations cannot be negative."
            )

        detected_lines = []

        for y in range(0, self._edge_image.data.shape[0] - window_size + 1, step):
            for x in range(0, self._edge_image.data.shape[1] - window_size + 1, step):
                window = self._edge_image.data[y:y + window_size, x:x + window_size].copy()
                
                # Call RANSAC directly on the window
                line = ransac_line_fitting(
                    edge_map=window,
                    max_iterations=max_iterations,
                    max_lsq_iterations=max_lsq_iterations,
                    distance_threshold=distance_threshold,
                    min_inlier_count=min_inliers,
                )
                
                if line is not None:
                    # Adjust line parameters to the original image coordinates
                    a, b, c = line
                    adjusted_c = c - a * x - b * y
                    adjusted_line = (a, b, adjusted_c)
                    detected_lines.append(adjusted_line)
                    if erase:
                        self._remove_line(adjusted_line, distance_threshold)

        self._lines.extend(detected_lines)    
        return detected_lines
    
    def get_line_support(
        self,
        line: tuple,
        distance_threshold: float,
        density_threshold: float,
        min_segment_length: float = 0.0,
        erase: bool = False,
    ) -> np.ndarray:
        """Get the support points of a given line from the edge image. Takes only the pixels
        that satisfy the line equation. Must be called after fitting a line with RANSAC
        with erase=False.

        Args:
            line: A tuple (a, b, c) representing the line equation ax + by + c = 0
            distance_threshold: Maximum distance from the line to consider a point as support.
            density_threshold: Proportion of points that must satisfy the line equation to be considered a valid segment.
            min_segment_length: Minimum length of the segment to be considered valid. Set to 0 to skip this check.
            erase: If True, remove the support points from the edge image.
        Returns:
            (x_coords, y_coords): Arrays of x and y coordinates of the support points. None if not enough support.
        """
        if line is None or len(line) != 3:
            raise ValueError("Line must be a tuple of (a, b, c).")
        a, b, c = line
        if abs(a) < 1e-6 and abs(b) < 1e-6:
            raise ValueError(
                "Invalid line parameters: a and b cannot both be zero."
            )
        if density_threshold <= 0 or density_threshold > 1:
            raise ValueError("Density threshold must be in the range (0, 1].")
        if distance_threshold <= 0:
            raise ValueError("Distance threshold must be positive.")

        # Normalize line parameters
        norm = np.sqrt(a**2 + b**2)
        a_norm = a / norm
        b_norm = b / norm
        c_norm = c / norm

        # Mask the points that satisfy the line equation
        yy, xx = np.nonzero(self._edge_image.data)
        distances = a_norm * xx + b_norm * yy + c_norm
        inlier_mask = np.abs(distances) <= distance_threshold

        xx = xx[inlier_mask]
        yy = yy[inlier_mask]
        distances = distances[inlier_mask]
        x_proj = xx - distances * a_norm
        y_proj = yy - distances * b_norm
        projections = np.array(list(zip(x_proj, y_proj)))

        if len(xx) < 2:
            return None  # No support points

        # Sort points, projections and the corresponding coordinate arrays with the same permutation
        sort_col = 0 if abs(b_norm) > abs(a_norm) else 1
        sorted_idx = np.argsort(projections[:, sort_col])
        projections = projections[sorted_idx]
        xx = xx[sorted_idx]
        yy = yy[sorted_idx]
        # Store distance to first point
        points = np.linalg.norm(projections - projections[0], axis=1)
        N = len(points)

        ## Kadane algorithm to find longest segment with points within distance threshold

        # Initialize
        A = []  # A[k] = 1 - thr * (points[k+1]-points[k])
        for i in range(N - 1):
            delta_x = points[i + 1] - points[i]
            A.append(1 - density_threshold * delta_x)
        global_max_sum = 0.0
        current_max_sum = 0.0
        best_start_index = -1
        best_end_index = -1
        temp_start_index = 0

        # Loop over array A
        for k in range(len(A)):
            # Expand the current segment
            current_max_sum = current_max_sum + A[k]

            # If the current segment is better than the best found so far, update best segment
            if current_max_sum > global_max_sum:
                global_max_sum = current_max_sum
                best_start_index = (
                    temp_start_index  # Current segment starts at 'temp'
                )
                best_end_index = (
                    k + 1
                )  # And ends at k+1 (since A has length N-1)

            # If the current sum is non-positive, reset it and move the start index
            if current_max_sum <= 0:
                current_max_sum = 0.0
                temp_start_index = k + 1

        # Compute the metrics of the found segment
        if global_max_sum > 0:
            # The indices are from the original array of POINTS
            start_x = xx[best_start_index]
            start_y = yy[best_start_index]
            end_x = xx[best_end_index]
            end_y = yy[best_end_index]
            ans = ([start_x, start_y], [end_x, end_y])

            if erase:
                # Remove support points from edge image
                for i in range(best_start_index, best_end_index + 1):
                    self._edge_image.data[yy[i], xx[i]] = False

            if min_segment_length > 0.0:
                length = np.sqrt(
                    (end_x - start_x) ** 2 + (end_y - start_y) ** 2
                )
                if length < min_segment_length:
                    return None

            self._segments.append(ans)
            return ans
        else:
            return None

    def ransac_segment_fitting(
        self,
        max_iterations: int,
        distance_threshold_line: float,
        distance_threshold_segment: float,
        density_threshold: float,
        min_inliers: int,
        max_lsq_iterations: int = 0,
        min_segment_length: float = 0.0,
        erase: bool = False,
    ) -> tuple:
        """Fit a line to edge points using RANSAC and get its support segment.

        Args:
            max_iterations: Number of RANSAC iterations
            distance_threshold_line: Distance threshold to consider a point as an inlier of the line
            distance_threshold_segment: Distance threshold to consider a point as support of the segment
            density_threshold: Proportion of points that must satisfy the line equation to be considered a valid segment.
            min_inliers: Minimum number of inliers to accept a model
            max_lsq_iterations: Number of least squares refinement iterations. Set to 0 to skip refinement.
            min_segment_length: Minimum length of the segment to be considered valid. Set to 0 to skip this check.
            erase: If True two case are possible:
                If a no support segment is found, remove the inliers of the detected line from the edge image.
                If a support segment is found, remove the support points from the edge image.

        Returns:
           One segment endpoints as ([x_start, x_end], [y_start, y_end]) or None if no valid segment found.

        """
        # First, fit a line using RANSAC
        line = self.ransac_line_fitting(
            max_iterations=max_iterations,
            distance_threshold=distance_threshold_line,
            min_inliers=min_inliers,
            max_lsq_iterations=max_lsq_iterations,
            erase=False,
        )

        # If no line found, return None
        if line is None:
            return None, None

        # Then, get the support segment of the fitted line
        segment = self.get_line_support(
            line=line,
            distance_threshold=distance_threshold_segment,
            density_threshold=density_threshold,
            min_segment_length=min_segment_length,
            erase=erase,  # Support points will be erased even if segment length is smaller than min_segment_length
        )

        if segment is None and erase:
            # Remove inliers of the detected line from the edge image
            self._remove_line(line, distance_threshold_line)

        return line, segment

    @property
    def image(self) -> Image:
        """Get the current edge image.

        Returns:
            The current edge image.
        """
        return self._edge_image
    
    @property
    def binary_image(self) -> np.ndarray:
        """Get the current edge image as a binary numpy array.

        Returns:
            The current edge image as a binary numpy array.
        """
        return self._edge_image.data

    def ransac_circle_fitting(
        self,
        max_iterations: int,
        distance_threshold: float,
        min_inlier_ratio: float,
        min_radius: float = 0.0,
        max_radius: float = 0.0,
        erase: bool = False,
    ) -> tuple:
        """Fit a circle to edge points using RANSAC.

        Args:
            max_iterations: Number of RANSAC iterations
            distance_threshold: Distance threshold to consider a point as an inlier
            min_inlier_ratio: Minimum ratio of inliers to radius to accept a model. Recommended minimum is 3.
            min_radius: Minimum radius of the circle to be detected. Set to 0 to skip this check.
            max_radius: Maximum radius of the circle to be detected. Set to 0 to skip this check.
            erase: If True, remove the inliers of the detected circle from the edge image.

        Returns:
            A tuple (x_center, y_center, radius) representing the circle parameters.
        """
        if min_inlier_ratio <= 0:
            raise ValueError("Minimum inlier ratio must be positive.")
        if max_iterations <= 0:
            raise ValueError("Maximum number of iterations must be positive.")
        if distance_threshold <= 0:
            raise ValueError("Distance threshold must be positive.")

        result = ransac_circle_fitting(
            edge_map=self._edge_image.data,
            max_iterations=max_iterations,
            distance_threshold=distance_threshold,
            min_inlier_ratio=min_inlier_ratio,
            min_radius=min_radius,
            max_radius=max_radius,
        )

        if result is None:
            return None

        self._circles.append(result)

        if erase:
            self._remove_circle(result, distance_threshold)

        return result

    def _remove_circle(self, circle: tuple, distance_threshold: float):
        """Remove inliers of a given circle from the edge image.

        Args:
            circle: A tuple (x_center, y_center, radius) representing the circle parameters.
            distance_threshold: Distance threshold to consider a point as an inlier.

        Returns:
            The edge image with inliers removed.
        """
        if circle is None or len(circle) != 3:
            raise ValueError(
                "Circle must be a tuple of (x_center, y_center, radius)."
            )
        x_center, y_center, radius = circle
        if radius <= 0:
            raise ValueError(
                "Invalid circle parameters: radius must be positive."
            )
        if distance_threshold <= 0:
            raise ValueError("Distance threshold must be positive.")

        yy, xx = np.nonzero(self._edge_image.data)
        distances = np.sqrt((xx - x_center) ** 2 + (yy - y_center) ** 2)
        inlier_mask = np.abs(distances - radius) <= distance_threshold
        self._edge_image.data[yy[inlier_mask], xx[inlier_mask]] = False


    def paint_lines_on_image(self, color='red'):
        copy = self._original_edge_image.clone()
        image_with_lines = copy.data
        if (copy.data.ndim == 2):
            # Convert boolean to uint8 first, then to BGR
            image_with_lines = (copy.data * 255).astype(np.uint8)
            image_with_lines = cv2.cvtColor(image_with_lines, cv2.COLOR_GRAY2BGR)
        height, width = image_with_lines.shape[:2]
        for line in self._lines:
            a, b, c = line
            if (b < 1e-6 and b > -1e-6):
                # Hay linea vertical
                y_vals = np.array([0, height - 1])
                x_vals = np.array([-c / a, -c / a])
            elif (a < 1e-6 and a > -1e-6):
                # Hay linea horizontal
                x_vals = np.array([0, width - 1])
                y_vals = np.array([-c / b, -c / b])
            else:
                # Calculamos dos puntos en los extremos de la imagen
                x_vals = np.array([0, width - 1])
                y_vals = (-a * x_vals - c) / b
                if (y_vals[0] < 0):
                    y_vals[0] = 0
                    x_vals[0] = (-b * y_vals[0] - c) / a
                elif (y_vals[0] >= height):
                    y_vals[0] = height - 1
                    x_vals[0] = (-b * y_vals[0] - c) / a
                if (y_vals[1] < 0):
                    y_vals[1] = 0
                    x_vals[1] = (-b * y_vals[1] - c) / a
                elif (y_vals[1] >= height):
                    y_vals[1] = height - 1
                    x_vals[1] = (-b * y_vals[1] - c) / a
            pt1 = (int(round(x_vals[0])), int(round(y_vals[0])))
            pt2 = (int(round(x_vals[1])), int(round(y_vals[1])))
            cv2.line(image_with_lines, pt1, pt2, color, 2)

        copy.data = image_with_lines
        return copy
    

    def paint_segments_on_image(self, color='blue'):
        copy = self._original_edge_image.clone()
        image_with_segments = copy.data
        if (copy.data.ndim == 2):
            # Convert boolean to uint8 first, then to BGR
            image_with_segments = (copy.data * 255).astype(np.uint8)
            image_with_segments = cv2.cvtColor(image_with_segments, cv2.COLOR_GRAY2BGR)
        height, width = image_with_segments.shape[:2]
        for segment in self._segments:
            (start_x, start_y), (end_x, end_y) = segment
            pt1 = (int(round(start_x)), int(round(start_y)))
            pt2 = (int(round(end_x)), int(round(end_y)))
            cv2.line(image_with_segments, pt1, pt2, color, 2)
        copy.data = image_with_segments
        return copy
    
    def paint_circles_on_image(self, color='green', base_image: Optional[Image] = None):
        if base_image is None:
            copy = self._original_edge_image.clone()
        else:
            copy = base_image.clone()
        image_with_circles = copy.data
        if (copy.data.ndim == 2):
            # Convert boolean to uint8 first, then to BGR
            image_with_circles = (copy.data * 255).astype(np.uint8)
            image_with_circles = cv2.cvtColor(image_with_circles, cv2.COLOR_GRAY2BGR)
        height, width = image_with_circles.shape[:2]
        for circle in self._circles:
            cx, cy, r = circle
            center = (int(round(cx)), int(round(cy)))
            radius = int(round(r))
            cv2.circle(image_with_circles, center, radius, color, 2)
        copy.data = image_with_circles
        return copy
