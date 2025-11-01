"""Foreign Function Interface for filters."""

import os
import subprocess

# Run the Makefile to compile the C library.
makefile_dir = os.path.dirname(__file__)
makefile_path = os.path.join(makefile_dir, "Makefile")
# Only run make if the Makefile exists.
if os.path.exists(makefile_path):
    try:
        subprocess.run(["make", "--quiet", "-C", makefile_dir], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Makefile execution failed: {e}")
else:
    raise FileNotFoundError(f"Makefile not found in {makefile_dir}")

<<<<<<< HEAD
from .wrapper import box_filter, gaussian_filter, canny_edge_detection, kannala_brandt_undistort, kannala_brandt_map_points_to_undistorted, phase_congruency, threshold_filter, ransac_line_fitting
=======
from .wrapper import (
    box_filter,
    gaussian_filter,
    canny_edge_detection,
    kannala_brandt_undistort,
    kannala_brandt_map_points_to_undistorted,
    phase_congruency,
    threshold_filter,
    ransac_line_fitting,
    ransac_circle_fitting,
)

>>>>>>> main
__all__ = [
    "box_filter",
    "gaussian_filter",
    "canny_edge_detection",
    "kannala_brandt_undistort",
    "kannala_brandt_map_points_to_undistorted",
    "phase_congruency",
    "threshold_filter",
    "ransac_line_fitting",
<<<<<<< HEAD
]
=======
    "ransac_circle_fitting",
]
>>>>>>> main
