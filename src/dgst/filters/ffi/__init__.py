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
        raise RuntimeError(f"Makefile execution failed: {e}") from e
else:
    raise FileNotFoundError(f"Makefile not found in {makefile_dir}")

from .wrapper import (  # noqa: E402
    box_filter,
    canny_edge_detection,
    gaussian_filter,
    kannala_brandt_map_points_to_undistorted,
    kannala_brandt_undistort,
    phase_congruency,
    ransac_circle_fitting,
    ransac_line_fitting,
    threshold_filter,
)

__all__ = [
    "box_filter",
    "gaussian_filter",
    "canny_edge_detection",
    "kannala_brandt_undistort",
    "kannala_brandt_map_points_to_undistorted",
    "phase_congruency",
    "threshold_filter",
    "ransac_line_fitting",
    "ransac_circle_fitting",
]
