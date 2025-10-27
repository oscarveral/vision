"""mi_proyecto package init."""

__version__ = "0.1.0"

import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
