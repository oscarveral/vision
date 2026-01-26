"""mi_proyecto package init."""

__version__ = "0.1.0"

import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

from .detection import dectect_board, PATTERN_SIZE
from .chess import ChessPiece, Chessboard

__all__ = ["dectect_board", "PATTERN_SIZE", "ChessPiece", "Chessboard"]