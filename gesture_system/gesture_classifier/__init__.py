"""
Gesture Classifier Package
Clean preprocessing + classification module for hand gesture recognition.
"""

from .classifier import GestureClassifier
from .config import PreprocessingConfig

__version__ = "1.0.0"
__all__ = ["GestureClassifier", "PreprocessingConfig"]
