"""
Camera Module
Real-time camera capture for hand gesture recognition.
"""

from .camera import Camera
from .config import CameraConfig
from .exceptions import CameraError, CameraNotFoundError, CameraConnectionError

__version__ = "1.0.0"
__all__ = ["Camera", "CameraConfig", "CameraError", "CameraNotFoundError", "CameraConnectionError"]
