"""
Camera configuration settings.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CameraConfig:
    """Configuration for camera capture."""
    
    # Camera settings
    camera_index: int = 0  # Which camera to use (0 = default/built-in)
    
    # Resolution
    width: int = 640
    height: int = 480
    
    # Performance
    target_fps: int = 30  # Target frame rate
    
    # Image processing
    mirror: bool = True  # Flip horizontally for natural view
    auto_resize: bool = False  # Resize frames before returning
    resize_width: Optional[int] = None  # Target width if auto_resize=True
    resize_height: Optional[int] = None  # Target height if auto_resize=True
    
    # Error handling
    max_consecutive_failures: int = 10  # Max failed reads before shutdown
    retry_on_disconnect: bool = True  # Try to reconnect if camera disconnects
    retry_delay: float = 1.0  # Seconds to wait before retry
    
    # Debug
    show_fps: bool = False  # Print FPS info
    verbose: bool = False  # Print detailed info
    
    def __post_init__(self):
        """Validate configuration."""
        if self.camera_index < 0:
            raise ValueError("camera_index must be >= 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
