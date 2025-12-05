"""
Utility functions for camera module.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


def flip_frame(frame: np.ndarray, horizontal: bool = True, vertical: bool = False) -> np.ndarray:
    """
    Flip frame horizontally and/or vertically.
    
    Args:
        frame: Input frame
        horizontal: Flip horizontally (mirror)
        vertical: Flip vertically
    
    Returns:
        Flipped frame
    """
    if horizontal and vertical:
        return cv2.flip(frame, -1)
    elif horizontal:
        return cv2.flip(frame, 1)
    elif vertical:
        return cv2.flip(frame, 0)
    else:
        return frame


def resize_frame(
    frame: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize frame to target dimensions.
    
    Args:
        frame: Input frame
        width: Target width (None = keep original)
        height: Target height (None = keep original)
        maintain_aspect: Maintain aspect ratio
    
    Returns:
        Resized frame
    """
    if width is None and height is None:
        return frame
    
    h, w = frame.shape[:2]
    
    if maintain_aspect:
        if width is not None and height is None:
            # Calculate height based on width
            aspect = h / w
            height = int(width * aspect)
        elif height is not None and width is None:
            # Calculate width based on height
            aspect = w / h
            width = int(height * aspect)
    else:
        if width is None:
            width = w
        if height is None:
            height = h
    
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def validate_frame(frame: Optional[np.ndarray]) -> bool:
    """
    Validate that frame is usable.
    
    Args:
        frame: Frame to validate
    
    Returns:
        True if frame is valid, False otherwise
    """
    if frame is None:
        return False
    if not isinstance(frame, np.ndarray):
        return False
    if frame.size == 0:
        return False
    if len(frame.shape) not in [2, 3]:
        return False
    return True
