"""
Main Camera class for real-time frame capture.
"""

import cv2
import numpy as np
import time
from typing import Optional, Generator
from pathlib import Path

from .config import CameraConfig
from .exceptions import CameraNotFoundError, CameraConnectionError
from .utils import flip_frame, resize_frame, validate_frame


class Camera:
    """
    Camera capture class with error handling and frame processing.
    
    Usage:
        camera = Camera()
        for frame in camera.stream():
            # Process frame
            pass
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        """
        Initialize camera.
        
        Args:
            config: CameraConfig object (uses defaults if None)
        """
        self.config = config or CameraConfig()
        self.cap = None
        self.is_running = False
        self.consecutive_failures = 0
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = None
        self.fps = 0.0
        
        # Initialize camera
        self._open_camera()
    
    def _open_camera(self) -> None:
        """
        Open camera device.
        
        Raises:
            CameraNotFoundError: If camera cannot be opened
        """
        if self.config.verbose:
            print(f"Opening camera {self.config.camera_index}...")
        
        self.cap = cv2.VideoCapture(self.config.camera_index)
        
        if not self.cap.isOpened():
            raise CameraNotFoundError(
                f"Could not open camera {self.config.camera_index}. "
                f"Make sure camera is connected and not in use by another application."
            )
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        
        # Set FPS (may not work on all cameras)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        
        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        if self.config.verbose:
            print(f"Camera opened successfully")
            print(f"  Resolution: {actual_width}x{actual_height} (requested: {self.config.width}x{self.config.height})")
            print(f"  FPS: {actual_fps} (requested: {self.config.target_fps})")
        
        self.is_running = True
        self.start_time = time.time()
    
    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to camera.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        if self.config.verbose:
            print("Attempting to reconnect to camera...")
        
        try:
            self._close_camera()
            time.sleep(self.config.retry_delay)
            self._open_camera()
            self.consecutive_failures = 0
            
            if self.config.verbose:
                print("Reconnection successful")
            
            return True
            
        except CameraNotFoundError:
            if self.config.verbose:
                print("Reconnection failed")
            return False
    
    def _close_camera(self) -> None:
        """Close camera device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get a single frame from camera.
        
        Returns:
            Frame as numpy array (BGR format), or None if read fails
        
        Raises:
            CameraConnectionError: If too many consecutive failures occur
        """
        if not self.is_running:
            return None
        
        ret, frame = self.cap.read()
        
        if not ret or not validate_frame(frame):
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= self.config.max_consecutive_failures:
                if self.config.retry_on_disconnect:
                    if not self._reconnect():
                        raise CameraConnectionError(
                            f"Lost connection to camera after {self.consecutive_failures} failures"
                        )
                else:
                    raise CameraConnectionError(
                        f"Too many consecutive frame read failures ({self.consecutive_failures})"
                    )
            
            return None
        
        # Reset failure counter on successful read
        self.consecutive_failures = 0
        
        # Apply mirror if configured
        if self.config.mirror:
            frame = flip_frame(frame, horizontal=True)
        
        # Apply resize if configured
        if self.config.auto_resize:
            frame = resize_frame(
                frame,
                width=self.config.resize_width,
                height=self.config.resize_height
            )
        
        # Update FPS tracking
        self._update_fps()
        
        return frame
    
    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        Stream frames from camera as generator.
        
        Yields:
            Frames as numpy arrays (BGR format)
        
        Example:
            for frame in camera.stream():
                process(frame)
        """
        while self.is_running:
            frame = self.get_frame()
            if frame is not None:
                yield frame
    
    def _update_fps(self) -> None:
        """Update FPS calculation."""
        self.frame_count += 1
        
        if self.config.show_fps and self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed
            print(f"Camera FPS: {self.fps:.1f}")
    
    def get_fps(self) -> float:
        """
        Get current FPS.
        
        Returns:
            Current frames per second
        """
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def get_info(self) -> dict:
        """
        Get camera information.
        
        Returns:
            Dictionary with camera info
        """
        if not self.is_running or self.cap is None:
            return {}
        
        return {
            "camera_index": self.config.camera_index,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self.cap.get(cv2.CAP_PROP_FPS)),
            "actual_fps": self.get_fps(),
            "is_running": self.is_running,
            "frame_count": self.frame_count
        }
    
    def stop(self) -> None:
        """Stop camera and release resources."""
        if self.config.verbose:
            print("Stopping camera...")
        
        self._close_camera()
        
        if self.config.verbose:
            print(f"Camera stopped. Total frames captured: {self.frame_count}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
