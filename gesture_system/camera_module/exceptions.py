"""
Custom exceptions for camera module.
"""


class CameraError(Exception):
    """Base exception for camera-related errors."""
    pass


class CameraNotFoundError(CameraError):
    """Raised when camera cannot be found or opened."""
    pass


class CameraConnectionError(CameraError):
    """Raised when camera connection is lost during operation."""
    pass


class CameraConfigurationError(CameraError):
    """Raised when camera configuration is invalid."""
    pass
