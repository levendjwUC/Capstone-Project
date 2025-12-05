"""
Dry run configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DryRunConfig:
    """Configuration for dry run test."""
    
    # Classification settings
    classification_interval: float = 2.0  # Seconds between classifications
    confidence_threshold: float = 0.5     # Minimum confidence to report
    
    # Model settings
    model_path: str = "best_model.keras"
    class_names: list = None  # None = use defaults
    
    # Camera settings
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    mirror: bool = True
    
    # Output settings
    verbose: bool = True          # Show system info at startup
    show_all_confidences: bool = False  # Show all class confidences
    
    # Runtime settings
    max_classifications: Optional[int] = None  # None = run until Ctrl+C
    max_runtime: Optional[float] = None        # None = run until Ctrl+C
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['five', 'four', 'one', 'three', 'two', 'zero']
