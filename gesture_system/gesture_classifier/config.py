"""
Configuration for preprocessing pipeline.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # Primary pipeline settings (always executed)
    crop_size: Tuple[int, int] = (256, 256)
    crop_min_padding: float = 0.05
    crop_max_padding: float = 0.4
    crop_target_hand_ratio: float = 0.7
    
    orientation_target_angle: float = -90.0
    orientation_method: str = "wrist_to_middle"
    orientation_smoothing: bool = False
    orientation_is_mirrored: bool = False
    
    alignment_method: str = "center_of_mass"
    alignment_preserve_aspect: bool = True
    alignment_bg_color: Tuple[int, int, int] = (0, 0, 0)
    
    binary_mask_method: str = "edge_based"
    binary_mask_edge_sensitivity: float = 0.3
    binary_mask_morphology_size: int = 3
    
    # Optional modules
    optional_modules: List[str] = None
    use_all_optionals: bool = False
    
    # Optional: Distance Transform settings
    distance_transform_metric: str = "L2"
    distance_transform_mode: str = "normalized"
    distance_transform_colormap: int = 2  # cv2.COLORMAP_JET
    distance_transform_smoothing: bool = True
    
    # Optional: Edge Enhancement settings
    edge_enhancement_method: str = "canny_overlay"
    edge_enhancement_thickness: int = 2
    edge_enhancement_intensity: float = 1.0
    edge_enhancement_preserve_original: bool = True
    
    # Optional: Skeleton Extraction settings
    skeleton_method: str = "zhang_suen"
    skeleton_thickness: int = 2
    skeleton_enhance_junctions: bool = True
    skeleton_apply_pruning: bool = False
    
    # General settings
    debug_mode: bool = False
    save_intermediate: bool = False
    num_workers: int = 1
    
    def __post_init__(self):
        """Initialize optional_modules if not provided."""
        if self.optional_modules is None:
            self.optional_modules = []
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
