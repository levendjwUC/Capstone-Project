"""
Unified gesture preprocessor combining all preprocessing stages.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import time

from .config import PreprocessingConfig
from .components.cropper import HandCropProcessor
from .components.normalizer import HandOrientationNormalizer
from .components.aligner import HandCenterAligner
from .components.binary_mask import HandBinaryMaskProcessor
from .components.distance_transform import HandDistanceTransform
from .components.edge_enhancer import HandEdgeEnhancer
from .components.skeleton import HandSkeletonExtractor


class GesturePreprocessor:
    """
    Unified gesture preprocessing pipeline.
    Combines all preprocessing stages with configurable optional modules.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: PreprocessingConfig object (uses defaults if None)
        """
        self.config = config or PreprocessingConfig()
        
        # Initialize primary pipeline modules
        self._init_primary_modules()
        
        # Initialize optional modules
        self._init_optional_modules()
    
    def _init_primary_modules(self):
        """Initialize the 4 core preprocessing modules."""
        # 1. Hand Cropper
        self.cropper = HandCropProcessor(
            output_size=self.config.crop_size,
            min_padding_percent=self.config.crop_min_padding,
            max_padding_percent=self.config.crop_max_padding,
            target_hand_ratio=self.config.crop_target_hand_ratio,
            debug_mode=False
        )
        
        # 2. Orientation Normalizer
        self.orientation_normalizer = HandOrientationNormalizer(
            target_angle=self.config.orientation_target_angle,
            angle_calculation_method=self.config.orientation_method,
            smoothing_enabled=self.config.orientation_smoothing,
            is_mirrored=self.config.orientation_is_mirrored,
            debug_mode=False
        )
        
        # 3. Center Aligner
        self.center_aligner = HandCenterAligner(
            center_method=self.config.alignment_method,
            preserve_aspect_ratio=self.config.alignment_preserve_aspect,
            background_color=self.config.alignment_bg_color,
            debug_mode=False
        )
        
        # 4. Binary Mask Processor
        self.binary_mask_processor = HandBinaryMaskProcessor(
            contour_method=self.config.binary_mask_method,
            edge_sensitivity=self.config.binary_mask_edge_sensitivity,
            morphology_size=self.config.binary_mask_morphology_size,
            debug_mode=False
        )
    
    def _init_optional_modules(self):
        """Initialize optional preprocessing modules."""
        self.optional_processors = {}
        
        if "distance_transform" in self.config.optional_modules:
            metric_map = {
                "L1": cv2.DIST_L1,
                "L2": cv2.DIST_L2,
                "CHESSBOARD": cv2.DIST_C
            }
            metric = metric_map.get(self.config.distance_transform_metric, cv2.DIST_L2)
            
            self.optional_processors["distance_transform"] = HandDistanceTransform(
                distance_type=metric,
                visualization_mode=self.config.distance_transform_mode,
                colormap=self.config.distance_transform_colormap,
                apply_smoothing=self.config.distance_transform_smoothing,
                debug_mode=False
            )
        
        if "edge_enhancement" in self.config.optional_modules:
            self.optional_processors["edge_enhancement"] = HandEdgeEnhancer(
                enhancement_method=self.config.edge_enhancement_method,
                edge_thickness=self.config.edge_enhancement_thickness,
                edge_intensity=self.config.edge_enhancement_intensity,
                preserve_original=self.config.edge_enhancement_preserve_original,
                debug_mode=False
            )
        
        if "skeleton_extraction" in self.config.optional_modules:
            self.optional_processors["skeleton_extraction"] = HandSkeletonExtractor(
                skeletonization_method=self.config.skeleton_method,
                skeleton_thickness=self.config.skeleton_thickness,
                enhance_junctions=self.config.skeleton_enhance_junctions,
                apply_pruning=self.config.skeleton_apply_pruning,
                debug_mode=False
            )
    
    def process_image(
        self,
        image_path: Path,
        output_path: Path,
        save_intermediate: Optional[bool] = None
    ) -> Tuple[bool, Dict]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            output_path: Path for output image
            save_intermediate: Whether to save intermediate results
        
        Returns:
            Tuple of (success, stage_results)
        """
        start_time = time.time()
        stage_results = {
            'image_name': image_path.name,
            'success': False,
            'stages': {},
            'errors': [],
            'total_time_ms': 0
        }
        
        save_intermediate = save_intermediate or self.config.save_intermediate
        intermediate_dir = output_path.parent
        
        try:
            # STAGE 1: Crop
            temp_crop_path = intermediate_dir / f"{image_path.stem}_01_cropped.png"
            success_1 = self.cropper.process_image(image_path, temp_crop_path)
            
            if not success_1:
                stage_results['errors'].append("Cropping failed")
                return False, stage_results
            
            # STAGE 2: Orientation
            temp_orient_path = intermediate_dir / f"{image_path.stem}_02_oriented.png"
            image_1 = cv2.imread(str(temp_crop_path))
            cv2.imwrite(str(temp_orient_path), image_1)
            
            success_2, rotation_2, _ = self.orientation_normalizer.process_image(
                temp_orient_path, temp_orient_path
            )
            
            if not success_2:
                stage_results['errors'].append("Orientation normalization failed")
                return False, stage_results
            
            # STAGE 3: Alignment
            temp_align_path = intermediate_dir / f"{image_path.stem}_03_centered.png"
            image_2 = cv2.imread(str(temp_orient_path))
            cv2.imwrite(str(temp_align_path), image_2)
            
            success_3, translation_3, _ = self.center_aligner.process_image(
                temp_align_path, temp_align_path
            )
            
            if not success_3:
                stage_results['errors'].append("Center alignment failed")
                return False, stage_results
            
            # STAGE 4: Binary Mask
            image_3 = cv2.imread(str(temp_align_path))
            temp_binary_input = intermediate_dir / f"{image_path.stem}_temp_for_binary.png"
            cv2.imwrite(str(temp_binary_input), image_3)
            
            success_4 = self.binary_mask_processor.process_image(
                temp_binary_input, output_path
            )
            
            if not success_4:
                stage_results['errors'].append("Binary mask creation failed")
                return False, stage_results
            
            # Clean up temp files if not saving intermediate
            if not save_intermediate:
                temp_crop_path.unlink(missing_ok=True)
                temp_orient_path.unlink(missing_ok=True)
                temp_align_path.unlink(missing_ok=True)
            temp_binary_input.unlink(missing_ok=True)
            
            # Load binary mask for optional processing
            current_image = cv2.imread(str(output_path))
            
            # OPTIONAL STAGES
            if "distance_transform" in self.config.optional_modules:
                processor = self.optional_processors["distance_transform"]
                temp_path = intermediate_dir / f"{image_path.stem}_05_distance.png"
                cv2.imwrite(str(temp_path), current_image)
                success, _, _ = processor.process_image(temp_path, output_path)
                if success:
                    current_image = cv2.imread(str(output_path))
                if not save_intermediate:
                    temp_path.unlink(missing_ok=True)
            
            if "edge_enhancement" in self.config.optional_modules:
                processor = self.optional_processors["edge_enhancement"]
                temp_path = intermediate_dir / f"{image_path.stem}_06_edges.png"
                cv2.imwrite(str(temp_path), current_image)
                success, _, _ = processor.process_image(temp_path, output_path)
                if success:
                    current_image = cv2.imread(str(output_path))
                if not save_intermediate:
                    temp_path.unlink(missing_ok=True)
            
            if "skeleton_extraction" in self.config.optional_modules:
                processor = self.optional_processors["skeleton_extraction"]
                temp_path = intermediate_dir / f"{image_path.stem}_07_skeleton.png"
                cv2.imwrite(str(temp_path), current_image)
                success, _, _ = processor.process_image(temp_path, output_path)
                if not save_intermediate:
                    temp_path.unlink(missing_ok=True)
            
            stage_results['success'] = True
            
        except Exception as e:
            stage_results['errors'].append(f"Exception: {str(e)}")
            return False, stage_results
        
        finally:
            stage_results['total_time_ms'] = (time.time() - start_time) * 1000
        
        return stage_results['success'], stage_results
