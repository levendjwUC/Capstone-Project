"""
Main GestureClassifier class.
Handles preprocessing and classification of hand gestures.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import load_model

from .preprocessor import GesturePreprocessor
from .config import PreprocessingConfig


class GestureClassifier:
    """
    Hand gesture classifier with preprocessing pipeline.
    
    Usage:
        classifier = GestureClassifier(model_path="model.keras")
        gesture, confidence = classifier.classify(frame)
    """
    
    def __init__(
        self,
        model_path: str,
        class_names: Optional[list] = None,
        use_distance_transform: bool = False,
        use_edge_enhancement: bool = False,
        use_skeleton: bool = False,
        preprocessing_config: Optional[PreprocessingConfig] = None
    ):
        """
        Initialize gesture classifier.
        
        Args:
            model_path: Path to trained Keras model
            class_names: List of gesture class names (default: ['five', 'four', 'one', 'three', 'two', 'zero'])
            use_distance_transform: Enable distance transform preprocessing
            use_edge_enhancement: Enable edge enhancement preprocessing
            use_skeleton: Enable skeleton extraction preprocessing
            preprocessing_config: Custom preprocessing config (optional)
        """
        # Load model
        self.model = load_model(model_path)
        
        # Set class names
        self.class_names = class_names or ['five', 'four', 'one', 'three', 'two', 'zero']
        
        # Configure preprocessing
        if preprocessing_config is None:
            preprocessing_config = PreprocessingConfig(
                debug_mode=False,
                save_intermediate=False
            )
        
        # Set optional modules
        optional_modules = []
        if use_distance_transform:
            optional_modules.append("distance_transform")
        if use_edge_enhancement:
            optional_modules.append("edge_enhancement")
        if use_skeleton:
            optional_modules.append("skeleton_extraction")
        
        preprocessing_config.optional_modules = optional_modules
        
        # Initialize preprocessor
        self.preprocessor = GesturePreprocessor(config=preprocessing_config)
    
    def classify(
        self,
        image: np.ndarray,
        return_all_confidences: bool = False
    ) -> Tuple[Optional[str], float, Optional[dict]]:
        """
        Classify a hand gesture from an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            return_all_confidences: If True, return confidence for all classes
        
        Returns:
            Tuple of (gesture_name, confidence, all_confidences_dict)
            Returns (None, 0.0, None) if preprocessing fails
        """
        # Preprocess image
        processed = self._preprocess_image(image)
        
        if processed is None:
            return None, 0.0, None
        
        # Make prediction
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get best prediction
        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx])
        gesture_name = self.class_names[class_idx]
        
        # Prepare all confidences if requested
        all_confidences = None
        if return_all_confidences:
            all_confidences = {
                name: float(conf)
                for name, conf in zip(self.class_names, predictions)
            }
        
        return gesture_name, confidence, all_confidences
    
    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess image through the pipeline.
        
        Args:
            image: Input image as numpy array
        
        Returns:
            Preprocessed image ready for model, or None if preprocessing fails
        """
        # Save to temp file (preprocessor expects file paths)
        temp_input = Path("_temp_input.jpg")
        temp_output = Path("_temp_output.png")
        
        try:
            # Save input image
            cv2.imwrite(str(temp_input), image)
            
            # Process through pipeline
            success, results = self.preprocessor.process_image(
                temp_input,
                temp_output,
                save_intermediate=False
            )
            
            if not success or not temp_output.exists():
                return None
            
            # Read processed image
            processed = cv2.imread(str(temp_output))
            
            if processed is None:
                return None
            
            # Normalize for model input
            processed = processed / 255.0
            processed = np.expand_dims(processed, axis=0)
            
            return processed
            
        except Exception as e:
            return None
            
        finally:
            # Clean up temp files
            temp_input.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)
    
    def classify_batch(
        self,
        images: list
    ) -> list:
        """
        Classify multiple images in batch.
        
        Args:
            images: List of input images as numpy arrays
        
        Returns:
            List of (gesture_name, confidence) tuples
        """
        results = []
        for image in images:
            gesture, confidence, _ = self.classify(image)
            results.append((gesture, confidence))
        return results
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "num_classes": len(self.class_names),
            "class_names": self.class_names
        }
