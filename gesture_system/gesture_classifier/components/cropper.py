import cv2
import mediapipe as mp
import os
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import math


class HandCropProcessor:
    """
    Processes images to detect, crop, and normalize hand regions for gesture classification.
    Uses dynamic padding to ensure consistent 256x256 output regardless of hand distance.
    """

    def __init__(
            self,
            output_size: Tuple[int, int] = (256, 256),
            min_padding_percent: float = 0.05,  # Minimum 5% padding
            max_padding_percent: float = 0.4,  # Maximum 40% padding
            target_hand_ratio: float = 0.7,  # Hand should occupy ~70% of final image
            debug_mode: bool = True,
            min_detection_confidence: float = 0.5,
            max_hand_ratio: float = 0.95  # If hand is >95% of image, consider it too large
    ):
        """
        Initialize the hand crop processor with dynamic padding.

        Args:
            output_size: Target size for normalized output (width, height)
            min_padding_percent: Minimum padding around hand
            max_padding_percent: Maximum padding around hand
            target_hand_ratio: Desired ratio of hand size to final image
            debug_mode: If True, print issues; if False, delete bad images
            min_detection_confidence: MediaPipe detection confidence threshold
            max_hand_ratio: Threshold for detecting hands that are too large
        """
        self.output_size = output_size
        self.min_padding_percent = min_padding_percent
        self.max_padding_percent = max_padding_percent
        self.target_hand_ratio = target_hand_ratio
        self.debug_mode = debug_mode
        self.max_hand_ratio = max_hand_ratio

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=10,
            min_detection_confidence=min_detection_confidence
        )

        # Statistics tracking
        self.stats = {
            'processed': 0,
            'no_hands_found': 0,
            'multiple_hands_found': 0,
            'hands_too_large': 0,
            'successful': 0,
            'avg_padding_used': []
        }

    def calculate_dynamic_padding(
            self,
            hand_bbox: Tuple[int, int, int, int],
            image_shape: Tuple[int, int, int]
    ) -> Tuple[float, str]:
        """
        Calculate dynamic padding based on hand size relative to image.

        Strategy:
        - Small hands (far away): More padding to reach target ratio
        - Large hands (close): Less padding, may need to zoom out or handle specially

        Args:
            hand_bbox: Raw hand bounding box (x_min, y_min, x_max, y_max)
            image_shape: Shape of the image (height, width, channels)

        Returns:
            Tuple of (padding_percent, reasoning_string)
        """
        h, w, _ = image_shape
        x_min, y_min, x_max, y_max = hand_bbox

        hand_width = x_max - x_min
        hand_height = y_max - y_min
        hand_area = hand_width * hand_height
        image_area = w * h

        # Calculate current hand ratio to image
        current_hand_ratio = hand_area / image_area

        # Calculate what the final crop area should be to achieve target ratio
        # If target_hand_ratio = 0.7, then crop_area = hand_area / 0.7
        target_crop_area = hand_area / self.target_hand_ratio
        target_crop_side = math.sqrt(target_crop_area)

        # Current hand dimensions
        current_hand_side = math.sqrt(hand_area)

        # Calculate required padding to reach target
        required_expansion = (target_crop_side - current_hand_side) / current_hand_side
        padding_percent = max(self.min_padding_percent,
                              min(self.max_padding_percent, required_expansion))

        # Determine reasoning
        if current_hand_ratio > self.max_hand_ratio:
            reasoning = f"HAND_TOO_LARGE (ratio: {current_hand_ratio:.2f})"
        elif current_hand_ratio < 0.1:
            reasoning = f"HAND_VERY_SMALL (ratio: {current_hand_ratio:.2f}, using max padding)"
            padding_percent = self.max_padding_percent
        elif current_hand_ratio > 0.6:
            reasoning = f"HAND_LARGE (ratio: {current_hand_ratio:.2f}, using min padding)"
            padding_percent = self.min_padding_percent
        else:
            reasoning = f"HAND_NORMAL (ratio: {current_hand_ratio:.2f}, dynamic padding)"

        return padding_percent, reasoning

    def get_hand_bounding_box(
            self,
            hand_landmarks,
            image_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calculate raw bounding box for a detected hand (without padding).
        """
        h, w, _ = image_shape

        # Get all landmark coordinates
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        # Find tight bounding box
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        return x_min, y_min, x_max, y_max

    def apply_padding_to_bbox(
            self,
            bbox: Tuple[int, int, int, int],
            padding_percent: float,
            image_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Apply calculated padding to bounding box with boundary checks.
        """
        h, w, _ = image_shape
        x_min, y_min, x_max, y_max = bbox

        width = x_max - x_min
        height = y_max - y_min

        pad_x = int(width * padding_percent)
        pad_y = int(height * padding_percent)

        # Apply padding with boundary checks
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        return x_min, y_min, x_max, y_max

    def handle_large_hand(
            self,
            image: np.ndarray,
            bbox: Tuple[int, int, int, int],
            image_name: str
    ) -> Optional[np.ndarray]:
        """
        Handle cases where the hand is too large for the image.

        Strategies:
        1. Use minimum padding and crop what we can
        2. Scale down the entire image first, then crop
        3. Crop to hand center with fixed size
        """
        h, w, _ = image.shape
        x_min, y_min, x_max, y_max = bbox

        if self.debug_mode:
            print(f"  → HANDLING LARGE HAND: {image_name}")
            print(f"    Hand size: {x_max - x_min}x{y_max - y_min}, Image size: {w}x{h}")

        # Strategy 1: Center crop with minimum viable area
        hand_center_x = (x_min + x_max) // 2
        hand_center_y = (y_min + y_max) // 2

        # Use the smaller dimension to ensure we stay within image bounds
        crop_size = min(w, h, max(x_max - x_min, y_max - y_min) * 1.1)
        half_crop = int(crop_size // 2)

        # Calculate crop bounds centered on hand
        crop_x_min = max(0, hand_center_x - half_crop)
        crop_y_min = max(0, hand_center_y - half_crop)
        crop_x_max = min(w, crop_x_min + crop_size)
        crop_y_max = min(h, crop_y_min + crop_size)

        # Adjust if we hit boundaries
        if crop_x_max - crop_x_min < crop_size:
            crop_x_min = max(0, crop_x_max - crop_size)
        if crop_y_max - crop_y_min < crop_size:
            crop_y_min = max(0, crop_y_max - crop_size)

        cropped = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        if cropped.size == 0:
            return None

        if self.debug_mode:
            print(f"    → Center cropped to: {cropped.shape}")

        return cropped

    def get_bounding_box_area(self, bbox: Tuple[int, int, int, int]) -> int:
        """Calculate area of bounding box in pixels."""
        x_min, y_min, x_max, y_max = bbox
        return (x_max - x_min) * (y_max - y_min)

    def find_largest_hand_with_analysis(
            self,
            results,
            image_shape: Tuple[int, int, int]
    ) -> Optional[Tuple[Tuple[int, int, int, int], float, str]]:
        """
        Find the largest hand and analyze its size characteristics.

        Returns:
            Tuple of (bbox, padding_percent, reasoning) or None if no hands
        """
        if not results.multi_hand_landmarks:
            return None

        num_hands = len(results.multi_hand_landmarks)

        if num_hands > 1:
            self.stats['multiple_hands_found'] += 1
            if self.debug_mode:
                print(f"  → Found {num_hands} hands, selecting largest")

        # Get all bounding boxes with analysis
        hand_data = []
        for hand_landmarks in results.multi_hand_landmarks:
            bbox = self.get_hand_bounding_box(hand_landmarks, image_shape)
            padding, reasoning = self.calculate_dynamic_padding(bbox, image_shape)
            area = self.get_bounding_box_area(bbox)
            hand_data.append((bbox, padding, reasoning, area))

        # Find largest by area
        largest_hand = max(hand_data, key=lambda x: x[3])
        bbox, padding, reasoning, area = largest_hand

        return bbox, padding, reasoning

    def crop_and_resize(
            self,
            image: np.ndarray,
            bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crop image to bounding box and resize to target size.
        """
        x_min, y_min, x_max, y_max = bbox
        cropped = image[y_min:y_max, x_min:x_max]

        if cropped.size == 0:
            raise ValueError("Cropped image is empty")

        # Resize to target size with high-quality interpolation
        resized = cv2.resize(cropped, self.output_size, interpolation=cv2.INTER_LANCZOS4)

        return resized

    def process_image(
            self,
            image_path: Path,
            output_path: Optional[Path] = None
    ) -> bool:
        """
        Process a single image with dynamic padding and large hand handling.
        """
        self.stats['processed'] += 1
        image_name = image_path.name

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            if self.debug_mode:
                print(f"❌ ERROR: Could not read image: {image_name}")
            return False

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = self.hands.process(image_rgb)

        # Find largest hand with analysis
        hand_analysis = self.find_largest_hand_with_analysis(results, image.shape)

        if hand_analysis is None:
            # No hand detected
            self.stats['no_hands_found'] += 1

            if self.debug_mode:
                print(f"⚠️  NO HAND DETECTED: {image_name}")
            else:
                os.remove(image_path)
                print(f"🗑️  DELETED (no hand): {image_name}")

            return False

        bbox, padding_percent, reasoning = hand_analysis

        if self.debug_mode:
            print(f"🔍 ANALYZING: {image_name}")
            print(f"  → {reasoning}")
            print(f"  → Using {padding_percent * 100:.1f}% padding")

        # Check if hand is too large
        if "HAND_TOO_LARGE" in reasoning:
            self.stats['hands_too_large'] += 1
            processed_image = self.handle_large_hand(image, bbox, image_name)
            if processed_image is None:
                if self.debug_mode:
                    print(f"❌ FAILED to handle large hand: {image_name}")
                return False
        else:
            # Normal processing with dynamic padding
            padded_bbox = self.apply_padding_to_bbox(bbox, padding_percent, image.shape)
            try:
                processed_image = self.crop_and_resize(image, padded_bbox)
            except ValueError as e:
                if self.debug_mode:
                    print(f"❌ CROP ERROR: {image_name} - {e}")
                return False

        # Final resize to ensure exact output size
        if processed_image.shape[:2] != self.output_size:
            processed_image = cv2.resize(processed_image, self.output_size,
                                         interpolation=cv2.INTER_LANCZOS4)

        # Save
        save_path = output_path if output_path else image_path
        cv2.imwrite(str(save_path), processed_image)

        # Track statistics
        self.stats['successful'] += 1
        self.stats['avg_padding_used'].append(padding_percent)

        if self.debug_mode:
            print(f"✅ SUCCESS: {image_name} → {processed_image.shape}")

        return True

    def process_directory(
            self,
            input_dir: Path,
            output_dir: Optional[Path] = None,
            file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    ):
        """
        Process all images in a directory with enhanced reporting.
        """
        input_dir = Path(input_dir)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in file_extensions
        ]

        print(f"\n{'=' * 70}")
        print(f"HAND CROP PROCESSOR - DYNAMIC PADDING MODE")
        print(f"{'=' * 70}")
        print(f"Images to process: {len(image_files)}")
        print(f"Debug Mode: {'ON' if self.debug_mode else 'OFF'}")
        print(f"Output Size: {self.output_size}")
        print(f"Target Hand Ratio: {self.target_hand_ratio * 100:.1f}%")
        print(f"Padding Range: {self.min_padding_percent * 100:.1f}% - {self.max_padding_percent * 100:.1f}%")
        print(f"{'=' * 70}\n")

        # Process each image
        for image_file in image_files:
            output_path = output_dir / image_file.name if output_dir else None
            self.process_image(image_file, output_path)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print detailed processing statistics."""
        avg_padding = np.mean(self.stats['avg_padding_used']) if self.stats['avg_padding_used'] else 0

        print(f"\n{'=' * 70}")
        print("PROCESSING SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total Processed:        {self.stats['processed']}")
        print(f"Successfully Cropped:   {self.stats['successful']}")
        print(f"No Hands Found:         {self.stats['no_hands_found']}")
        print(f"Multiple Hands Found:   {self.stats['multiple_hands_found']}")
        print(f"Hands Too Large:        {self.stats['hands_too_large']}")
        print(f"Success Rate:           {self.stats['successful'] / max(1, self.stats['processed']) * 100:.1f}%")
        print(f"Average Padding Used:   {avg_padding * 100:.1f}%")
        print(f"{'=' * 70}\n")

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "mixed_gesture_test_dataset"
    OUTPUT_DIR = "processed_gesture_images"  # Set to None to overwrite originals

    # Initialize processor with dynamic padding
    processor = HandCropProcessor(
        output_size=(256, 256),
        min_padding_percent=0.05,  # 5% minimum
        max_padding_percent=0.4,  # 40% maximum
        target_hand_ratio=0.7,  # Hand should be ~70% of final image
        debug_mode=True,  # KEEP TRUE DURING DEVELOPMENT
        min_detection_confidence=0.5,
        max_hand_ratio=0.95  # Flag hands >95% of image as too large
    )

    # Process all images
    processor.process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )
