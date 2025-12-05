import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from datetime import datetime
import json
import math


class HandOrientationNormalizer:
    """
    Normalizes hand orientation to consistent upright position.
    Uses wrist-to-middle-finger-base vector for orientation calculation.
    Handles both mirrored (live camera) and non-mirrored (file) images.
    """

    def __init__(
            self,
            target_angle: float = -90.0,  # -90 = pointing up, 0 = pointing right
            angle_calculation_method: str = "wrist_to_middle",
            # "wrist_to_middle", "wrist_to_fingertips", "palm_orientation"
            smoothing_enabled: bool = True,
            max_rotation_angle: float = 180.0,  # Maximum rotation to apply (safety)
            interpolation_method: int = cv2.INTER_LINEAR,
            is_mirrored: bool = False,  # Set True for mirrored camera feeds
            debug_mode: bool = True
    ):
        """
        Initialize orientation normalizer.

        Args:
            target_angle: Target orientation in degrees (-90 = up, 0 = right, 90 = down)
            angle_calculation_method: Method for determining hand orientation
            smoothing_enabled: Apply smoothing to rotation to reduce noise
            max_rotation_angle: Maximum rotation angle for safety
            interpolation_method: CV2 interpolation method
            is_mirrored: True if processing mirrored/flipped camera feed
            debug_mode: Print detailed processing info
        """
        self.target_angle = target_angle
        self.angle_calculation_method = angle_calculation_method
        self.smoothing_enabled = smoothing_enabled
        self.max_rotation_angle = max_rotation_angle
        self.interpolation_method = interpolation_method
        self.is_mirrored = is_mirrored
        self.debug_mode = debug_mode

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

        # Key landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        self.MIDDLE_MCP = 9  # Base of middle finger
        self.INDEX_MCP = 5  # Base of index finger
        self.RING_MCP = 13  # Base of ring finger
        self.PINKY_MCP = 17  # Base of pinky finger

        # For palm orientation calculation
        self.THUMB_CMC = 1  # Thumb base
        self.PINKY_MCP = 17  # Pinky base

        # Statistics
        self.stats = {
            'processed': 0,
            'successful': 0,
            'no_hands_found': 0,
            'excessive_rotation': 0,
            'avg_rotation_applied': [],
            'rotation_range': {'min': float('inf'), 'max': float('-inf')},
            'method_failures': 0,
            'errors': 0
        }

        # Smoothing buffer for live processing
        self.angle_history = []
        self.history_size = 5

    def calculate_wrist_to_middle_angle(
            self,
            hand_landmarks,
            image_shape: Tuple[int, int, int]
    ) -> float:
        """
        Calculate orientation using wrist to middle finger base vector.
        Most stable method for general hand poses.
        """
        h, w, _ = image_shape

        # Get key points
        wrist = hand_landmarks.landmark[self.WRIST]
        middle_mcp = hand_landmarks.landmark[self.MIDDLE_MCP]

        # Convert to pixel coordinates
        wrist_x, wrist_y = wrist.x * w, wrist.y * h
        middle_x, middle_y = middle_mcp.x * w, middle_mcp.y * h

        # Calculate vector from wrist to middle finger base
        dx = middle_x - wrist_x
        dy = middle_y - wrist_y

        # Handle edge case
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        # Calculate angle in degrees
        angle = math.degrees(math.atan2(dy, dx))

        # IMPORTANT: If image is mirrored (like in camera view),
        # the X-axis is flipped, so we need to invert the horizontal component
        # This is equivalent to negating the angle and reflecting across Y-axis
        if self.is_mirrored:
            # Mirror the angle: angle -> (180 - angle) for positive, (-180 - angle) for negative
            angle = 180.0 - angle if angle > 0 else -180.0 - angle

        if self.debug_mode and self.is_mirrored:
            print(f"  → Mirrored mode: angle adjusted for horizontal flip")

        return angle

    def calculate_wrist_to_fingertips_angle(
            self,
            hand_landmarks,
            image_shape: Tuple[int, int, int]
    ) -> float:
        """
        Calculate orientation using wrist to average fingertips vector.
        More stable for gestures where middle finger might be bent.
        """
        h, w, _ = image_shape

        # Get wrist
        wrist = hand_landmarks.landmark[self.WRIST]
        wrist_x, wrist_y = wrist.x * w, wrist.y * h

        # Get fingertips (excluding thumb as it's often at different angle)
        fingertips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]

        avg_x = 0
        avg_y = 0
        for tip_idx in fingertips:
            tip = hand_landmarks.landmark[tip_idx]
            avg_x += tip.x * w
            avg_y += tip.y * h

        avg_x /= len(fingertips)
        avg_y /= len(fingertips)

        # Calculate vector
        dx = avg_x - wrist_x
        dy = avg_y - wrist_y

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        angle = math.degrees(math.atan2(dy, dx))

        # Account for mirroring
        if self.is_mirrored:
            angle = 180.0 - angle if angle > 0 else -180.0 - angle

        return angle

    def calculate_palm_orientation_angle(
            self,
            hand_landmarks,
            image_shape: Tuple[int, int, int]
    ) -> float:
        """
        Calculate orientation using palm orientation (thumb to pinky base).
        Good for closed fist or when fingers are not clearly separated.
        """
        h, w, _ = image_shape

        # Get wrist (most stable reference point)
        wrist = hand_landmarks.landmark[self.WRIST]
        wrist_x, wrist_y = wrist.x * w, wrist.y * h

        # Get middle of palm (average of finger bases)
        palm_landmarks = [self.INDEX_MCP, self.MIDDLE_MCP, self.RING_MCP, self.PINKY_MCP]
        palm_x = sum(hand_landmarks.landmark[i].x * w for i in palm_landmarks) / len(palm_landmarks)
        palm_y = sum(hand_landmarks.landmark[i].y * h for i in palm_landmarks) / len(palm_landmarks)

        # Calculate vector from wrist to palm center
        dx = palm_x - wrist_x
        dy = palm_y - wrist_y

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        angle = math.degrees(math.atan2(dy, dx))

        # Account for mirroring
        if self.is_mirrored:
            angle = 180.0 - angle if angle > 0 else -180.0 - angle

        return angle

    def get_hand_orientation_angle(
            self,
            hand_landmarks,
            image_shape: Tuple[int, int, int]
    ) -> Optional[float]:
        """
        Calculate hand orientation angle using specified method.
        """
        try:
            if self.angle_calculation_method == "wrist_to_middle":
                angle = self.calculate_wrist_to_middle_angle(hand_landmarks, image_shape)
            elif self.angle_calculation_method == "wrist_to_fingertips":
                angle = self.calculate_wrist_to_fingertips_angle(hand_landmarks, image_shape)
            elif self.angle_calculation_method == "palm_orientation":
                angle = self.calculate_palm_orientation_angle(hand_landmarks, image_shape)
            else:
                # Default to wrist_to_middle
                angle = self.calculate_wrist_to_middle_angle(hand_landmarks, image_shape)

            return angle

        except Exception as e:
            if self.debug_mode:
                print(f"  ❌ Angle calculation failed: {e}")
            self.stats['method_failures'] += 1
            return None

    def smooth_angle(self, angle: float) -> float:
        """
        Apply smoothing to angle using history buffer.
        Reduces jitter in live processing.
        """
        if not self.smoothing_enabled:
            return angle

        # Add to history
        self.angle_history.append(angle)

        # Keep only recent history
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)

        # Handle angle wraparound for averaging
        # Convert to unit vectors, average, then back to angle
        cos_sum = sum(math.cos(math.radians(a)) for a in self.angle_history)
        sin_sum = sum(math.sin(math.radians(a)) for a in self.angle_history)

        avg_cos = cos_sum / len(self.angle_history)
        avg_sin = sin_sum / len(self.angle_history)

        smoothed_angle = math.degrees(math.atan2(avg_sin, avg_cos))

        return smoothed_angle

    def calculate_rotation_needed(self, current_angle: float) -> float:
        """
        Calculate rotation needed to reach target angle.
        Takes shortest rotation path.
        """
        # Calculate difference
        diff = self.target_angle - current_angle

        # Normalize to [-180, 180] range (shortest path)
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        return diff

    def rotate_image_around_center(
            self,
            image: np.ndarray,
            angle: float,
            center: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Rotate image around specified center point.
        """
        h, w = image.shape[:2]

        if center is None:
            center = (w // 2, h // 2)

        # Get rotation matrix
        # Note: OpenCV rotation is counter-clockwise for positive angles
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=self.interpolation_method,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)  # Black background
        )

        return rotated

    def process_image(
            self,
            image_path: Path,
            output_path: Optional[Path] = None,
            original_image: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[float], Optional[np.ndarray]]:
        """
        Process image to normalize hand orientation.

        Returns:
            Tuple of (success, rotation_applied, rotated_image)
        """
        self.stats['processed'] += 1
        image_name = image_path.name if image_path else "live_frame"

        if self.debug_mode:
            print(f"\n{'=' * 60}")
            print(f"Processing: {image_name}")
            print(f"  → Mirrored mode: {self.is_mirrored}")

        # Read image
        if original_image is not None:
            image = original_image.copy()
        else:
            image = cv2.imread(str(image_path))

        if image is None:
            if self.debug_mode:
                print(f"❌ Could not read image")
            self.stats['errors'] += 1
            return False, None, None

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            self.stats['no_hands_found'] += 1
            if self.debug_mode:
                print(f"⚠️  No hand detected")

            # Save original if output path provided
            if output_path and original_image is None:
                cv2.imwrite(str(output_path), image)

            return False, None, image

        # Get hand orientation
        hand_landmarks = results.multi_hand_landmarks[0]
        current_angle = self.get_hand_orientation_angle(hand_landmarks, image.shape)

        if current_angle is None:
            self.stats['method_failures'] += 1
            if self.debug_mode:
                print(f"❌ Failed to calculate orientation")
            return False, None, image

        # Apply smoothing (for live processing)
        smoothed_angle = self.smooth_angle(current_angle)

        # Calculate rotation needed
        rotation_needed = self.calculate_rotation_needed(smoothed_angle)

        # Safety check
        if abs(rotation_needed) > self.max_rotation_angle:
            self.stats['excessive_rotation'] += 1
            if self.debug_mode:
                print(f"⚠️  Excessive rotation needed: {rotation_needed:.1f}° (max: {self.max_rotation_angle}°)")
            rotation_needed = np.clip(rotation_needed, -self.max_rotation_angle, self.max_rotation_angle)

        # Apply rotation
        rotated_image = self.rotate_image_around_center(image, rotation_needed)

        # Save result if output path provided
        if output_path:
            cv2.imwrite(str(output_path), rotated_image)

        # Update statistics
        self.stats['successful'] += 1
        self.stats['avg_rotation_applied'].append(abs(rotation_needed))
        self.stats['rotation_range']['min'] = min(self.stats['rotation_range']['min'], rotation_needed)
        self.stats['rotation_range']['max'] = max(self.stats['rotation_range']['max'], rotation_needed)

        if self.debug_mode:
            print(f"✅ SUCCESS:")
            print(f"  → Current angle: {current_angle:.1f}°")
            if self.smoothing_enabled:
                print(f"  → Smoothed angle: {smoothed_angle:.1f}°")
            print(f"  → Rotation applied: {rotation_needed:.1f}°")
            print(f"  → Method: {self.angle_calculation_method}")

        return True, rotation_needed, rotated_image

    def process_directory(
            self,
            input_dir: Path,
            output_dir: Optional[Path] = None,
            file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
    ):
        """
        Process all images in directory to normalize orientation.
        """
        input_dir = Path(input_dir)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in file_extensions
               and not any(suffix in f.stem for suffix in ['_normalized', '_rotated', '_oriented'])
        ]

        print(f"\n{'=' * 70}")
        print(f"HAND ORIENTATION NORMALIZER")
        print(f"{'=' * 70}")
        print(f"Images to process: {len(image_files)}")
        print(f"Target angle: {self.target_angle}° ({'up' if self.target_angle == -90 else 'custom'})")
        print(f"Calculation method: {self.angle_calculation_method}")
        print(f"Mirrored mode: {self.is_mirrored}")
        print(f"Smoothing: {'ON' if self.smoothing_enabled else 'OFF'}")
        print(f"Max rotation: {self.max_rotation_angle}°")
        print(f"{'=' * 70}\n")

        # Process each image
        for image_file in image_files:
            output_path = output_dir / image_file.name if output_dir else None
            self.process_image(image_file, output_path)

        # Print summary
        self.print_summary()

        # Save processing report
        if output_dir:
            self.save_processing_report(output_dir)

    def print_summary(self):
        """Print processing statistics."""
        avg_rotation = np.mean(self.stats['avg_rotation_applied']) if self.stats['avg_rotation_applied'] else 0

        print(f"\n{'=' * 70}")
        print("ORIENTATION NORMALIZATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total Processed:       {self.stats['processed']}")
        print(f"Successfully Rotated:  {self.stats['successful']}")
        print(f"No Hands Found:        {self.stats['no_hands_found']}")
        print(f"Method Failures:       {self.stats['method_failures']}")
        print(f"Excessive Rotations:   {self.stats['excessive_rotation']}")
        print(f"Errors:                {self.stats['errors']}")
        print(f"Success Rate:          {self.stats['successful'] / max(1, self.stats['processed']) * 100:.1f}%")
        print(f"\nRotation Statistics:")
        print(f"Average Rotation:      {avg_rotation:.1f}°")
        if self.stats['rotation_range']['min'] != float('inf'):
            print(
                f"Rotation Range:        {self.stats['rotation_range']['min']:.1f}° to {self.stats['rotation_range']['max']:.1f}°")
        print(f"{'=' * 70}\n")

    def save_processing_report(self, output_dir: Path):
        """Save detailed processing report."""
        avg_rotation = np.mean(self.stats['avg_rotation_applied']) if self.stats['avg_rotation_applied'] else 0

        report = {
            'processing_date': datetime.now().isoformat(),
            'configuration': {
                'target_angle': self.target_angle,
                'angle_calculation_method': self.angle_calculation_method,
                'is_mirrored': self.is_mirrored,
                'smoothing_enabled': self.smoothing_enabled,
                'max_rotation_angle': self.max_rotation_angle,
                'interpolation_method': str(self.interpolation_method)
            },
            'statistics': {
                'processed': self.stats['processed'],
                'successful': self.stats['successful'],
                'no_hands_found': self.stats['no_hands_found'],
                'method_failures': self.stats['method_failures'],
                'excessive_rotation': self.stats['excessive_rotation'],
                'errors': self.stats['errors'],
                'avg_rotation': avg_rotation,
                'rotation_range': self.stats['rotation_range'] if self.stats['rotation_range']['min'] != float(
                    'inf') else None
            },
            'description': 'Hand orientation normalization for consistent gesture positioning'
        }

        report_path = output_dir / 'orientation_normalization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Processing report saved: {report_path}")

    def reset_smoothing(self):
        """Reset smoothing history (useful for live processing when changing settings)."""
        self.angle_history.clear()

    def set_mirrored_mode(self, is_mirrored: bool):
        """
        Change mirrored mode dynamically.
        Useful for switching between camera and file processing.
        """
        self.is_mirrored = is_mirrored
        self.reset_smoothing()  # Reset smoothing when changing modes
        if self.debug_mode:
            print(f"🔄 Mirrored mode set to: {is_mirrored}")

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":

    print("🚀 Starting Hand Orientation Normalizer...")
    print("📋 This script normalizes hand orientation to consistent upright position")
    print()

    # Configuration
    INPUT_DIR = "accurate_binary_gestures"  # Input folder
    OUTPUT_DIR = "orientation_normalized"  # Output folder

    # Example 1: Standard upright normalization for file processing
    print("=" * 70)
    print("STANDARD FILE PROCESSING (NON-MIRRORED)")
    print("=" * 70)

    normalizer = HandOrientationNormalizer(
        target_angle=-90.0,  # Point up
        angle_calculation_method="wrist_to_middle",  # Most stable
        smoothing_enabled=False,  # Not needed for files
        max_rotation_angle=180.0,  # Allow full rotation
        interpolation_method=cv2.INTER_LINEAR,  # Good balance
        is_mirrored=False,  # Files are not mirrored
        debug_mode=True
    )

    normalizer.process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )

    print("\n✅ Orientation normalization complete!")
    print(f"💡 TIP: Check '{OUTPUT_DIR}' folder to see normalized hands")
    print("   All hands should now point upward consistently!")

    # Example 2: Test different methods (for comparison)
    print(f"\n{'=' * 70}")
    print("TESTING DIFFERENT CALCULATION METHODS")
    print(f"{'=' * 70}")

    methods = ["wrist_to_middle", "wrist_to_fingertips", "palm_orientation"]

    for method in methods:
        print(f"\nTesting method: {method}")
        test_normalizer = HandOrientationNormalizer(
            target_angle=-90.0,
            angle_calculation_method=method,
            smoothing_enabled=False,
            is_mirrored=False,
            debug_mode=False
        )

        test_output = Path(OUTPUT_DIR) / f"method_{method}"
        test_normalizer.process_directory(
            input_dir=INPUT_DIR,
            output_dir=test_output
        )

        print(f"\nMethod '{method}' results:")
        test_normalizer.print_summary()
