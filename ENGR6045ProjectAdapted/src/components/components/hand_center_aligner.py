import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from datetime import datetime
import json


class HandCenterAligner:
    """
    Aligns hand to the center of the image based on center of mass.
    Ensures consistent positioning for gesture classification.
    """

    def __init__(
            self,
            center_method: str = "center_of_mass",  # "center_of_mass", "bounding_box_center", "both"
            preserve_aspect_ratio: bool = True,
            background_color: Tuple[int, int, int] = (0, 0, 0),  # Black background
            interpolation_method: int = cv2.INTER_LINEAR,
            debug_mode: bool = True
    ):
        """
        Initialize center aligner.

        Args:
            center_method: Method for finding hand center
                - "center_of_mass": Use center of mass of white pixels (best for binary masks)
                - "bounding_box_center": Use center of bounding box (good for regular images)
                - "both": Average of both methods
            preserve_aspect_ratio: Maintain original aspect ratio when centering
            background_color: Background color for empty areas
            interpolation_method: CV2 interpolation method for translation
            debug_mode: Print detailed processing info
        """
        self.center_method = center_method
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.background_color = background_color
        self.interpolation_method = interpolation_method
        self.debug_mode = debug_mode

        # Statistics
        self.stats = {
            'processed': 0,
            'successful': 0,
            'no_hand_pixels': 0,
            'avg_translation_x': [],
            'avg_translation_y': [],
            'avg_translation_magnitude': [],
            'translation_range': {
                'x': {'min': float('inf'), 'max': float('-inf')},
                'y': {'min': float('inf'), 'max': float('-inf')}
            },
            'errors': 0
        }

    def calculate_center_of_mass(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Calculate center of mass of hand pixels.
        Works best with binary images (white hand on black background).
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate moments
        moments = cv2.moments(gray)

        # Check if any pixels found
        if moments['m00'] == 0:
            if self.debug_mode:
                print(f"  ⚠️ No hand pixels found for center of mass")
            return None

        # Calculate center of mass
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        if self.debug_mode:
            print(f"  → Center of mass: ({cx}, {cy})")

        return (cx, cy)

    def calculate_bounding_box_center(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Calculate center of bounding box around hand pixels.
        Good for both binary and regular images.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Threshold to find hand pixels
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            if self.debug_mode:
                print(f"  ⚠️ No contours found for bounding box")
            return None

        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate center
        cx = x + w // 2
        cy = y + h // 2

        if self.debug_mode:
            print(f"  → Bounding box center: ({cx}, {cy})")

        return (cx, cy)

    def get_hand_center(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Get hand center using specified method.
        """
        if self.center_method == "center_of_mass":
            return self.calculate_center_of_mass(image)

        elif self.center_method == "bounding_box_center":
            return self.calculate_bounding_box_center(image)

        elif self.center_method == "both":
            # Average both methods
            com = self.calculate_center_of_mass(image)
            bbox_center = self.calculate_bounding_box_center(image)

            if com is None and bbox_center is None:
                return None
            elif com is None:
                return bbox_center
            elif bbox_center is None:
                return com
            else:
                # Average the two
                avg_cx = (com[0] + bbox_center[0]) // 2
                avg_cy = (com[1] + bbox_center[1]) // 2
                if self.debug_mode:
                    print(f"  → Averaged center: ({avg_cx}, {avg_cy})")
                return (avg_cx, avg_cy)

        else:
            # Default to center of mass
            return self.calculate_center_of_mass(image)

    def calculate_translation(
            self,
            hand_center: Tuple[int, int],
            image_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Calculate translation needed to center the hand.
        """
        h, w = image_shape[:2]
        image_center = (w // 2, h // 2)

        # Calculate translation
        tx = image_center[0] - hand_center[0]
        ty = image_center[1] - hand_center[1]

        return (tx, ty)

    def apply_translation(
            self,
            image: np.ndarray,
            translation: Tuple[int, int]
    ) -> np.ndarray:
        """
        Apply translation to center the hand.
        """
        tx, ty = translation
        h, w = image.shape[:2]

        # Create translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        # Apply translation
        centered = cv2.warpAffine(
            image,
            translation_matrix,
            (w, h),
            flags=self.interpolation_method,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.background_color
        )

        return centered

    def process_image(
            self,
            image_path: Optional[Path] = None,
            output_path: Optional[Path] = None,
            original_image: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[Tuple[int, int]], Optional[np.ndarray]]:
        """
        Process image to center the hand.

        Returns:
            Tuple of (success, translation_applied, centered_image)
        """
        self.stats['processed'] += 1
        image_name = image_path.name if image_path else "live_frame"

        if self.debug_mode:
            print(f"\n{'=' * 60}")
            print(f"Processing: {image_name}")

        # Read image
        if original_image is not None:
            image = original_image.copy()
        else:
            if image_path is None:
                if self.debug_mode:
                    print(f"❌ No image provided")
                self.stats['errors'] += 1
                return False, None, None
            image = cv2.imread(str(image_path))

        if image is None:
            if self.debug_mode:
                print(f"❌ Could not read image")
            self.stats['errors'] += 1
            return False, None, None

        # Get hand center
        hand_center = self.get_hand_center(image)

        if hand_center is None:
            self.stats['no_hand_pixels'] += 1
            if self.debug_mode:
                print(f"⚠️ Could not find hand center")

            # Save original if output path provided
            if output_path:
                cv2.imwrite(str(output_path), image)

            return False, None, image

        # Calculate translation needed
        translation = self.calculate_translation(hand_center, image.shape)
        tx, ty = translation

        # Apply translation
        centered_image = self.apply_translation(image, translation)

        # Save result if output path provided
        if output_path:
            cv2.imwrite(str(output_path), centered_image)

        # Update statistics
        self.stats['successful'] += 1
        self.stats['avg_translation_x'].append(tx)
        self.stats['avg_translation_y'].append(ty)

        # Calculate translation magnitude (distance)
        magnitude = np.sqrt(tx ** 2 + ty ** 2)
        self.stats['avg_translation_magnitude'].append(magnitude)

        # Update ranges
        self.stats['translation_range']['x']['min'] = min(
            self.stats['translation_range']['x']['min'], tx
        )
        self.stats['translation_range']['x']['max'] = max(
            self.stats['translation_range']['x']['max'], tx
        )
        self.stats['translation_range']['y']['min'] = min(
            self.stats['translation_range']['y']['min'], ty
        )
        self.stats['translation_range']['y']['max'] = max(
            self.stats['translation_range']['y']['max'], ty
        )

        if self.debug_mode:
            print(f"✅ SUCCESS:")
            print(f"  → Hand center: {hand_center}")
            print(f"  → Image center: ({image.shape[1] // 2}, {image.shape[0] // 2})")
            print(f"  → Translation: ({tx:+d}, {ty:+d}) pixels")
            print(f"  → Distance: {magnitude:.1f} pixels")
            print(f"  → Method: {self.center_method}")

        return True, translation, centered_image

    def process_directory(
            self,
            input_dir: Path,
            output_dir: Optional[Path] = None,
            file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
    ):
        """
        Process all images in directory to center hands.
        """
        input_dir = Path(input_dir)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in file_extensions
               and not any(suffix in f.stem for suffix in ['_centered', '_aligned'])
        ]

        print(f"\n{'=' * 70}")
        print(f"HAND CENTER ALIGNER")
        print(f"{'=' * 70}")
        print(f"Images to process: {len(image_files)}")
        print(f"Center method: {self.center_method}")
        print(f"Background color: {self.background_color}")
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
        avg_tx = np.mean(self.stats['avg_translation_x']) if self.stats['avg_translation_x'] else 0
        avg_ty = np.mean(self.stats['avg_translation_y']) if self.stats['avg_translation_y'] else 0
        avg_mag = np.mean(self.stats['avg_translation_magnitude']) if self.stats['avg_translation_magnitude'] else 0

        print(f"\n{'=' * 70}")
        print("CENTER ALIGNMENT SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total Processed:       {self.stats['processed']}")
        print(f"Successfully Centered: {self.stats['successful']}")
        print(f"No Hand Pixels:        {self.stats['no_hand_pixels']}")
        print(f"Errors:                {self.stats['errors']}")
        print(f"Success Rate:          {self.stats['successful'] / max(1, self.stats['processed']) * 100:.1f}%")
        print(f"\nTranslation Statistics:")
        print(f"Average X Translation: {avg_tx:+.1f} pixels")
        print(f"Average Y Translation: {avg_ty:+.1f} pixels")
        print(f"Average Distance:      {avg_mag:.1f} pixels")

        if self.stats['translation_range']['x']['min'] != float('inf'):
            print(f"\nTranslation Ranges:")
            print(
                f"X: {self.stats['translation_range']['x']['min']:+d} to {self.stats['translation_range']['x']['max']:+d} pixels")
            print(
                f"Y: {self.stats['translation_range']['y']['min']:+d} to {self.stats['translation_range']['y']['max']:+d} pixels")

        print(f"{'=' * 70}\n")

    def save_processing_report(self, output_dir: Path):
        """Save detailed processing report."""
        avg_tx = np.mean(self.stats['avg_translation_x']) if self.stats['avg_translation_x'] else 0
        avg_ty = np.mean(self.stats['avg_translation_y']) if self.stats['avg_translation_y'] else 0
        avg_mag = np.mean(self.stats['avg_translation_magnitude']) if self.stats['avg_translation_magnitude'] else 0

        report = {
            'processing_date': datetime.now().isoformat(),
            'configuration': {
                'center_method': self.center_method,
                'preserve_aspect_ratio': self.preserve_aspect_ratio,
                'background_color': self.background_color,
                'interpolation_method': str(self.interpolation_method)
            },
            'statistics': {
                'processed': self.stats['processed'],
                'successful': self.stats['successful'],
                'no_hand_pixels': self.stats['no_hand_pixels'],
                'errors': self.stats['errors'],
                'avg_translation_x': avg_tx,
                'avg_translation_y': avg_ty,
                'avg_translation_magnitude': avg_mag,
                'translation_range': self.stats['translation_range'] if self.stats['translation_range']['x'][
                                                                            'min'] != float('inf') else None
            },
            'description': 'Hand center alignment for consistent positioning'
        }

        report_path = output_dir / 'center_alignment_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Processing report saved: {report_path}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":

    print("🚀 Starting Hand Center Aligner...")
    print("📋 This script centers hands in images for consistent positioning")
    print()

    # Configuration
    INPUT_DIR = "orientation_normalized"  # Input folder (after orientation)
    OUTPUT_DIR = "centered_hands"  # Output folder

    # Example 1: Center of mass method (RECOMMENDED for binary images)
    print("=" * 70)
    print("CENTER OF MASS ALIGNMENT (Best for binary masks)")
    print("=" * 70)

    aligner = HandCenterAligner(
        center_method="center_of_mass",  # Best for binary masks
        preserve_aspect_ratio=True,
        background_color=(0, 0, 0),  # Black background
        interpolation_method=cv2.INTER_LINEAR,
        debug_mode=True
    )

    aligner.process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )

    print("\n✅ Center alignment complete!")
    print(f"💡 TIP: Check '{OUTPUT_DIR}' folder to see centered hands")
    print("   All hands should now be in the exact center!")

    # Example 2: Test different methods
    print(f"\n{'=' * 70}")
    print("TESTING DIFFERENT CENTER METHODS")
    print(f"{'=' * 70}")

    methods = ["center_of_mass", "bounding_box_center", "both"]

    for method in methods:
        print(f"\nTesting method: {method}")
        test_aligner = HandCenterAligner(
            center_method=method,
            debug_mode=False
        )

        test_output = Path(OUTPUT_DIR) / f"method_{method}"
        test_aligner.process_directory(
            input_dir=INPUT_DIR,
            output_dir=test_output
        )

        print(f"\nMethod '{method}' results:")
        test_aligner.print_summary()