import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from datetime import datetime
import json
from scipy.spatial.distance import cdist
from scipy import ndimage
from skimage import segmentation, measure


class HandBinaryMaskProcessor:
    """
    Creates accurate binary masks by tracing actual hand contours,
    preserving finger separation and gesture details.
    """

    def __init__(
            self,
            debug_mode: bool = True,
            min_detection_confidence: float = 0.5,
            contour_method: str = "watershed",  # "watershed", "grabcut", or "edge_based"
            edge_sensitivity: float = 0.3,  # For edge detection (0.1-0.9)
            morphology_size: int = 3,  # Size of morphological operations
            min_contour_area: int = 1000  # Minimum contour area to consider
    ):
        """
        Initialize the accurate hand mask processor.

        Args:
            debug_mode: If True, print detailed processing info
            min_detection_confidence: MediaPipe detection confidence threshold
            contour_method: Method for accurate contour detection
            edge_sensitivity: Sensitivity for edge detection (lower = more sensitive)
            morphology_size: Size of morphological kernels
            min_contour_area: Minimum area for valid hand contours
        """
        self.debug_mode = debug_mode
        self.contour_method = contour_method
        self.edge_sensitivity = edge_sensitivity
        self.morphology_size = morphology_size
        self.min_contour_area = min_contour_area

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=10,
            min_detection_confidence=min_detection_confidence
        )

        # Hand landmark groups for better processing
        self.finger_landmarks = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        self.palm_landmarks = [0, 1, 2, 5, 9, 13, 17]

        # Statistics
        self.stats = {
            'processed': 0,
            'successful': 0,
            'no_hands_found': 0,
            'multiple_hands_found': 0,
            'contour_failures': 0,
            'method_fallbacks': 0,
            'errors': 0
        }

    def create_landmark_guidance_mask(
            self,
            hand_landmarks,
            image_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Create a guidance mask from MediaPipe landmarks without filling gaps.
        """
        h, w, _ = image_shape
        guidance_mask = np.zeros((h, w), dtype=np.uint8)

        # Extract landmark coordinates
        points = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        points = np.array(points, dtype=np.int32)

        # Draw palm area (more conservative)
        palm_points = points[self.palm_landmarks]
        palm_hull = cv2.convexHull(palm_points)
        cv2.fillConvexPoly(guidance_mask, palm_hull, 255)

        # Draw finger paths (lines, not filled areas)
        for finger_name, finger_indices in self.finger_landmarks.items():
            finger_points = points[finger_indices]

            # Draw finger as connected lines with thickness
            for i in range(len(finger_points) - 1):
                pt1 = tuple(finger_points[i])
                pt2 = tuple(finger_points[i + 1])
                cv2.line(guidance_mask, pt1, pt2, 255, thickness=8)

            # Add circles at finger joints
            for point in finger_points:
                cv2.circle(guidance_mask, tuple(point), 6, 255, -1)

        return guidance_mask

    def watershed_segmentation(
            self,
            image: np.ndarray,
            guidance_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Use watershed segmentation for accurate hand boundary detection.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Calculate gradient magnitude
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
            gradient = np.uint8(gradient / gradient.max() * 255)

            # Create markers for watershed
            markers = np.zeros_like(gray, dtype=np.int32)

            # Sure background (dilated guidance mask inverted)
            sure_bg = cv2.dilate(guidance_mask,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)),
                                 iterations=1)
            markers[sure_bg == 0] = 1

            # Sure foreground (eroded guidance mask)
            sure_fg = cv2.erode(guidance_mask,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                iterations=1)
            markers[sure_fg == 255] = 2

            # Apply watershed
            watershed_result = cv2.watershed(image, markers)

            # Extract hand region
            hand_mask = np.zeros_like(gray)
            hand_mask[watershed_result == 2] = 255

            return hand_mask

        except Exception as e:
            if self.debug_mode:
                print(f"  ❌ Watershed failed: {e}")
            return None

    def grabcut_segmentation(
            self,
            image: np.ndarray,
            guidance_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Use GrabCut algorithm for accurate foreground extraction.
        """
        try:
            # Create GrabCut mask
            gc_mask = np.full(image.shape[:2], cv2.GC_BGD, dtype=np.uint8)

            # Set probable foreground from guidance
            gc_mask[guidance_mask == 255] = cv2.GC_FGD

            # Set probable background (area around hand)
            dilated_guidance = cv2.dilate(guidance_mask,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)),
                                          iterations=1)
            uncertain_area = cv2.subtract(dilated_guidance, guidance_mask)
            gc_mask[uncertain_area == 255] = cv2.GC_PR_BGD

            # Initialize background and foreground models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Apply GrabCut
            cv2.grabCut(image, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)

            # Extract foreground
            hand_mask = np.zeros_like(guidance_mask)
            hand_mask[(gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)] = 255

            return hand_mask

        except Exception as e:
            if self.debug_mode:
                print(f"  ❌ GrabCut failed: {e}")
            return None

    def edge_based_segmentation(
            self,
            image: np.ndarray,
            guidance_mask: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Use edge detection and contour finding for hand boundary.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)

            # Adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Edge detection
            edges = cv2.Canny(filtered,
                              int(255 * self.edge_sensitivity),
                              int(255 * self.edge_sensitivity * 2))

            # Combine with guidance mask
            combined = cv2.bitwise_or(edges, guidance_mask)

            # Find contours
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Filter contours by area and proximity to guidance mask
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_contour_area:
                    # Check if contour overlaps with guidance mask
                    temp_mask = np.zeros_like(gray)
                    cv2.fillPoly(temp_mask, [contour], 255)
                    overlap = cv2.bitwise_and(temp_mask, guidance_mask)
                    if np.count_nonzero(overlap) > 100:  # Sufficient overlap
                        valid_contours.append(contour)

            if not valid_contours:
                return None

            # Create mask from valid contours
            hand_mask = np.zeros_like(gray)
            cv2.fillPoly(hand_mask, valid_contours, 255)

            return hand_mask

        except Exception as e:
            if self.debug_mode:
                print(f"  ❌ Edge-based segmentation failed: {e}")
            return None

    def refine_hand_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine the hand mask with morphological operations.
        """
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (self.morphology_size, self.morphology_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

        # Fill small holes
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (self.morphology_size * 2, self.morphology_size * 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)

        # Smooth edges slightly
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask

    def create_accurate_hand_mask(
            self,
            image: np.ndarray,
            hand_landmarks,
            image_shape: Tuple[int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Create an accurate hand mask using the specified method.
        """
        # Create guidance mask from landmarks
        guidance_mask = self.create_landmark_guidance_mask(hand_landmarks, image_shape)

        if self.debug_mode:
            print(f"  → Using {self.contour_method} method")

        # Try primary method
        if self.contour_method == "watershed":
            hand_mask = self.watershed_segmentation(image, guidance_mask)
        elif self.contour_method == "grabcut":
            hand_mask = self.grabcut_segmentation(image, guidance_mask)
        elif self.contour_method == "edge_based":
            hand_mask = self.edge_based_segmentation(image, guidance_mask)
        else:
            hand_mask = None

        # Fallback methods if primary fails
        if hand_mask is None:
            self.stats['method_fallbacks'] += 1
            if self.debug_mode:
                print(f"  ⚠️ Primary method failed, trying fallbacks...")

            # Try other methods in order
            fallback_methods = ["edge_based", "watershed", "grabcut"]
            if self.contour_method in fallback_methods:
                fallback_methods.remove(self.contour_method)

            for method in fallback_methods:
                if method == "watershed":
                    hand_mask = self.watershed_segmentation(image, guidance_mask)
                elif method == "grabcut":
                    hand_mask = self.grabcut_segmentation(image, guidance_mask)
                elif method == "edge_based":
                    hand_mask = self.edge_based_segmentation(image, guidance_mask)

                if hand_mask is not None:
                    if self.debug_mode:
                        print(f"  ✅ Fallback method '{method}' succeeded")
                    break

        # Final fallback: use guidance mask
        if hand_mask is None:
            if self.debug_mode:
                print(f"  ⚠️ All methods failed, using guidance mask")
            hand_mask = guidance_mask
            self.stats['contour_failures'] += 1

        # Refine the mask
        if hand_mask is not None:
            hand_mask = self.refine_hand_mask(hand_mask)

        return hand_mask

    def combine_multiple_hand_masks(
            self,
            image: np.ndarray,
            results,
            image_shape: Tuple[int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Create combined accurate mask for all detected hands.
        """
        if not results.multi_hand_landmarks:
            return None

        num_hands = len(results.multi_hand_landmarks)

        if num_hands > 1:
            self.stats['multiple_hands_found'] += 1
            if self.debug_mode:
                print(f"  → Processing {num_hands} hands separately")

        # Create mask for first hand
        combined_mask = self.create_accurate_hand_mask(
            image,
            results.multi_hand_landmarks[0],
            image_shape
        )

        if combined_mask is None:
            return None

        # Add masks for additional hands
        for i in range(1, num_hands):
            hand_mask = self.create_accurate_hand_mask(
                image,
                results.multi_hand_landmarks[i],
                image_shape
            )
            if hand_mask is not None:
                combined_mask = cv2.bitwise_or(combined_mask, hand_mask)

        return combined_mask

    def apply_binary_mask(
            self,
            image: np.ndarray,
            mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply binary mask to create white hand on black background.
        """
        binary_image = np.zeros_like(image)
        binary_image[mask == 255] = [255, 255, 255]
        return binary_image

    def process_image(
            self,
            image_path: Path,
            output_path: Optional[Path] = None,
            save_debug_images: bool = False
    ) -> bool:
        """
        Process a single image to create accurate binary hand mask.
        """
        self.stats['processed'] += 1
        image_name = image_path.name

        if self.debug_mode:
            print(f"\n{'=' * 60}")
            print(f"Processing: {image_name}")

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            if self.debug_mode:
                print(f"❌ ERROR: Could not read image: {image_name}")
            self.stats['errors'] += 1
            return False

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = self.hands.process(image_rgb)

        # Create accurate combined mask for all hands
        mask = self.combine_multiple_hand_masks(image, results, image.shape)

        if mask is None:
            self.stats['no_hands_found'] += 1
            if self.debug_mode:
                print(f"⚠️  NO HAND DETECTED: {image_name}")
            else:
                image_path.unlink()
                if self.debug_mode:
                    print(f"🗑️  DELETED (no hand): {image_name}")
            return False

        # Create binary image
        binary_image = self.apply_binary_mask(image, mask)

        # Save processed image
        save_path = output_path if output_path else image_path
        cv2.imwrite(str(save_path), binary_image)

        # Save debug images if requested
        if save_debug_images and output_path:
            # Save mask
            mask_path = save_path.parent / f"{save_path.stem}_mask{save_path.suffix}"
            cv2.imwrite(str(mask_path), mask)

            # Save guidance mask for comparison
            guidance_mask = self.create_landmark_guidance_mask(
                results.multi_hand_landmarks[0], image.shape)
            guidance_path = save_path.parent / f"{save_path.stem}_guidance{save_path.suffix}"
            cv2.imwrite(str(guidance_path), guidance_mask)

        self.stats['successful'] += 1

        if self.debug_mode:
            hand_pixels = np.count_nonzero(mask)
            total_pixels = mask.size
            print(f"✅ SUCCESS: Accurate hand mask created")
            print(f"  → Hand pixels: {hand_pixels:,}")
            print(f"  → Coverage: {hand_pixels / total_pixels * 100:.1f}%")

        return True

    def process_directory(
            self,
            input_dir: Path,
            output_dir: Optional[Path] = None,
            file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
            save_debug_images: bool = False
    ):
        """
        Process all images in a directory with accurate hand tracing.
        """
        input_dir = Path(input_dir)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in file_extensions
               and not any(suffix in f.stem for suffix in ['_mask', '_guidance', '_binary'])
        ]

        print(f"\n{'=' * 70}")
        print(f"ACCURATE HAND CONTOUR PROCESSOR")
        print(f"{'=' * 70}")
        print(f"Purpose: Create accurate binary masks preserving finger separation")
        print(f"Images to process: {len(image_files)}")
        print(f"Method: {self.contour_method}")
        print(f"Edge Sensitivity: {self.edge_sensitivity}")
        print(f"Debug Mode: {'ON' if self.debug_mode else 'OFF'}")
        print(f"Save Debug Images: {'Yes' if save_debug_images else 'No'}")
        print(f"{'=' * 70}\n")

        # Process each image
        for image_file in image_files:
            output_path = output_dir / image_file.name if output_dir else None
            self.process_image(image_file, output_path, save_debug_images)

        # Print summary
        self.print_summary()

        # Save processing report
        if output_dir:
            self.save_processing_report(output_dir)

    def print_summary(self):
        """Print processing statistics."""
        print(f"\n{'=' * 70}")
        print("PROCESSING SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total Processed:        {self.stats['processed']}")
        print(f"Successfully Converted: {self.stats['successful']}")
        print(f"No Hands Found:         {self.stats['no_hands_found']}")
        print(f"Multiple Hands Found:   {self.stats['multiple_hands_found']}")
        print(f"Method Fallbacks:       {self.stats['method_fallbacks']}")
        print(f"Contour Failures:       {self.stats['contour_failures']}")
        print(f"Errors:                 {self.stats['errors']}")
        print(f"Success Rate:           {self.stats['successful'] / max(1, self.stats['processed']) * 100:.1f}%")
        print(f"{'=' * 70}\n")

    def save_processing_report(self, output_dir: Path):
        """Save detailed processing report."""
        report = {
            'processing_date': datetime.now().isoformat(),
            'configuration': {
                'debug_mode': self.debug_mode,
                'contour_method': self.contour_method,
                'edge_sensitivity': self.edge_sensitivity,
                'morphology_size': self.morphology_size,
                'min_contour_area': self.min_contour_area
            },
            'statistics': self.stats,
            'description': 'Accurate hand contour tracing: preserves finger separation'
        }

        report_path = output_dir / 'accurate_mask_processing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Processing report saved: {report_path}")

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "processed_gesture_images"  # Your collected test data
    OUTPUT_DIR = "accurate_binary_gestures"  # Output folder

    print("🚀 Starting Accurate Hand Mask Processor...")
    print("📋 This script will:")
    print("   • Detect hands using MediaPipe landmarks")
    print("   • Trace actual hand contours (preserving finger gaps)")
    print("   • Convert to binary: hand=WHITE, background=BLACK")
    print("   • Maintain gesture details (fist vs open hand)")
    print()

    # Initialize processor
    processor = HandBinaryMaskProcessor(
        debug_mode=True,
        contour_method="edge_based",  # Try: "watershed", "grabcut", "edge_based"
        edge_sensitivity=0.3,  # Lower = more sensitive edge detection
        morphology_size=3,  # Size of smoothing operations
        min_contour_area=1000  # Minimum area for valid contours
    )

    # Process all images
    processor.process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        save_debug_images=True  # Save intermediate images for inspection
    )

    print("\n✅ Processing complete!")
    print(f"💡 TIP: Check '{OUTPUT_DIR}' folder for results")
    print("   Look for '_mask' and '_guidance' files to see the difference")
    print("   Compare fist vs open hand - fingers should be separated!")
