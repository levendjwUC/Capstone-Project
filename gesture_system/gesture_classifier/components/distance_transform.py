import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from datetime import datetime
import json


class HandDistanceTransform:
    """
    Applies distance transform to hand images.
    Shows distance from each hand pixel to nearest background pixel.
    Useful for understanding hand structure and finger width.
    """

    def __init__(
            self,
            distance_type: int = cv2.DIST_L2,  # Distance calculation type
            mask_size: int = 5,  # Mask size for distance calculation (3 or 5)
            visualization_mode: str = "normalized",  # "normalized", "colormap", "binary_threshold", "skeleton_weighted"
            colormap: int = cv2.COLORMAP_JET,  # Colormap for visualization
            normalize_range: Tuple[int, int] = (0, 255),  # Range for normalization
            apply_smoothing: bool = True,  # Smooth the distance map
            smoothing_kernel: int = 5,  # Gaussian blur kernel size
            binary_threshold: Optional[int] = None,  # Threshold for binary visualization
            enhance_ridges: bool = False,  # Enhance ridge lines (medial axis)
            ridge_threshold: float = 0.5,  # Threshold for ridge detection (0-1)
            background_color: Tuple[int, int, int] = (0, 0, 0),  # Background color
            debug_mode: bool = True
    ):
        """
        Initialize distance transform processor.

        Args:
            distance_type: Type of distance metric
                - cv2.DIST_L1: Manhattan distance
                - cv2.DIST_L2: Euclidean distance (default, most accurate)
                - cv2.DIST_C: Chessboard distance
            mask_size: Size of distance transform mask (3 or 5)
            visualization_mode: How to visualize the distance transform
                - "normalized": Grayscale normalized to 0-255
                - "colormap": Apply colormap (heat map style)
                - "binary_threshold": Threshold distance map to binary
                - "skeleton_weighted": Combine with skeleton
            colormap: OpenCV colormap for visualization
            normalize_range: Output range for normalization
            apply_smoothing: Smooth the distance map
            smoothing_kernel: Size of Gaussian blur kernel
            binary_threshold: Distance threshold for binary visualization
            enhance_ridges: Highlight the medial axis (skeleton)
            ridge_threshold: Threshold for ridge detection
            background_color: Background color
            debug_mode: Print detailed processing info
        """
        self.distance_type = distance_type
        self.mask_size = mask_size if mask_size in [3, 5] else 5
        self.visualization_mode = visualization_mode
        self.colormap = colormap
        self.normalize_range = normalize_range
        self.apply_smoothing = apply_smoothing
        self.smoothing_kernel = smoothing_kernel if smoothing_kernel % 2 == 1 else smoothing_kernel + 1
        self.binary_threshold = binary_threshold
        self.enhance_ridges = enhance_ridges
        self.ridge_threshold = ridge_threshold
        self.background_color = background_color
        self.debug_mode = debug_mode

        # Statistics
        self.stats = {
            'processed': 0,
            'successful': 0,
            'no_hand_pixels': 0,
            'avg_max_distance': [],
            'avg_mean_distance': [],
            'distance_std_dev': [],
            'errors': 0
        }

    def preprocess_for_distance_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for distance transform.
        Convert to binary format required by distance transform.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Threshold to binary
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        return binary

    def calculate_distance_transform(self, binary: np.ndarray) -> np.ndarray:
        """
        Calculate distance transform.
        Returns distance from each foreground pixel to nearest background pixel.
        """
        # Apply distance transform
        if self.mask_size == 3:
            dist_transform = cv2.distanceTransform(binary, self.distance_type, 3)
        else:  # mask_size == 5
            dist_transform = cv2.distanceTransform(binary, self.distance_type, 5)

        if self.debug_mode:
            max_dist = np.max(dist_transform)
            mean_dist = np.mean(dist_transform[dist_transform > 0])
            std_dist = np.std(dist_transform[dist_transform > 0])
            print(f"  → Distance transform calculated")
            print(f"  → Max distance: {max_dist:.2f} pixels")
            print(f"  → Mean distance: {mean_dist:.2f} pixels")
            print(f"  → Std deviation: {std_dist:.2f} pixels")

        return dist_transform

    def smooth_distance_map(self, dist_transform: np.ndarray) -> np.ndarray:
        """
        Smooth the distance map to reduce noise.
        """
        if not self.apply_smoothing:
            return dist_transform

        # Apply Gaussian blur
        smoothed = cv2.GaussianBlur(dist_transform, (self.smoothing_kernel, self.smoothing_kernel), 0)

        if self.debug_mode:
            print(f"  → Applied smoothing (kernel: {self.smoothing_kernel})")

        return smoothed

    def detect_ridges(self, dist_transform: np.ndarray) -> np.ndarray:
        """
        Detect ridge lines in distance transform (medial axis).
        These represent the skeleton or center line of the hand.
        """
        # Calculate gradients
        grad_x = cv2.Sobel(dist_transform, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(dist_transform, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize gradient magnitude
        if np.max(grad_magnitude) > 0:
            grad_magnitude_norm = grad_magnitude / np.max(grad_magnitude)
        else:
            grad_magnitude_norm = grad_magnitude

        # Ridges are where gradient is low (local maxima in distance transform)
        # Invert so ridges are bright
        ridge_map = 1.0 - grad_magnitude_norm

        # Threshold to get ridge lines
        ridge_threshold_value = self.ridge_threshold
        ridges = (ridge_map > ridge_threshold_value).astype(np.uint8) * 255

        # Only keep ridges that are on the hand
        hand_mask = (dist_transform > 0).astype(np.uint8)
        ridges = cv2.bitwise_and(ridges, ridges, mask=hand_mask)

        if self.debug_mode:
            ridge_pixels = np.count_nonzero(ridges)
            print(f"  → Ridge pixels detected: {ridge_pixels:,}")

        return ridges

    def visualize_normalized(self, dist_transform: np.ndarray) -> np.ndarray:
        """
        Visualize distance transform as normalized grayscale.
        """
        # Normalize to specified range
        if np.max(dist_transform) > 0:
            normalized = cv2.normalize(
                dist_transform,
                None,
                self.normalize_range[0],
                self.normalize_range[1],
                cv2.NORM_MINMAX
            )
        else:
            normalized = np.zeros_like(dist_transform)

        # Convert to uint8
        normalized = normalized.astype(np.uint8)

        if self.debug_mode:
            print(f"  → Normalized to range {self.normalize_range}")

        return normalized

    def visualize_colormap(self, dist_transform: np.ndarray) -> np.ndarray:
        """
        Visualize distance transform with colormap (heat map style).
        """
        # Normalize to 0-255
        if np.max(dist_transform) > 0:
            normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        else:
            normalized = np.zeros_like(dist_transform)

        normalized = normalized.astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(normalized, self.colormap)

        # Set background to black
        background_mask = (dist_transform == 0)
        colored[background_mask] = self.background_color

        if self.debug_mode:
            print(f"  → Applied colormap (type: {self.colormap})")

        return colored

    def visualize_binary_threshold(self, dist_transform: np.ndarray) -> np.ndarray:
        """
        Visualize distance transform as binary (threshold at specific distance).
        Shows regions that are at least X pixels from the edge.
        """
        if self.binary_threshold is None:
            # Auto threshold at half of max distance
            threshold_value = np.max(dist_transform) / 2
        else:
            threshold_value = self.binary_threshold

        # Threshold
        _, binary = cv2.threshold(dist_transform, threshold_value, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)

        if self.debug_mode:
            print(f"  → Binary threshold at {threshold_value:.2f} pixels")
            thresholded_pixels = np.count_nonzero(binary)
            print(f"  → Pixels above threshold: {thresholded_pixels:,}")

        return binary

    def visualize_skeleton_weighted(
            self,
            dist_transform: np.ndarray,
            ridges: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Visualize distance transform weighted by skeleton.
        Shows distance map with skeleton overlay.
        """
        # Normalize distance transform
        normalized = self.visualize_normalized(dist_transform)

        # Convert to color for overlay
        colored = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

        # Overlay ridges if available
        if ridges is not None and self.enhance_ridges:
            # Make ridges stand out
            colored[ridges > 0] = [255, 255, 255]  # White ridges

        if self.debug_mode:
            print(f"  → Created skeleton-weighted visualization")

        return colored

    def create_distance_visualization(
            self,
            dist_transform: np.ndarray,
            original_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Create final visualization based on selected mode.
        """
        # Detect ridges if needed
        ridges = None
        if self.enhance_ridges:
            ridges = self.detect_ridges(dist_transform)

        # Create visualization based on mode
        if self.visualization_mode == "normalized":
            result = self.visualize_normalized(dist_transform)
            # Convert to color if original was color
            if len(original_shape) == 3:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elif self.visualization_mode == "colormap":
            result = self.visualize_colormap(dist_transform)

        elif self.visualization_mode == "binary_threshold":
            result = self.visualize_binary_threshold(dist_transform)
            # Convert to color if original was color
            if len(original_shape) == 3:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elif self.visualization_mode == "skeleton_weighted":
            result = self.visualize_skeleton_weighted(dist_transform, ridges)

        else:
            # Default to normalized
            result = self.visualize_normalized(dist_transform)
            if len(original_shape) == 3:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # Overlay ridges if requested (for non-skeleton_weighted modes)
        if self.enhance_ridges and ridges is not None and self.visualization_mode != "skeleton_weighted":
            if len(result.shape) == 2:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            result[ridges > 0] = [255, 255, 255]  # White ridge lines

        return result

    def process_image(
            self,
            image_path: Optional[Path] = None,
            output_path: Optional[Path] = None,
            original_image: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[np.ndarray], Optional[Dict]]:
        """
        Process image to apply distance transform.

        Returns:
            Tuple of (success, transformed_image, distance_stats)
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

        # Preprocess
        binary = self.preprocess_for_distance_transform(image)

        # Check if any hand pixels
        if np.count_nonzero(binary) == 0:
            self.stats['no_hand_pixels'] += 1
            if self.debug_mode:
                print(f"⚠️ No hand pixels found")
            return False, image, None

        # Calculate distance transform
        dist_transform = self.calculate_distance_transform(binary)

        # Smooth if requested
        if self.apply_smoothing:
            dist_transform = self.smooth_distance_map(dist_transform)

        # Create visualization
        result = self.create_distance_visualization(dist_transform, image.shape)

        # Save result if output path provided
        if output_path:
            cv2.imwrite(str(output_path), result)

        # Calculate statistics
        max_distance = np.max(dist_transform)
        mean_distance = np.mean(dist_transform[dist_transform > 0]) if np.any(dist_transform > 0) else 0
        std_distance = np.std(dist_transform[dist_transform > 0]) if np.any(dist_transform > 0) else 0

        distance_stats = {
            'max_distance': max_distance,
            'mean_distance': mean_distance,
            'std_distance': std_distance
        }

        # Update global statistics
        self.stats['successful'] += 1
        self.stats['avg_max_distance'].append(max_distance)
        self.stats['avg_mean_distance'].append(mean_distance)
        self.stats['distance_std_dev'].append(std_distance)

        if self.debug_mode:
            print(f"✅ SUCCESS:")
            print(f"  → Visualization: {self.visualization_mode}")
            print(f"  → Distance type: {self.get_distance_type_name()}")
            print(f"  → Max distance: {max_distance:.2f}px")
            print(f"  → Mean distance: {mean_distance:.2f}px")

        return True, result, distance_stats

    def get_distance_type_name(self) -> str:
        """Get human-readable name for distance type."""
        if self.distance_type == cv2.DIST_L1:
            return "Manhattan (L1)"
        elif self.distance_type == cv2.DIST_L2:
            return "Euclidean (L2)"
        elif self.distance_type == cv2.DIST_C:
            return "Chessboard"
        else:
            return "Unknown"

    def process_directory(
            self,
            input_dir: Path,
            output_dir: Optional[Path] = None,
            file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
    ):
        """
        Process all images in directory to apply distance transform.
        """
        input_dir = Path(input_dir)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in file_extensions
               and not any(suffix in f.stem for suffix in ['_dist', '_distance'])
        ]

        print(f"\n{'=' * 70}")
        print(f"HAND DISTANCE TRANSFORM")
        print(f"{'=' * 70}")
        print(f"Images to process: {len(image_files)}")
        print(f"Visualization mode: {self.visualization_mode}")
        print(f"Distance type: {self.get_distance_type_name()}")
        print(f"Mask size: {self.mask_size}")
        print(f"Smoothing: {'ON' if self.apply_smoothing else 'OFF'}")
        print(f"Enhance ridges: {'ON' if self.enhance_ridges else 'OFF'}")
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
        avg_max = np.mean(self.stats['avg_max_distance']) if self.stats['avg_max_distance'] else 0
        avg_mean = np.mean(self.stats['avg_mean_distance']) if self.stats['avg_mean_distance'] else 0
        avg_std = np.mean(self.stats['distance_std_dev']) if self.stats['distance_std_dev'] else 0

        print(f"\n{'=' * 70}")
        print("DISTANCE TRANSFORM SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total Processed:       {self.stats['processed']}")
        print(f"Successfully Transformed: {self.stats['successful']}")
        print(f"No Hand Pixels:        {self.stats['no_hand_pixels']}")
        print(f"Errors:                {self.stats['errors']}")
        print(f"Success Rate:          {self.stats['successful'] / max(1, self.stats['processed']) * 100:.1f}%")
        print(f"\nDistance Statistics:")
        print(f"Avg Max Distance:      {avg_max:.2f} pixels")
        print(f"Avg Mean Distance:     {avg_mean:.2f} pixels")
        print(f"Avg Std Deviation:     {avg_std:.2f} pixels")
        print(f"Visualization Mode:    {self.visualization_mode}")
        print(f"{'=' * 70}\n")

    def save_processing_report(self, output_dir: Path):
        """Save detailed processing report."""
        avg_max = np.mean(self.stats['avg_max_distance']) if self.stats['avg_max_distance'] else 0
        avg_mean = np.mean(self.stats['avg_mean_distance']) if self.stats['avg_mean_distance'] else 0
        avg_std = np.mean(self.stats['distance_std_dev']) if self.stats['distance_std_dev'] else 0

        report = {
            'processing_date': datetime.now().isoformat(),
            'configuration': {
                'distance_type': self.get_distance_type_name(),
                'mask_size': self.mask_size,
                'visualization_mode': self.visualization_mode,
                'colormap': self.colormap if self.visualization_mode == "colormap" else None,
                'apply_smoothing': self.apply_smoothing,
                'smoothing_kernel': self.smoothing_kernel if self.apply_smoothing else None,
                'binary_threshold': self.binary_threshold,
                'enhance_ridges': self.enhance_ridges,
                'ridge_threshold': self.ridge_threshold if self.enhance_ridges else None
            },
            'statistics': {
                'processed': self.stats['processed'],
                'successful': self.stats['successful'],
                'no_hand_pixels': self.stats['no_hand_pixels'],
                'errors': self.stats['errors'],
                'avg_max_distance': avg_max,
                'avg_mean_distance': avg_mean,
                'avg_std_deviation': avg_std
            },
            'description': 'Distance transform for finger width and hand structure analysis'
        }

        report_path = output_dir / 'distance_transform_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Processing report saved: {report_path}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":

    print("🚀 Starting Hand Distance Transform...")
    print("📋 This script applies distance transform to show hand structure")
    print("📏 Shows distance from each pixel to nearest edge (finger width info)")
    print()

    # Configuration
    INPUT_DIR = "centered_hands"  # Input folder (binary masks work best)
    OUTPUT_DIR = "distance_transformed"  # Output folder

    # Example 1: Normalized grayscale (Simple, clear)
    print("=" * 70)
    print("NORMALIZED DISTANCE TRANSFORM")
    print("=" * 70)

    transformer = HandDistanceTransform(
        distance_type=cv2.DIST_L2,  # Euclidean distance
        mask_size=5,  # 5x5 mask
        visualization_mode="normalized",  # Grayscale
        apply_smoothing=True,  # Smooth the map
        smoothing_kernel=5,  # 5x5 blur
        enhance_ridges=False,  # Don't highlight skeleton yet
        debug_mode=True
    )

    transformer.process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )

    print("\n✅ Distance transform complete!")
    print(f"💡 TIP: Check '{OUTPUT_DIR}' folder to see distance maps")
    print("   Brighter = further from edge, darker = closer to edge")

    # Example 2: Colormap visualization (Heat map style)
    print(f"\n{'=' * 70}")
    print("COLORMAP DISTANCE TRANSFORM (Heat Map)")
    print(f"{'=' * 70}")

    colormap_transformer = HandDistanceTransform(
        distance_type=cv2.DIST_L2,
        mask_size=5,
        visualization_mode="colormap",  # Colormap!
        colormap=cv2.COLORMAP_JET,  # Jet colormap (blue->red)
        apply_smoothing=True,
        smoothing_kernel=5,
        debug_mode=True
    )

    colormap_output = Path(OUTPUT_DIR) / "colormap"
    colormap_transformer.process_directory(
        input_dir=INPUT_DIR,
        output_dir=colormap_output
    )

    # Example 3: Skeleton-weighted (Shows structure)
    print(f"\n{'=' * 70}")
    print("SKELETON-WEIGHTED DISTANCE TRANSFORM")
    print(f"{'=' * 70}")

    skeleton_transformer = HandDistanceTransform(
        distance_type=cv2.DIST_L2,
        mask_size=5,
        visualization_mode="skeleton_weighted",  # With skeleton overlay
        apply_smoothing=True,
        smoothing_kernel=5,
        enhance_ridges=True,  # Highlight medial axis
        ridge_threshold=0.5,
        debug_mode=True
    )

    skeleton_output = Path(OUTPUT_DIR) / "skeleton_weighted"
    skeleton_transformer.process_directory(
        input_dir=INPUT_DIR,
        output_dir=skeleton_output
    )

    # Example 4: Binary threshold (Core regions only)
    print(f"\n{'=' * 70}")
    print("BINARY THRESHOLD DISTANCE TRANSFORM")
    print(f"{'=' * 70}")

    binary_transformer = HandDistanceTransform(
        distance_type=cv2.DIST_L2,
        mask_size=5,
        visualization_mode="binary_threshold",  # Binary output
        binary_threshold=10,  # Pixels > 10px from edge
        apply_smoothing=False,
        debug_mode=True
    )

    binary_output = Path(OUTPUT_DIR) / "binary_threshold"
    binary_transformer.process_directory(
        input_dir=INPUT_DIR,
        output_dir=binary_output
    )

    # Example 5: Different distance metrics
    print(f"\n{'=' * 70}")
    print("TESTING DIFFERENT DISTANCE METRICS")
    print(f"{'=' * 70}")

    distance_types = [
        (cv2.DIST_L1, "manhattan"),
        (cv2.DIST_L2, "euclidean"),
        (cv2.DIST_C, "chessboard")
    ]

    for dist_type, name in distance_types:
        print(f"\nTesting distance type: {name}")
        test_transformer = HandDistanceTransform(
            distance_type=dist_type,
            mask_size=5,
            visualization_mode="normalized",
            apply_smoothing=True,
            debug_mode=False
        )

        test_output = Path(OUTPUT_DIR) / f"distance_{name}"
        test_transformer.process_directory(
            input_dir=INPUT_DIR,
            output_dir=test_output
        )

        print(f"\nDistance type '{name}' results:")
        test_transformer.print_summary()

    print("\n📏 All distance transform methods tested!")
    print("Different visualizations show different aspects of hand structure:")
    print("  • Normalized: Clear grayscale distance map")
    print("  • Colormap: Eye-catching heat map visualization")
    print("  • Skeleton-weighted: Distance + structural information")
    print("  • Binary threshold: Core regions only (palm/thick fingers)")
