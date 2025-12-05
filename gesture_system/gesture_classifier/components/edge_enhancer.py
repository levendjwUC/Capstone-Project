import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from datetime import datetime
import json


class HandEdgeEnhancer:
    """
    Enhances edges in hand images to improve feature detection for CNNs.
    Adds emphasized contours while preserving original hand structure.
    """

    def __init__(
            self,
            enhancement_method: str = "canny_overlay",
            # "canny_overlay", "gradient_overlay", "laplacian_overlay", "morphological_edges"
            edge_thickness: int = 2,  # Thickness of enhanced edges
            edge_intensity: float = 1.0,  # Intensity multiplier for edges (0.5-2.0)
            canny_low_threshold: int = 50,  # Canny edge lower threshold
            canny_high_threshold: int = 150,  # Canny edge higher threshold
            gaussian_blur_kernel: int = 3,  # Pre-processing blur to reduce noise
            morphology_kernel_size: int = 3,  # For morphological edge detection
            preserve_original: bool = True,  # Keep original pixels and add edges
            background_color: Tuple[int, int, int] = (0, 0, 0),  # Background color
            debug_mode: bool = True
    ):
        """
        Initialize edge enhancer.

        Args:
            enhancement_method: Method for edge enhancement
                - "canny_overlay": Canny edges overlaid on original (recommended)
                - "gradient_overlay": Sobel gradient edges overlaid
                - "laplacian_overlay": Laplacian edges overlaid
                - "morphological_edges": Morphological gradient edges
            edge_thickness: Thickness of enhanced edge lines
            edge_intensity: How bright/prominent edges should be
            canny_low_threshold: Lower threshold for Canny edge detection
            canny_high_threshold: Upper threshold for Canny edge detection
            gaussian_blur_kernel: Blur kernel size for noise reduction
            morphology_kernel_size: Kernel size for morphological operations
            preserve_original: If True, adds edges to original; if False, edges only
            background_color: Background color for non-hand areas
            debug_mode: Print detailed processing info
        """
        self.enhancement_method = enhancement_method
        self.edge_thickness = edge_thickness
        self.edge_intensity = edge_intensity
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.morphology_kernel_size = morphology_kernel_size
        self.preserve_original = preserve_original
        self.background_color = background_color
        self.debug_mode = debug_mode

        # Validate parameters
        self.edge_intensity = max(0.1, min(3.0, self.edge_intensity))
        if self.gaussian_blur_kernel % 2 == 0:
            self.gaussian_blur_kernel += 1  # Must be odd
        if self.morphology_kernel_size % 2 == 0:
            self.morphology_kernel_size += 1  # Must be odd

        # Statistics
        self.stats = {
            'processed': 0,
            'successful': 0,
            'no_edges_found': 0,
            'avg_edge_pixels': [],
            'edge_coverage_percent': [],
            'errors': 0
        }

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for edge detection.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur to reduce noise
        if self.gaussian_blur_kernel > 1:
            blurred = cv2.GaussianBlur(gray, (self.gaussian_blur_kernel, self.gaussian_blur_kernel), 0)
        else:
            blurred = gray

        return blurred

    def detect_canny_edges(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Detect edges using Canny edge detection.
        Most popular and reliable method.
        """
        edges = cv2.Canny(
            preprocessed,
            self.canny_low_threshold,
            self.canny_high_threshold
        )

        if self.debug_mode:
            edge_count = np.count_nonzero(edges)
            print(f"  → Canny edges detected: {edge_count:,} pixels")

        return edges

    def detect_gradient_edges(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Detect edges using Sobel gradient.
        Good for detecting directional edges.
        """
        # Calculate gradients
        grad_x = cv2.Sobel(preprocessed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(preprocessed, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize to 0-255
        gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)

        # Threshold to get binary edges
        _, edges = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

        if self.debug_mode:
            edge_count = np.count_nonzero(edges)
            print(f"  → Gradient edges detected: {edge_count:,} pixels")

        return edges

    def detect_laplacian_edges(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Detect edges using Laplacian operator.
        Good for detecting rapid intensity changes.
        """
        # Apply Laplacian
        laplacian = cv2.Laplacian(preprocessed, cv2.CV_64F)

        # Take absolute value and normalize
        laplacian_abs = np.absolute(laplacian)
        laplacian_norm = np.uint8(laplacian_abs / laplacian_abs.max() * 255)

        # Threshold to get binary edges
        _, edges = cv2.threshold(laplacian_norm, 30, 255, cv2.THRESH_BINARY)

        if self.debug_mode:
            edge_count = np.count_nonzero(edges)
            print(f"  → Laplacian edges detected: {edge_count:,} pixels")

        return edges

    def detect_morphological_edges(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Detect edges using morphological gradient.
        Good for thick, continuous edges.
        """
        # Create morphological kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )

        # Apply morphological gradient (dilation - erosion)
        gradient = cv2.morphologyEx(preprocessed, cv2.MORPH_GRADIENT, kernel)

        # Threshold to get binary edges
        _, edges = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)

        if self.debug_mode:
            edge_count = np.count_nonzero(edges)
            print(f"  → Morphological edges detected: {edge_count:,} pixels")

        return edges

    def get_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Get edges using specified method.
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image)

        # Detect edges based on method
        if self.enhancement_method == "canny_overlay":
            edges = self.detect_canny_edges(preprocessed)
        elif self.enhancement_method == "gradient_overlay":
            edges = self.detect_gradient_edges(preprocessed)
        elif self.enhancement_method == "laplacian_overlay":
            edges = self.detect_laplacian_edges(preprocessed)
        elif self.enhancement_method == "morphological_edges":
            edges = self.detect_morphological_edges(preprocessed)
        else:
            # Default to Canny
            edges = self.detect_canny_edges(preprocessed)

        return edges

    def thicken_edges(self, edges: np.ndarray) -> np.ndarray:
        """
        Thicken edge lines for better visibility.
        """
        if self.edge_thickness <= 1:
            return edges

        # Create kernel for thickening
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.edge_thickness, self.edge_thickness)
        )

        # Dilate edges to make them thicker
        thickened = cv2.dilate(edges, kernel, iterations=1)

        if self.debug_mode:
            original_count = np.count_nonzero(edges)
            thickened_count = np.count_nonzero(thickened)
            print(f"  → Edges thickened: {original_count:,} → {thickened_count:,} pixels")

        return thickened

    def apply_edge_intensity(self, edges: np.ndarray) -> np.ndarray:
        """
        Apply intensity scaling to edges.
        """
        if self.edge_intensity == 1.0:
            return edges

        # Scale edge intensity
        scaled_edges = (edges * self.edge_intensity).astype(np.uint8)
        scaled_edges = np.clip(scaled_edges, 0, 255)

        return scaled_edges

    def overlay_edges_on_image(self, original: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Overlay detected edges on original image.
        """
        if not self.preserve_original:
            # Return edges only (white edges on black background)
            if len(original.shape) == 3:
                result = np.zeros_like(original)
                result[edges > 0] = [255, 255, 255]
            else:
                result = edges.copy()
            return result

        # Overlay edges on original image
        result = original.copy()

        if len(result.shape) == 3:
            # Color image
            result[edges > 0] = [255, 255, 255]  # White edges
        else:
            # Grayscale image
            result = cv2.add(result, edges)
            result = np.clip(result, 0, 255)

        return result

    def process_image(
            self,
            image_path: Optional[Path] = None,
            output_path: Optional[Path] = None,
            original_image: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[np.ndarray], Optional[int]]:
        """
        Process image to enhance edges.

        Returns:
            Tuple of (success, enhanced_image, edge_count)
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

        # Detect edges
        edges = self.get_edges(image)

        # Check if any edges found
        edge_count = np.count_nonzero(edges)
        if edge_count == 0:
            self.stats['no_edges_found'] += 1
            if self.debug_mode:
                print(f"⚠️ No edges detected")

            # Return original image if no edges
            if output_path:
                cv2.imwrite(str(output_path), image)
            return False, image, 0

        # Thicken edges if requested
        if self.edge_thickness > 1:
            edges = self.thicken_edges(edges)
            edge_count = np.count_nonzero(edges)

        # Apply intensity scaling
        edges = self.apply_edge_intensity(edges)

        # Overlay edges on original image
        enhanced_image = self.overlay_edges_on_image(image, edges)

        # Save result if output path provided
        if output_path:
            cv2.imwrite(str(output_path), enhanced_image)

        # Update statistics
        self.stats['successful'] += 1
        self.stats['avg_edge_pixels'].append(edge_count)

        # Calculate edge coverage percentage
        total_pixels = image.shape[0] * image.shape[1]
        coverage_percent = (edge_count / total_pixels) * 100
        self.stats['edge_coverage_percent'].append(coverage_percent)

        if self.debug_mode:
            print(f"✅ SUCCESS:")
            print(f"  → Method: {self.enhancement_method}")
            print(f"  → Edge pixels: {edge_count:,} ({coverage_percent:.1f}% of image)")
            print(f"  → Edge thickness: {self.edge_thickness}px")
            print(f"  → Edge intensity: {self.edge_intensity}x")
            print(f"  → Preserve original: {self.preserve_original}")

        return True, enhanced_image, edge_count

    def process_directory(
            self,
            input_dir: Path,
            output_dir: Optional[Path] = None,
            file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
    ):
        """
        Process all images in directory to enhance edges.
        """
        input_dir = Path(input_dir)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in file_extensions
               and not any(suffix in f.stem for suffix in ['_edges', '_enhanced'])
        ]

        print(f"\n{'=' * 70}")
        print(f"HAND EDGE ENHANCER")
        print(f"{'=' * 70}")
        print(f"Images to process: {len(image_files)}")
        print(f"Enhancement method: {self.enhancement_method}")
        print(f"Edge thickness: {self.edge_thickness}px")
        print(f"Edge intensity: {self.edge_intensity}x")
        print(f"Preserve original: {self.preserve_original}")
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
        avg_edges = np.mean(self.stats['avg_edge_pixels']) if self.stats['avg_edge_pixels'] else 0
        avg_coverage = np.mean(self.stats['edge_coverage_percent']) if self.stats['edge_coverage_percent'] else 0

        print(f"\n{'=' * 70}")
        print("EDGE ENHANCEMENT SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total Processed:       {self.stats['processed']}")
        print(f"Successfully Enhanced: {self.stats['successful']}")
        print(f"No Edges Found:        {self.stats['no_edges_found']}")
        print(f"Errors:                {self.stats['errors']}")
        print(f"Success Rate:          {self.stats['successful'] / max(1, self.stats['processed']) * 100:.1f}%")
        print(f"\nEdge Statistics:")
        print(f"Average Edge Pixels:   {avg_edges:,.0f}")
        print(f"Average Coverage:      {avg_coverage:.1f}% of image")
        print(f"Method Used:           {self.enhancement_method}")
        print(f"{'=' * 70}\n")

    def save_processing_report(self, output_dir: Path):
        """Save detailed processing report."""
        avg_edges = np.mean(self.stats['avg_edge_pixels']) if self.stats['avg_edge_pixels'] else 0
        avg_coverage = np.mean(self.stats['edge_coverage_percent']) if self.stats['edge_coverage_percent'] else 0

        report = {
            'processing_date': datetime.now().isoformat(),
            'configuration': {
                'enhancement_method': self.enhancement_method,
                'edge_thickness': self.edge_thickness,
                'edge_intensity': self.edge_intensity,
                'canny_low_threshold': self.canny_low_threshold,
                'canny_high_threshold': self.canny_high_threshold,
                'gaussian_blur_kernel': self.gaussian_blur_kernel,
                'morphology_kernel_size': self.morphology_kernel_size,
                'preserve_original': self.preserve_original,
                'background_color': self.background_color
            },
            'statistics': {
                'processed': self.stats['processed'],
                'successful': self.stats['successful'],
                'no_edges_found': self.stats['no_edges_found'],
                'errors': self.stats['errors'],
                'avg_edge_pixels': avg_edges,
                'avg_coverage_percent': avg_coverage
            },
            'description': 'Edge enhancement for improved CNN feature detection'
        }

        report_path = output_dir / 'edge_enhancement_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Processing report saved: {report_path}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":

    print("🚀 Starting Hand Edge Enhancer...")
    print("📋 This script enhances edges in hand images for better CNN training")
    print()

    # Configuration
    INPUT_DIR = "centered_hands"  # Input folder (after centering)
    OUTPUT_DIR = "edge_enhanced_hands"  # Output folder

    # Example 1: Canny edge overlay (RECOMMENDED)
    print("=" * 70)
    print("CANNY EDGE ENHANCEMENT (Recommended)")
    print("=" * 70)

    enhancer = HandEdgeEnhancer(
        enhancement_method="canny_overlay",  # Most reliable method
        edge_thickness=2,  # 2-pixel thick edges
        edge_intensity=1.0,  # Normal intensity
        canny_low_threshold=50,  # Lower threshold
        canny_high_threshold=150,  # Upper threshold
        gaussian_blur_kernel=3,  # Small blur for noise reduction
        preserve_original=True,  # Keep original + add edges
        background_color=(0, 0, 0),  # Black background
        debug_mode=True
    )

    enhancer.process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )

    print("\n✅ Edge enhancement complete!")
    print(f"💡 TIP: Check '{OUTPUT_DIR}' folder to see enhanced edges")
    print("   Edges should be clearly visible for better CNN feature detection!")

    # Example 2: Test different methods
    print(f"\n{'=' * 70}")
    print("TESTING DIFFERENT ENHANCEMENT METHODS")
    print(f"{'=' * 70}")

    methods = [
        ("gradient_overlay", "Sobel gradient edges"),
        ("laplacian_overlay", "Laplacian edges"),
        ("morphological_edges", "Morphological gradient")
    ]

    for method, description in methods:
        print(f"\nTesting method: {method} ({description})")
        test_enhancer = HandEdgeEnhancer(
            enhancement_method=method,
            edge_thickness=2,
            edge_intensity=1.0,
            debug_mode=False
        )

        test_output = Path(OUTPUT_DIR) / f"method_{method}"
        test_enhancer.process_directory(
            input_dir=INPUT_DIR,
            output_dir=test_output
        )

        print(f"\nMethod '{method}' results:")
        test_enhancer.print_summary()

    # Example 3: Edge-only output (no original pixels)
    print(f"\n{'=' * 70}")
    print("EDGES-ONLY OUTPUT (For specialized models)")
    print(f"{'=' * 70}")

    edges_only_enhancer = HandEdgeEnhancer(
        enhancement_method="canny_overlay",
        edge_thickness=3,
        edge_intensity=1.0,
        preserve_original=False,  # Edges only!
        debug_mode=True
    )

    edges_only_output = Path(OUTPUT_DIR) / "edges_only"
    edges_only_enhancer.process_directory(
        input_dir=INPUT_DIR,
        output_dir=edges_only_output
    )

    print("\n🎯 All edge enhancement methods tested!")
    print("Compare the different outputs to see which works best for your model.")
