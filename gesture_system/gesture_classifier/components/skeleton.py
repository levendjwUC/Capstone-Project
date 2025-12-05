import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from datetime import datetime
import json
from skimage import morphology
from scipy import ndimage


class HandSkeletonExtractor:
    """
    Extracts morphological skeleton from hand images.
    Reduces hand to 1-pixel wide representation for shape-based analysis.
    """

    def __init__(
            self,
            skeletonization_method: str = "zhang_suen",  # "zhang_suen", "medial_axis", "morphological", "thinning"
            skeleton_thickness: int = 1,  # Thickness of skeleton lines (1-3)
            enhance_junctions: bool = True,  # Highlight branch points and endpoints
            junction_radius: int = 3,  # Radius for junction highlighting
            preserve_topology: bool = True,  # Preserve topological structure
            min_branch_length: int = 5,  # Minimum branch length to keep (pruning)
            apply_pruning: bool = False,  # Remove small branches
            background_color: Tuple[int, int, int] = (0, 0, 0),  # Background color
            skeleton_color: Tuple[int, int, int] = (255, 255, 255),  # Skeleton color
            junction_color: Tuple[int, int, int] = (200, 200, 200),  # Junction point color
            debug_mode: bool = True
    ):
        """
        Initialize skeleton extractor.

        Args:
            skeletonization_method: Method for skeleton extraction
                - "zhang_suen": Zhang-Suen thinning algorithm (fast, good)
                - "medial_axis": Medial axis transform (smooth, preserves structure)
                - "morphological": Morphological thinning (simple)
                - "thinning": Lee's thinning algorithm (preserves connectivity)
            skeleton_thickness: Thickness of skeleton lines in final output
            enhance_junctions: Highlight branch points and endpoints
            junction_radius: Size of junction point markers
            preserve_topology: Maintain topological correctness
            min_branch_length: Minimum length for branches (for pruning)
            apply_pruning: Remove small spurious branches
            background_color: Background color
            skeleton_color: Color for skeleton lines
            junction_color: Color for junction markers
            debug_mode: Print detailed processing info
        """
        self.skeletonization_method = skeletonization_method
        self.skeleton_thickness = max(1, min(5, skeleton_thickness))
        self.enhance_junctions = enhance_junctions
        self.junction_radius = junction_radius
        self.preserve_topology = preserve_topology
        self.min_branch_length = min_branch_length
        self.apply_pruning = apply_pruning
        self.background_color = background_color
        self.skeleton_color = skeleton_color
        self.junction_color = junction_color
        self.debug_mode = debug_mode

        # Statistics
        self.stats = {
            'processed': 0,
            'successful': 0,
            'no_hand_pixels': 0,
            'avg_skeleton_pixels': [],
            'avg_branch_points': [],
            'avg_end_points': [],
            'skeleton_coverage_percent': [],
            'errors': 0
        }

    def preprocess_for_skeletonization(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for skeletonization.
        Convert to binary format required by skeleton algorithms.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Threshold to binary
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Convert to boolean for skimage
        binary_bool = binary > 0

        return binary_bool

    def extract_zhang_suen_skeleton(self, binary: np.ndarray) -> np.ndarray:
        """
        Extract skeleton using Zhang-Suen thinning algorithm.
        Fast and produces good results for hand shapes.
        """
        try:
            skeleton = morphology.skeletonize(binary)

            if self.debug_mode:
                skeleton_pixels = np.count_nonzero(skeleton)
                print(f"  → Zhang-Suen skeleton pixels: {skeleton_pixels:,}")

            return skeleton
        except Exception as e:
            if self.debug_mode:
                print(f"  ❌ Zhang-Suen failed: {e}")
            return None

    def extract_medial_axis_skeleton(self, binary: np.ndarray) -> np.ndarray:
        """
        Extract skeleton using medial axis transform.
        Produces smooth skeleton that follows the shape's medial axis.
        """
        try:
            from skimage.morphology import medial_axis as skimage_medial_axis

            skeleton, distance = skimage_medial_axis(binary, return_distance=True)

            if self.debug_mode:
                skeleton_pixels = np.count_nonzero(skeleton)
                print(f"  → Medial axis skeleton pixels: {skeleton_pixels:,}")

            return skeleton
        except Exception as e:
            if self.debug_mode:
                print(f"  ❌ Medial axis failed: {e}")
            return None

    def extract_morphological_skeleton(self, binary: np.ndarray) -> np.ndarray:
        """
        Extract skeleton using morphological operations.
        Simple iterative thinning approach.
        """
        try:
            # Convert to uint8 for cv2
            binary_uint8 = binary.astype(np.uint8) * 255

            # Morphological skeleton
            skeleton = np.zeros_like(binary_uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

            done = False
            while not done:
                eroded = cv2.erode(binary_uint8, element)
                opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
                subset = eroded - opened
                skeleton = cv2.bitwise_or(skeleton, subset)
                binary_uint8 = eroded.copy()

                if cv2.countNonZero(binary_uint8) == 0:
                    done = True

            skeleton_bool = skeleton > 0

            if self.debug_mode:
                skeleton_pixels = np.count_nonzero(skeleton_bool)
                print(f"  → Morphological skeleton pixels: {skeleton_pixels:,}")

            return skeleton_bool
        except Exception as e:
            if self.debug_mode:
                print(f"  ❌ Morphological skeleton failed: {e}")
            return None

    def extract_thinning_skeleton(self, binary: np.ndarray) -> np.ndarray:
        """
        Extract skeleton using cv2 thinning (Guo-Hall algorithm).
        Good balance of speed and quality.
        """
        try:
            # Convert to uint8 for cv2
            binary_uint8 = binary.astype(np.uint8) * 255

            # Apply cv2 thinning
            skeleton = cv2.ximgproc.thinning(binary_uint8, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

            skeleton_bool = skeleton > 0

            if self.debug_mode:
                skeleton_pixels = np.count_nonzero(skeleton_bool)
                print(f"  → Thinning skeleton pixels: {skeleton_pixels:,}")

            return skeleton_bool
        except AttributeError:
            # cv2.ximgproc might not be available
            if self.debug_mode:
                print(f"  ⚠️ cv2.ximgproc not available, falling back to Zhang-Suen")
            return self.extract_zhang_suen_skeleton(binary)
        except Exception as e:
            if self.debug_mode:
                print(f"  ❌ Thinning failed: {e}")
            return None

    def get_skeleton(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get skeleton using specified method.
        """
        # Preprocess image
        binary = self.preprocess_for_skeletonization(image)

        # Check if any pixels to skeletonize
        if not np.any(binary):
            if self.debug_mode:
                print(f"  ⚠️ No hand pixels to skeletonize")
            return None

        # Extract skeleton based on method
        if self.skeletonization_method == "zhang_suen":
            skeleton = self.extract_zhang_suen_skeleton(binary)
        elif self.skeletonization_method == "medial_axis":
            skeleton = self.extract_medial_axis_skeleton(binary)
        elif self.skeletonization_method == "morphological":
            skeleton = self.extract_morphological_skeleton(binary)
        elif self.skeletonization_method == "thinning":
            skeleton = self.extract_thinning_skeleton(binary)
        else:
            # Default to zhang_suen
            skeleton = self.extract_zhang_suen_skeleton(binary)

        return skeleton

    def prune_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Prune small branches from skeleton.
        Removes spurious short branches while keeping main structure.
        """
        if not self.apply_pruning:
            return skeleton

        # Convert to uint8
        skeleton_uint8 = skeleton.astype(np.uint8) * 255

        # Find endpoints
        endpoints = self.find_endpoints(skeleton)

        # For each endpoint, trace back and remove if branch is too short
        pruned = skeleton.copy()

        for y, x in zip(*np.where(endpoints)):
            # Trace branch length
            branch_length = self.trace_branch_length(skeleton, (x, y))

            if branch_length < self.min_branch_length:
                # Remove this short branch
                self.remove_branch(pruned, (x, y), branch_length)

        if self.debug_mode:
            original_pixels = np.count_nonzero(skeleton)
            pruned_pixels = np.count_nonzero(pruned)
            removed = original_pixels - pruned_pixels
            print(f"  → Pruned {removed:,} pixels from short branches")

        return pruned

    def trace_branch_length(self, skeleton: np.ndarray, start: Tuple[int, int]) -> int:
        """
        Trace the length of a branch starting from an endpoint.
        """
        # Simple implementation: count pixels until junction or end
        visited = set()
        current = start
        length = 0

        while True:
            visited.add(current)
            length += 1

            # Find neighbors
            neighbors = self.get_skeleton_neighbors(skeleton, current)
            unvisited = [n for n in neighbors if n not in visited]

            if len(unvisited) == 0:
                break  # Dead end
            elif len(unvisited) == 1:
                current = unvisited[0]  # Continue along branch
            else:
                break  # Junction point

            if length > self.min_branch_length:
                break  # Branch is long enough

        return length

    def get_skeleton_neighbors(self, skeleton: np.ndarray, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get neighboring skeleton pixels (8-connected).
        """
        x, y = pos
        h, w = skeleton.shape
        neighbors = []

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skeleton[ny, nx]:
                    neighbors.append((nx, ny))

        return neighbors

    def remove_branch(self, skeleton: np.ndarray, start: Tuple[int, int], max_length: int):
        """
        Remove a branch starting from an endpoint.
        """
        visited = set()
        current = start
        length = 0

        while length < max_length:
            skeleton[current[1], current[0]] = False
            visited.add(current)

            neighbors = self.get_skeleton_neighbors(skeleton, current)
            unvisited = [n for n in neighbors if n not in visited]

            if len(unvisited) == 0:
                break
            elif len(unvisited) == 1:
                current = unvisited[0]
                length += 1
            else:
                break  # Junction, stop removing

    def find_endpoints(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Find skeleton endpoints (pixels with only 1 neighbor).
        """
        # Create kernel to count neighbors
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0

        # Count neighbors for each skeleton pixel
        skeleton_uint8 = skeleton.astype(np.uint8)
        neighbor_count = ndimage.convolve(skeleton_uint8, kernel, mode='constant')

        # Endpoints have exactly 1 neighbor
        endpoints = (skeleton_uint8 == 1) & (neighbor_count == 1)

        return endpoints

    def find_branch_points(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Find skeleton branch points (pixels with 3+ neighbors).
        """
        # Create kernel to count neighbors
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0

        # Count neighbors for each skeleton pixel
        skeleton_uint8 = skeleton.astype(np.uint8)
        neighbor_count = ndimage.convolve(skeleton_uint8, kernel, mode='constant')

        # Branch points have 3 or more neighbors
        branch_points = (skeleton_uint8 == 1) & (neighbor_count >= 3)

        return branch_points

    def thicken_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Thicken skeleton for better visibility.
        """
        if self.skeleton_thickness <= 1:
            return skeleton

        # Convert to uint8
        skeleton_uint8 = skeleton.astype(np.uint8) * 255

        # Dilate to thicken
        kernel_size = self.skeleton_thickness * 2 - 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        thickened = cv2.dilate(skeleton_uint8, kernel, iterations=1)

        thickened_bool = thickened > 0

        return thickened_bool

    def create_skeleton_image(
            self,
            skeleton: np.ndarray,
            original_shape: Tuple[int, int, int],
            endpoints: Optional[np.ndarray] = None,
            branch_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create final skeleton image with optional junction highlighting.
        """
        h, w = original_shape[:2]

        # Create output image
        if len(original_shape) == 3:
            result = np.zeros((h, w, 3), dtype=np.uint8)
            result[:] = self.background_color
        else:
            result = np.zeros((h, w), dtype=np.uint8)

        # Draw skeleton
        if len(original_shape) == 3:
            result[skeleton] = self.skeleton_color
        else:
            result[skeleton] = 255

        # Enhance junctions if requested
        if self.enhance_junctions and len(original_shape) == 3:
            if endpoints is not None:
                # Draw endpoints
                endpoint_coords = np.argwhere(endpoints)
                for y, x in endpoint_coords:
                    cv2.circle(result, (x, y), self.junction_radius,
                               (255, 255, 255), -1)  # White circles

            if branch_points is not None:
                # Draw branch points
                branch_coords = np.argwhere(branch_points)
                for y, x in branch_coords:
                    cv2.circle(result, (x, y), self.junction_radius,
                               self.junction_color, -1)  # Colored circles

        return result

    def process_image(
            self,
            image_path: Optional[Path] = None,
            output_path: Optional[Path] = None,
            original_image: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[np.ndarray], Optional[Dict]]:
        """
        Process image to extract skeleton.

        Returns:
            Tuple of (success, skeleton_image, skeleton_stats)
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

        # Extract skeleton
        skeleton = self.get_skeleton(image)

        if skeleton is None or not np.any(skeleton):
            self.stats['no_hand_pixels'] += 1
            if self.debug_mode:
                print(f"⚠️ No skeleton extracted")
            return False, image, None

        # Apply pruning if requested
        if self.apply_pruning:
            skeleton = self.prune_skeleton(skeleton)

        # Find junctions
        endpoints = self.find_endpoints(skeleton)
        branch_points = self.find_branch_points(skeleton)

        # Thicken skeleton if requested
        if self.skeleton_thickness > 1:
            skeleton_display = self.thicken_skeleton(skeleton)
        else:
            skeleton_display = skeleton

        # Create final skeleton image
        skeleton_image = self.create_skeleton_image(
            skeleton_display,
            image.shape,
            endpoints if self.enhance_junctions else None,
            branch_points if self.enhance_junctions else None
        )

        # Save result if output path provided
        if output_path:
            cv2.imwrite(str(output_path), skeleton_image)

        # Calculate statistics
        skeleton_pixels = np.count_nonzero(skeleton)
        num_endpoints = np.count_nonzero(endpoints)
        num_branch_points = np.count_nonzero(branch_points)
        total_pixels = image.shape[0] * image.shape[1]
        coverage_percent = (skeleton_pixels / total_pixels) * 100

        skeleton_stats = {
            'skeleton_pixels': skeleton_pixels,
            'endpoints': num_endpoints,
            'branch_points': num_branch_points,
            'coverage_percent': coverage_percent
        }

        # Update global statistics
        self.stats['successful'] += 1
        self.stats['avg_skeleton_pixels'].append(skeleton_pixels)
        self.stats['avg_branch_points'].append(num_branch_points)
        self.stats['avg_end_points'].append(num_endpoints)
        self.stats['skeleton_coverage_percent'].append(coverage_percent)

        if self.debug_mode:
            print(f"✅ SUCCESS:")
            print(f"  → Method: {self.skeletonization_method}")
            print(f"  → Skeleton pixels: {skeleton_pixels:,} ({coverage_percent:.2f}% of image)")
            print(f"  → Endpoints: {num_endpoints}")
            print(f"  → Branch points: {num_branch_points}")
            print(f"  → Skeleton thickness: {self.skeleton_thickness}px")
            print(f"  → Junctions enhanced: {self.enhance_junctions}")

        return True, skeleton_image, skeleton_stats

    def process_directory(
            self,
            input_dir: Path,
            output_dir: Optional[Path] = None,
            file_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
    ):
        """
        Process all images in directory to extract skeletons.
        """
        input_dir = Path(input_dir)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in file_extensions
               and not any(suffix in f.stem for suffix in ['_skeleton', '_skel'])
        ]

        print(f"\n{'=' * 70}")
        print(f"HAND SKELETON EXTRACTOR")
        print(f"{'=' * 70}")
        print(f"Images to process: {len(image_files)}")
        print(f"Skeletonization method: {self.skeletonization_method}")
        print(f"Skeleton thickness: {self.skeleton_thickness}px")
        print(f"Enhance junctions: {self.enhance_junctions}")
        print(f"Apply pruning: {self.apply_pruning}")
        if self.apply_pruning:
            print(f"Min branch length: {self.min_branch_length}px")
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
        avg_skeleton = np.mean(self.stats['avg_skeleton_pixels']) if self.stats['avg_skeleton_pixels'] else 0
        avg_endpoints = np.mean(self.stats['avg_end_points']) if self.stats['avg_end_points'] else 0
        avg_branches = np.mean(self.stats['avg_branch_points']) if self.stats['avg_branch_points'] else 0
        avg_coverage = np.mean(self.stats['skeleton_coverage_percent']) if self.stats[
            'skeleton_coverage_percent'] else 0

        print(f"\n{'=' * 70}")
        print("SKELETON EXTRACTION SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total Processed:       {self.stats['processed']}")
        print(f"Successfully Extracted:{self.stats['successful']}")
        print(f"No Hand Pixels:        {self.stats['no_hand_pixels']}")
        print(f"Errors:                {self.stats['errors']}")
        print(f"Success Rate:          {self.stats['successful'] / max(1, self.stats['processed']) * 100:.1f}%")
        print(f"\nSkeleton Statistics:")
        print(f"Avg Skeleton Pixels:   {avg_skeleton:,.0f}")
        print(f"Avg Coverage:          {avg_coverage:.2f}% of image")
        print(f"Avg Endpoints:         {avg_endpoints:.1f}")
        print(f"Avg Branch Points:     {avg_branches:.1f}")
        print(f"Method Used:           {self.skeletonization_method}")
        print(f"{'=' * 70}\n")

    def save_processing_report(self, output_dir: Path):
        """Save detailed processing report."""
        avg_skeleton = np.mean(self.stats['avg_skeleton_pixels']) if self.stats['avg_skeleton_pixels'] else 0
        avg_endpoints = np.mean(self.stats['avg_end_points']) if self.stats['avg_end_points'] else 0
        avg_branches = np.mean(self.stats['avg_branch_points']) if self.stats['avg_branch_points'] else 0
        avg_coverage = np.mean(self.stats['skeleton_coverage_percent']) if self.stats[
            'skeleton_coverage_percent'] else 0

        report = {
            'processing_date': datetime.now().isoformat(),
            'configuration': {
                'skeletonization_method': self.skeletonization_method,
                'skeleton_thickness': self.skeleton_thickness,
                'enhance_junctions': self.enhance_junctions,
                'junction_radius': self.junction_radius,
                'preserve_topology': self.preserve_topology,
                'apply_pruning': self.apply_pruning,
                'min_branch_length': self.min_branch_length if self.apply_pruning else None,
                'background_color': self.background_color,
                'skeleton_color': self.skeleton_color
            },
            'statistics': {
                'processed': self.stats['processed'],
                'successful': self.stats['successful'],
                'no_hand_pixels': self.stats['no_hand_pixels'],
                'errors': self.stats['errors'],
                'avg_skeleton_pixels': avg_skeleton,
                'avg_endpoints': avg_endpoints,
                'avg_branch_points': avg_branches,
                'avg_coverage_percent': avg_coverage
            },
            'description': 'Skeleton extraction for shape-based hand gesture analysis'
        }

        report_path = output_dir / 'skeleton_extraction_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Processing report saved: {report_path}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":

    print("🚀 Starting Hand Skeleton Extractor...")
    print("📋 This script extracts morphological skeletons from hand images")
    print("🦴 Reduces hand to 1-pixel wide representation for shape analysis")
    print()

    # Configuration
    INPUT_DIR = "centered_hands"  # Input folder (binary masks work best)
    OUTPUT_DIR = "skeleton_extracted"  # Output folder

    # Example 1: Zhang-Suen skeleton (RECOMMENDED)
    print("=" * 70)
    print("ZHANG-SUEN SKELETON EXTRACTION (Recommended)")
    print("=" * 70)

    extractor = HandSkeletonExtractor(
        skeletonization_method="zhang_suen",  # Fast and reliable
        skeleton_thickness=2,  # 2px for visibility
        enhance_junctions=True,  # Highlight branch points
        junction_radius=3,  # Size of junction markers
        preserve_topology=True,  # Maintain structure
        apply_pruning=False,  # Don't remove branches initially
        min_branch_length=5,  # Min length if pruning
        background_color=(0, 0, 0),  # Black background
        skeleton_color=(255, 255, 255),  # White skeleton
        junction_color=(200, 200, 200),  # Gray junctions
        debug_mode=True
    )

    extractor.process_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )

    print("\n✅ Skeleton extraction complete!")
    print(f"💡 TIP: Check '{OUTPUT_DIR}' folder to see extracted skeletons")
    print("   Skeletons show the topological structure of hand shapes!")

    # Example 2: Test different methods
    print(f"\n{'=' * 70}")
    print("TESTING DIFFERENT SKELETONIZATION METHODS")
    print(f"{'=' * 70}")

    methods = [
        ("medial_axis", "Medial axis transform"),
        ("morphological", "Morphological thinning"),
        ("thinning", "Guo-Hall thinning")
    ]

    for method, description in methods:
        print(f"\nTesting method: {method} ({description})")
        test_extractor = HandSkeletonExtractor(
            skeletonization_method=method,
            skeleton_thickness=2,
            enhance_junctions=True,
            debug_mode=False
        )

        test_output = Path(OUTPUT_DIR) / f"method_{method}"
        test_extractor.process_directory(
            input_dir=INPUT_DIR,
            output_dir=test_output
        )

        print(f"\nMethod '{method}' results:")
        test_extractor.print_summary()

    # Example 3: Skeleton with pruning (remove noise)
    print(f"\n{'=' * 70}")
    print("SKELETON WITH PRUNING (Remove small branches)")
    print(f"{'=' * 70}")

    pruned_extractor = HandSkeletonExtractor(
        skeletonization_method="zhang_suen",
        skeleton_thickness=2,
        enhance_junctions=True,
        apply_pruning=True,  # Enable pruning
        min_branch_length=10,  # Remove branches < 10px
        debug_mode=True
    )

    pruned_output = Path(OUTPUT_DIR) / "pruned"
    pruned_extractor.process_directory(
        input_dir=INPUT_DIR,
        output_dir=pruned_output
    )

    print("\n🦴 All skeleton extraction methods tested!")
    print("Compare different outputs to see which captures hand structure best.")
