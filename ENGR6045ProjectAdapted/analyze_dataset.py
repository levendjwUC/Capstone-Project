# file: analyze_dataset.py
# !/usr/bin/env python3
"""
Analyze the dataset to determine if we have enough clean data for training.
Provides statistics on images, class distribution, and data quality.
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import hashlib


def get_file_hash(filepath):
    """Get MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None


def analyze_image_quality(image_path):
    """Analyze image quality metrics."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        # Basic metrics
        height, width = img.shape[:2]

        # Check if image is too dark or too bright
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Check if image is mostly black or white
        black_pixels = np.sum(gray < 10)
        white_pixels = np.sum(gray > 245)
        total_pixels = height * width

        black_ratio = black_pixels / total_pixels
        white_ratio = white_pixels / total_pixels

        # Simple blur detection (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()

        return {
            'dimensions': (width, height),
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'black_ratio': float(black_ratio),
            'white_ratio': float(white_ratio),
            'blur_score': float(blur_score),
            'file_size_kb': image_path.stat().st_size / 1024
        }
    except Exception as e:
        return {'error': str(e)}


def analyze_directory(directory, recursive=True):
    """Analyze all images in a directory."""

    dir_path = Path(directory)
    if not dir_path.exists():
        return None

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

    # Collect all images
    if recursive:
        all_files = []
        for ext in image_extensions:
            all_files.extend(dir_path.rglob(f'*{ext}'))
            all_files.extend(dir_path.rglob(f'*{ext.upper()}'))
    else:
        all_files = []
        for ext in image_extensions:
            all_files.extend(dir_path.glob(f'*{ext}'))
            all_files.extend(dir_path.glob(f'*{ext.upper()}'))

    # Only keep files (not directories)
    image_files = [f for f in all_files if f.is_file()]

    # Group by subdirectory (class/gesture)
    by_class = defaultdict(list)
    for img_file in image_files:
        # Determine class name
        if img_file.parent == dir_path:
            class_name = 'root'
        else:
            # Get the immediate parent directory name
            class_name = img_file.parent.name
        by_class[class_name].append(img_file)

    return {
        'total_images': len(image_files),
        'by_class': dict(by_class),
        'all_files': image_files
    }


def find_duplicates(image_files):
    """Find duplicate images by hash."""
    hashes = defaultdict(list)

    print("  Checking for duplicates...", end='', flush=True)
    for img_file in image_files:
        file_hash = get_file_hash(img_file)
        if file_hash:
            hashes[file_hash].append(img_file)

    duplicates = {k: v for k, v in hashes.items() if len(v) > 1}
    print(f" found {len(duplicates)} duplicate groups")

    return duplicates


def main():
    print("=" * 80)
    print("HANDCROPPER DATASET ANALYSIS")
    print("=" * 80)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    data_dir = Path('data')

    if not data_dir.exists():
        print("❌ 'data' directory not found!")
        return

    # Analyze each subdirectory
    subdirs = ['raw', 'processed', 'test_samples']

    analysis_results = {}

    for subdir in subdirs:
        subdir_path = data_dir / subdir

        print(f"\n{'─' * 80}")
        print(f"📁 Analyzing: data/{subdir}/")
        print(f"{'─' * 80}")

        if not subdir_path.exists():
            print(f"  ⚠️  Directory not found")
            continue

        analysis = analyze_directory(subdir_path)

        if analysis is None or analysis['total_images'] == 0:
            print(f"  ⚠️  No images found")
            continue

        print(f"\n  📊 OVERVIEW:")
        print(f"     Total Images: {analysis['total_images']}")
        print(f"     Classes/Gestures: {len(analysis['by_class'])}")

        # Per-class breakdown
        print(f"\n  📋 CLASS DISTRIBUTION:")
        class_stats = []
        for class_name, files in sorted(analysis['by_class'].items()):
            count = len(files)
            class_stats.append((class_name, count, files))
            print(f"     {class_name:<20} {count:>5} images")

        analysis_results[subdir] = {
            'total_images': analysis['total_images'],
            'classes': {name: len(files) for name, files in analysis['by_class'].items()},
            'files_by_class': {name: [str(f) for f in files] for name, files in analysis['by_class'].items()}
        }

        # Check for duplicates
        print(f"\n  🔍 QUALITY CHECKS:")
        duplicates = find_duplicates(analysis['all_files'])

        if duplicates:
            print(f"     ⚠️  Found {len(duplicates)} groups of duplicate images")
            dup_count = sum(len(v) - 1 for v in duplicates.values())
            print(f"     ⚠️  {dup_count} duplicate images (can be removed)")
        else:
            print(f"     ✅ No duplicate images found")

        # Sample image quality analysis
        print(f"     Analyzing image quality...", end='', flush=True)

        quality_issues = {
            'too_dark': [],
            'too_bright': [],
            'mostly_black': [],
            'mostly_white': [],
            'very_blurry': [],
            'corrupted': []
        }

        dimensions = defaultdict(int)

        # Sample up to 100 images for quality check (or all if less)
        sample_size = min(100, len(analysis['all_files']))
        sample_files = np.random.choice(analysis['all_files'], sample_size, replace=False)

        for img_file in sample_files:
            quality = analyze_image_quality(img_file)

            if quality is None or 'error' in quality:
                quality_issues['corrupted'].append(str(img_file))
                continue

            dimensions[quality['dimensions']] += 1

            if quality['mean_brightness'] < 30:
                quality_issues['too_dark'].append(str(img_file))
            elif quality['mean_brightness'] > 225:
                quality_issues['too_bright'].append(str(img_file))

            if quality['black_ratio'] > 0.8:
                quality_issues['mostly_black'].append(str(img_file))
            elif quality['white_ratio'] > 0.8:
                quality_issues['mostly_white'].append(str(img_file))

            if quality['blur_score'] < 100:
                quality_issues['very_blurry'].append(str(img_file))

        print(f" done")

        # Report quality issues
        total_issues = sum(len(v) for v in quality_issues.values())
        if total_issues > 0:
            print(f"\n  ⚠️  QUALITY ISSUES (from {sample_size} sampled images):")
            for issue_type, files in quality_issues.items():
                if files:
                    percentage = (len(files) / sample_size) * 100
                    print(f"     {issue_type.replace('_', ' ').title():<20} {len(files):>3} ({percentage:.1f}%)")
        else:
            print(f"     ✅ No major quality issues detected")

        # Report dimensions
        if dimensions:
            print(f"\n  📐 IMAGE DIMENSIONS:")
            for dim, count in sorted(dimensions.items(), key=lambda x: -x[1])[:5]:
                print(f"     {dim[0]}×{dim[1]:<15} {count:>3} images")

    # Training Readiness Assessment
    print(f"\n{'=' * 80}")
    print("🎯 TRAINING READINESS ASSESSMENT")
    print(f"{'=' * 80}\n")

    # Check if we have processed data
    if 'processed' in analysis_results:
        processed = analysis_results['processed']
        total = processed['total_images']
        classes = processed['classes']
        num_classes = len(classes)

        print(f"📊 PROCESSED DATA (Ready for Training):")
        print(f"   Total Images: {total}")
        print(f"   Number of Classes: {num_classes}")

        if num_classes > 0:
            min_samples = min(classes.values())
            max_samples = max(classes.values())
            avg_samples = total / num_classes

            print(f"   Min samples per class: {min_samples}")
            print(f"   Max samples per class: {max_samples}")
            print(f"   Avg samples per class: {avg_samples:.1f}")

            # Imbalance ratio
            if min_samples > 0:
                imbalance_ratio = max_samples / min_samples
                print(f"   Class imbalance ratio: {imbalance_ratio:.2f}:1")

            print(f"\n📋 RECOMMENDATIONS:\n")

            # Minimum recommendations
            RECOMMENDED_MIN_PER_CLASS = 100
            RECOMMENDED_TOTAL = 1000
            GOOD_IMBALANCE_RATIO = 3.0

            status_emoji = "✅" if total >= RECOMMENDED_TOTAL else "⚠️"
            print(f"   {status_emoji} Total Images: {total} (recommended: >{RECOMMENDED_TOTAL})")

            status_emoji = "✅" if num_classes >= 5 else "⚠️"
            print(f"   {status_emoji} Number of Classes: {num_classes} (recommended: ≥5)")

            status_emoji = "✅" if min_samples >= RECOMMENDED_MIN_PER_CLASS else "⚠️"
            print(f"   {status_emoji} Min Samples/Class: {min_samples} (recommended: ≥{RECOMMENDED_MIN_PER_CLASS})")

            if min_samples > 0:
                status_emoji = "✅" if imbalance_ratio <= GOOD_IMBALANCE_RATIO else "⚠️"
                print(
                    f"   {status_emoji} Class Balance: {imbalance_ratio:.2f}:1 (recommended: <{GOOD_IMBALANCE_RATIO}:1)")

            # Overall assessment
            print(f"\n{'─' * 80}")

            if (total >= RECOMMENDED_TOTAL and
                    num_classes >= 5 and
                    min_samples >= RECOMMENDED_MIN_PER_CLASS and
                    (min_samples == 0 or imbalance_ratio <= GOOD_IMBALANCE_RATIO)):
                print("✅ READY FOR TRAINING!")
                print("   Your dataset meets all recommended criteria.")
            elif (total >= 500 and num_classes >= 3 and min_samples >= 50):
                print("⚠️  MINIMAL VIABLE DATASET")
                print("   You can start training, but results may be limited.")
                print("   Consider collecting more data for better performance.")
            else:
                print("❌ NOT READY FOR TRAINING")
                print("   Your dataset needs more images.")

                if total < 500:
                    print(f"   - Need at least 500 total images (currently: {total})")
                if num_classes < 3:
                    print(f"   - Need at least 3 classes (currently: {num_classes})")
                if min_samples < 50:
                    print(f"   - Each class needs at least 50 images (min: {min_samples})")

    else:
        print("❌ NO PROCESSED DATA FOUND")
        print("   You need to run preprocessing on your raw images first.")

        if 'raw' in analysis_results:
            raw = analysis_results['raw']
            print(f"\n   You have {raw['total_images']} raw images in {len(raw['classes'])} classes.")
            print(f"   Run preprocessing to prepare them for training:")
            print(f"   python3 src/pipeline/run_preprocessing.py")

    # Save detailed report
    report_path = Path('dataset_analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            'analysis_date': datetime.now().isoformat(),
            'results': analysis_results
        }, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"📄 Detailed report saved to: {report_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()