# HandCropper - Hand Gesture Preprocessing Pipeline

## Project Structure
HandCropper/
├── src/
│   ├── components/       # 7 preprocessing components
│   ├── pipeline/         # Main pipeline and wrapper
│   └── utils/           # Utility functions
├── tests/               # All test files
├── scripts/
│   ├── cleanup/         # Cleanup utilities
│   ├── analysis/        # Analysis tools
│   └── data_processing/ # Dataset processing
├── data/
│   ├── raw/            # Original datasets
│   ├── processed/      # Processed outputs
│   └── test_samples/   # Test samples
├── configs/            # Configuration files
└── docs/              # Documentation

## Components

1. **HandBinaryMaskProcessor** - Creates binary masks
2. **HandCenterAligner** - Centers hand in frame
3. **HandCropper** - Crops to hand region
4. **HandDistanceTransform** - Distance transform processing
5. **HandEdgeEnhancer** - Enhances edges
6. **HandOrientationNormalizer** - Normalizes orientation
7. **HandSkeletonExtractor** - Extracts skeleton

## Usage

```python
from src.pipeline import GesturePreprocessor
preprocessor = GesturePreprocessor()
Backup Information
A backup was created before reorganization at:
../HandCropper_backup_20251014_134528
