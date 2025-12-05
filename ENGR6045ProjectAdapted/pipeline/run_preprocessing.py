# file: src/pipeline/run_preprocessing.py
# Run crop-only preprocessing for the 3-gesture dataset

import sys
from pathlib import Path
import shutil
import time

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.preprocessing_pipeline import PreprocessingPipeline

RAW = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "processed" / "crop_only"

print("\n=== CROP-ONLY PREPROCESSING ===")
print(f"Raw data directory:      {RAW}")
print(f"Processed output folder: {OUT}")

# -------------------------------------------------------
# Step 1 – clean old processed data
# -------------------------------------------------------
if OUT.exists():
    print("\n🗑️  Removing previous processed data...")
    shutil.rmtree(OUT)

OUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# Step 2 – build pipeline
# -------------------------------------------------------
pipeline = PreprocessingPipeline(
    raw_dir=RAW,
    output_dir=OUT,
    debug=False
)

# -------------------------------------------------------
# Step 3 – run preprocessing
# -------------------------------------------------------
print("\n🚀 Starting preprocessing...\n")
start = time.time()

pipeline.process()

elapsed = time.time() - start

print("\n=== DONE ===")
print(f"Processing time: {elapsed:.1f} seconds")
print(f"Saved to: {OUT.resolve()}")
