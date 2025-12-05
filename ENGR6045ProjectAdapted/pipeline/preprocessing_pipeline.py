# file: src/pipeline/preprocessing_pipeline.py
# Minimal pipeline for crop-only preprocessing

from pathlib import Path
import shutil

from .gesture_preprocessor import GesturePreprocessor, PreprocessingConfig


class PreprocessingPipeline:
    def __init__(self, raw_dir: Path, output_dir: Path, debug=False):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config = PreprocessingConfig(
            output_size=(256, 256),
            debug_mode=debug
        )
        self.preprocessor = GesturePreprocessor(config)

    def process(self):
        for split in ["train", "val", "test"]:
            split_in = self.raw_dir / split
            if not split_in.exists():
                continue

            split_out = self.output_dir / split
            split_out.mkdir(parents=True, exist_ok=True)

            for gesture_dir in split_in.iterdir():
                if not gesture_dir.is_dir():
                    continue

                gesture_out = split_out / gesture_dir.name
                gesture_out.mkdir(exist_ok=True)

                for img_path in gesture_dir.glob("*.*"):
                    out_path = gesture_out / img_path.name

                    ok = self.preprocessor.process_image(img_path, out_path)
                    if not ok:
                        # delete broken output if it exists
                        if out_path.exists():
                            out_path.unlink()
