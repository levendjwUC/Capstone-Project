# file: debug_single_frame.py
# Quick sanity check: capture ONE frame, run through the SAME
# MediaPipe cropper + three_gesture_model, and print raw probabilities.

import cv2
import numpy as np
from pathlib import Path
import sys
import json
import os

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.pipeline.gesture_preprocessor import GesturePreprocessor, PreprocessingConfig

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model


def main():
    model_path = PROJECT_ROOT / "three_gesture_model.keras"
    map_path = PROJECT_ROOT / "three_gesture_class_map.json"

    print(f"Loading model: {model_path}")
    model = load_model(model_path)

    print(f"Loading class map: {map_path}")
    with open(map_path, "r") as f:
        raw_map = json.load(f)

    idx_to_label = {v: k for k, v in raw_map.items()}
    print("Index → label:", idx_to_label)

    # Preprocessor identical to training
    config = PreprocessingConfig(
        output_size=(256, 256),
        debug_mode=True,
        min_detection_confidence=0.5,
    )
    pre = GesturePreprocessor(config)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("\nShow a gesture (zero / one / five).")
    print("Press SPACE to capture, 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        disp = cv2.flip(frame, 1)
        cv2.putText(
            disp,
            "SPACE = capture, q = quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Debug Capture", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            # Use the ORIGINAL frame (non-mirrored) for preprocess
            tmp = PROJECT_ROOT / "_debug_frame.jpg"
            cv2.imwrite(str(tmp), frame)

            ok = pre.process_image(tmp, tmp)
            if not ok:
                print("\n[NO HAND DETECTED / BAD CROP]")
                continue

            cropped = cv2.imread(str(tmp))
            if cropped is None:
                print("\n[FAILED TO READ CROPPED IMAGE]")
                continue

            x = cropped.astype("float32") / 255.0
            x = np.expand_dims(x, axis=0)

            probs = model.predict(x, verbose=0)[0]
            print("\n=== SINGLE FRAME PREDICTION ===")
            for idx, p in enumerate(probs):
                print(f"  {idx}: {idx_to_label[idx]} → {p:.4f}")
            print(f"Argmax: {idx_to_label[int(np.argmax(probs))]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
