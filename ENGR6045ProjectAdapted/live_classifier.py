# file: live_three_gesture_classifier.py
# !/usr/bin/env python3
"""
Live classifier for the 3-gesture model (zero, one, five)

Requested modifications:
    - Show ONLY confidence (no gesture label)
    - Left panel: mirrored live feed
    - Right panel: EXACT model input (256x256 crop)
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from collections import deque
import json
import os
import time


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.pipeline.gesture_preprocessor import GesturePreprocessor, PreprocessingConfig

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.models import load_model


# =====================================================================
# Live Classifier
# =====================================================================
class LiveClassifier:
    def __init__(
        self,
        model_path="three_gesture_model_retrained.keras",
        map_path="three_gesture_class_map_retrained.json",
    ):
        print("\n===============================")
        print(" INITIALIZING LIVE CLASSIFIER")
        print("===============================\n")

        # ---- Load model ----
        print(f"📦 Loading model: {model_path}")
        self.model = load_model(model_path)

        # ---- Load class map (not used for display anymore) ----
        print(f"📄 Loading class map: {map_path}")
        with open(map_path, "r") as f:
            raw_map = json.load(f)
        self.idx_to_label = {v: k for k, v in raw_map.items()}
        print("   Class map loaded:", self.idx_to_label)

        # ---- Preprocessing identical to training ----
        config = PreprocessingConfig(
            output_size=(256, 256),
            debug_mode=False,
            min_detection_confidence=0.5,
        )
        self.preprocessor = GesturePreprocessor(config)

        print("\n✅ Live classifier ready.\n")

    # ------------------------------------------------------------------
    def preprocess_frame(self, frame):
        """
        Applies EXACT SAME preprocessing as training.
        Detects hand → crops → pads → resizes to 256x256.
        Returns (cropped_img) or None.
        """

        tmp = PROJECT_ROOT / "_live_tmp.jpg"
        cv2.imwrite(str(tmp), frame)

        ok = self.preprocessor.process_image(tmp, tmp)
        if not ok:
            return None

        cropped = cv2.imread(str(tmp))
        if cropped is None:
            return None

        return cropped

    # ------------------------------------------------------------------
    def predict_confidence(self, frame):
        """
        Returns:
            - highest confidence value
            - cropped image (or None)
        """

        cropped = self.preprocess_frame(frame)

        if cropped is None:
            return 0.0, None

        # Prepare input
        x = cropped.astype("float32") / 255.0
        x = np.expand_dims(x, axis=0)

        probs = self.model.predict(x, verbose=0)[0]

        conf = float(np.max(probs))  # only confidence
        return conf, cropped


# =====================================================================
# Main Loop
# =====================================================================
def main():
    clf = LiveClassifier()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("🎥 Live classification started. Press 'q' to quit.\n")

    RIGHT_W, RIGHT_H = 256, 256

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Left side = mirrored webcam view
        display_frame = cv2.flip(frame, 1)

        # Right side = model input crop
        confidence, cropped = clf.predict_confidence(frame)

        # -----------------------------
        # Build RIGHT PANEL (model crop)
        # -----------------------------
        if cropped is not None:
            right_panel = cv2.resize(cropped, (RIGHT_W, RIGHT_H))
        else:
            right_panel = np.zeros((RIGHT_H, RIGHT_W, 3), dtype=np.uint8)
            cv2.putText(
                right_panel, "NO HAND", (30, RIGHT_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

        # -----------------------------
        # Build LEFT PANEL (mirrored)
        # -----------------------------
        h, w = display_frame.shape[:2]
        scale = RIGHT_H / h
        left_panel = cv2.resize(display_frame, (int(w * scale), RIGHT_H))

        # Add ONLY confidence text
        conf_text = f"Conf: {confidence:.3f}"
        cv2.putText(
            left_panel, conf_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 255, 0) if confidence > 0.5 else (0, 128, 255),
            2
        )

        # -----------------------------
        # Equalize widths for stacking
        # -----------------------------
        w_left = left_panel.shape[1]
        w_right = right_panel.shape[1]
        if w_left < w_right:
            pad = np.zeros((RIGHT_H, w_right - w_left, 3), dtype=np.uint8)
            left_panel = np.hstack([left_panel, pad])
        elif w_right < w_left:
            pad = np.zeros((RIGHT_H, w_left - w_right, 3), dtype=np.uint8)
            right_panel = np.hstack([right_panel, pad])

        combined = np.hstack([left_panel, right_panel])
        cv2.imshow("Live Classifier (Left: Raw | Right: Model Input)", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =====================================================================
if __name__ == "__main__":
    main()
