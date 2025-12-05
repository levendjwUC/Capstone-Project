# file: src/pipeline/gesture_preprocessor.py
# Crop-only preprocessing using MediaPipe hand detection
# Output size: 256x256 RGB

import cv2
import mediapipe as mp
from pathlib import Path
import numpy as np


class PreprocessingConfig:
    def __init__(
            self,
            output_size=(256, 256),
            debug_mode=False,
            min_detection_confidence=0.5
    ):
        self.output_size = output_size
        self.debug_mode = debug_mode
        self.min_detection_confidence = min_detection_confidence


class GesturePreprocessor:
    """Crop-only hand detector & normalizer."""
    def __init__(self, config: PreprocessingConfig):
        self.config = config

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=config.min_detection_confidence
        )

    def _get_bbox(self, hand_landmarks, img_shape):
        h, w, _ = img_shape
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]
        return (
            int(min(xs)), int(min(ys)),
            int(max(xs)), int(max(ys))
        )

    def process_image(self, input_path: Path, output_path: Path) -> bool:
        img = cv2.imread(str(input_path))
        if img is None:
            return False

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if not res.multi_hand_landmarks:
            if self.config.debug_mode:
                print(f"[NO HAND] {input_path.name}")
            return False

        # Only largest hand (even though max_num_hands=1)
        lm = res.multi_hand_landmarks[0]
        x1, y1, x2, y2 = self._get_bbox(lm, img.shape)

        # Pad 15%
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img.shape[1], x2 + pad_x)
        y2 = min(img.shape[0], y2 + pad_y)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        crop = cv2.resize(crop, self.config.output_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output_path), crop)

        return True
