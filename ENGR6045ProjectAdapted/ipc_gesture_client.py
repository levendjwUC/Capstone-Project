# file: ipc_gesture_client.py
# 3-Gesture Client for the retrained classifier
#
# Model class indices (from three_gesture_class_map_retrained.json):
#   0 -> "five"  -> ascend
#   1 -> "one"   -> left
#   2 -> "zero"  -> forward
#
# Final mapping:
#   2 = forward
#   1 = left
#   0 = ascend

import cv2
import numpy as np
import socket
import sys
import time
from pathlib import Path
from collections import deque
from statistics import mode

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.gesture_preprocessor import GesturePreprocessor, PreprocessingConfig
from tensorflow.keras.models import load_model


MODEL_PATH = "three_gesture_model.keras"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5007

# -----------------------------
# CORRECT mapping based on JSON
# -----------------------------
GESTURE_COMMANDS = {
    2: "forward",   # zero → forward
    1: "left",      # one → left turn
    0: "ascend",    # five → ascend (open palm)
}

SMOOTHING_WINDOW = 24
CONF_THRESHOLD_MAIN = 0.60
CONF_THRESHOLD_FREEZE = 0.40
COMMAND_INTERVAL = 1.0


class GestureClient:
    def __init__(self):
        print("🚀 Initializing 3-Gesture Client (Correct Label Order)")

        self.model = load_model(MODEL_PATH)
        print("✅ Loaded three_gesture_model.keras")

        cfg = PreprocessingConfig(debug_mode=False, save_intermediate=False)
        self.pre = GesturePreprocessor(cfg)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((SERVER_HOST, SERVER_PORT))
        print(f"🔗 Connected to server {SERVER_HOST}:{SERVER_PORT}")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.pred_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.prev_command = "hover"
        self.last_send_time = 0

    def preprocess_frame(self, frame):
        tmp_in = Path("tmp_frame.jpg")
        tmp_out = Path("tmp_processed.png")
        cv2.imwrite(str(tmp_in), frame)

        success, _ = self.pre.process_image(tmp_in, tmp_out, save_intermediate=False)

        processed = None
        if success and tmp_out.exists():
            processed = cv2.imread(str(tmp_out), cv2.IMREAD_COLOR)

        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)
        return processed

    def smooth_gesture(self, g):
        self.pred_buffer.append(g)
        return mode(self.pred_buffer)

    def filter_command(self, g, conf):
        if conf < CONF_THRESHOLD_FREEZE:
            return "hover"

        if conf < CONF_THRESHOLD_MAIN:
            return self.prev_command

        return GESTURE_COMMANDS.get(g, "hover")

    def run(self):
        print("\n🎬 3-Gesture control running (ESC to quit)\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            cv2.imshow("Webcam - Original", frame)

            processed = self.preprocess_frame(frame)
            if processed is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.imshow("Webcam - Processed", blank)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            img = cv2.resize(processed, (64, 64)).astype("float32") / 255.0
            x = np.expand_dims(img, axis=0)

            preds = self.model.predict(x, verbose=0)[0]
            raw_id = int(np.argmax(preds))
            conf = float(np.max(preds))

            smooth_id = self.smooth_gesture(raw_id)
            command = self.filter_command(smooth_id, conf)

            now = time.time()
            if now - self.last_send_time >= COMMAND_INTERVAL:
                try:
                    self.sock.sendall(f"{command}\n".encode("utf-8"))
                except BrokenPipeError:
                    print("❌ Lost connection to command server")
                    break
                self.prev_command = command
                self.last_send_time = now

            cv2.imshow("Webcam - Processed", img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        self.sock.close()
        cv2.destroyAllWindows()
        print("👋 Client closed.")


if __name__ == "__main__":
    GestureClient().run()
