# file: collect_retrain_crops.py
import cv2
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.pipeline.gesture_preprocessor import GesturePreprocessor, PreprocessingConfig


GESTURES = ["zero", "one", "five"]
OUT_DIR = PROJECT_ROOT / "data" / "retrain"

def main():
    # Preprocessor identical to training
    pre = GesturePreprocessor(
        PreprocessingConfig(
            output_size=(256, 256),
            min_detection_confidence=0.5,
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for g in GESTURES:
        (OUT_DIR / g).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam error")
        return

    print("\n=== RETRAIN DATA COLLECTION ===")
    print("Press SPACE to capture one crop.")
    print("Press ESC to finish a gesture.")
    print("Press Q to quit.\n")

    for gesture in GESTURES:
        print(f"\n--- Collecting for {gesture.upper()} ---")
        print("Hold gesture, press SPACE repeatedly to save aligned crops.")

        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            disp = cv2.flip(frame, 1)
            cv2.putText(disp, f"Gesture: {gesture}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Retrain Capture", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == 27:  # ESC
                break
            if key == 32:  # SPACE
                tmp = PROJECT_ROOT / "_tmp_retrain.jpg"
                cv2.imwrite(str(tmp), frame)
                ok = pre.process_image(tmp, tmp)
                if not ok:
                    print("[NO HAND]")
                    continue
                out_path = OUT_DIR / gesture / f"{gesture}_{count}.jpg"
                Path(tmp).rename(out_path)
                print(f"Saved {out_path.name}")
                count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone collecting!")

main()