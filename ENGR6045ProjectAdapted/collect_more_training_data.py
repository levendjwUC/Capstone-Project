# file: collect_three_gesture_data.py
# !/usr/bin/env python3

"""
Minimal 3-gesture data collector for:
    zero, one, five

Collects clean, consistent hand images quickly.
Splits automatically into train/val/test (70/15/15).
"""

import cv2
import time
from pathlib import Path
from datetime import datetime
import random
import sys

GESTURES = ["zero", "one", "five"]

# ----------------------------------------------------------------------
# Directory Setup
# ----------------------------------------------------------------------

BASE_DIR = Path("data/raw")
for split in ["train", "val", "test"]:
    for gesture in GESTURES:
        (BASE_DIR / split / gesture).mkdir(parents=True, exist_ok=True)


def determine_split(counts, gesture):
    """Keep split ratio approx. 70/15/15."""
    total = sum(counts[gesture].values())

    if total == 0:
        r = random.random()
        if r < 0.7:
            return "train"
        elif r < 0.85:
            return "val"
        else:
            return "test"

    ideal_train = int(total * 0.70)
    ideal_val   = int(total * 0.15)
    ideal_test  = int(total * 0.15)

    if counts[gesture]["train"] < ideal_train:
        return "train"
    if counts[gesture]["val"] < ideal_val:
        return "val"
    if counts[gesture]["test"] < ideal_test:
        return "test"

    return "train"


def count_existing():
    """Count existing dataset for balance."""
    counts = {g: {"train": 0, "val": 0, "test": 0} for g in GESTURES}

    for split in ["train", "val", "test"]:
        for gesture in GESTURES:
            path = BASE_DIR / split / gesture
            counts[gesture][split] = len([f for f in path.glob("*.jpg")])

    return counts


# ----------------------------------------------------------------------
# Collecting Logic
# ----------------------------------------------------------------------

def collect_for_gesture(gesture, n_samples=300):
    """Collect n_samples images for a single gesture."""
    print(f"\n────────────────────────────────────────────")
    print(f"📸 Collecting {n_samples} samples for: {gesture.upper()}")
    print("Position your hand. Press SPACE to begin.")
    print("Press Q to quit this gesture.")
    print("────────────────────────────────────────────\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    counts = count_existing()
    collected = 0
    is_collecting = False
    last_capture_time = 0

    COLLECTION_DELAY = 0.05  # 20 fps

    while collected < n_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror for consistency

        # Draw simple “center square”
        h, w = frame.shape[:2]
        size = 350
        x1 = w//2 - size//2
        y1 = h//2 - size//2
        x2 = w//2 + size//2
        y2 = h//2 + size//2
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)

        cv2.putText(
            frame, f"GESTURE: {gesture.upper()}",
            (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3
        )
        cv2.putText(
            frame, f"{collected}/{n_samples}",
            (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2
        )

        if not is_collecting:
            cv2.putText(
                frame, "Press SPACE to start",
                (30,160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2
            )
        else:
            cv2.putText(
                frame, f"Recording… {1/COLLECTION_DELAY:.0f} img/s",
                (30,160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2
            )

        cv2.imshow("Collecting", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\n❌ Stopped early.")
            break

        if key == ord(" ") and not is_collecting:
            print("▶️ Starting capture…")
            is_collecting = True
            last_capture_time = time.time()

        if is_collecting:
            t = time.time()
            if t - last_capture_time >= COLLECTION_DELAY:
                # choose split
                split = determine_split(counts, gesture)

                # filename
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                out_path = BASE_DIR / split / gesture / f"{gesture}_{stamp}.jpg"

                cv2.imwrite(str(out_path), frame)

                counts[gesture][split] += 1
                collected += 1
                last_capture_time = t

                print(f"\rSaved {collected}/{n_samples} → {split}", end="")
                sys.stdout.flush()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✔ Done collecting for {gesture}!")
    return collected


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    print("\n====================================")
    print(" THREE-GESTURE DATA COLLECTION TOOL ")
    print("====================================\n")
    print("Gestures to collect:")
    print(" • zero")
    print(" • one")
    print(" • five")
    print("------------------------------------\n")

    for gesture in GESTURES:
        collect_for_gesture(gesture, n_samples=300)

    print("\n🎉 All gestures collected!")


if __name__ == "__main__":
    main()
