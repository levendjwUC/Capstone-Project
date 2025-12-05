# file: live_classifier.py
# !/usr/bin/env python3
"""
Live hand gesture preprocessor viewer.
Shows raw webcam frame and preprocessed frame side by side.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import preprocessing components
from src.pipeline.gesture_preprocessor import GesturePreprocessor, PreprocessingConfig

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LivePreprocessorViewer:
    """Live viewer to compare original vs preprocessed frames."""

    def __init__(self):
        print("🚀 Initializing Live Preprocessor Viewer...")

        # Initialize preprocessor (same as training)
        print("🔧 Initializing preprocessor...")
        config = PreprocessingConfig(
            debug_mode=False,
            save_intermediate=False
        )
        self.preprocessor = GesturePreprocessor(config=config)
        print("✅ Preprocessor initialized")

        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def preprocess_frame(self, frame):
        """
        Run the frame through the existing preprocessing pipeline and
        return the processed image (uint8 BGR) for visualization.
        """
        temp_path = Path('temp_frame.jpg')
        output_path = Path('temp_processed.png')

        try:
            # Save current frame
            cv2.imwrite(str(temp_path), frame)

            # Process through the pipeline (same call you already use)
            success, _ = self.preprocessor.process_image(
                temp_path,
                output_path,
                save_intermediate=False
            )

            processed = None
            if success and output_path.exists():
                processed = cv2.imread(str(output_path))

        except Exception as e:
            print(f"Preprocessing error: {e}")
            processed = None
        finally:
            # Clean up temp files regardless of success/failure
            temp_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        return processed

    def compose_display(self, original, processed):
        """
        Create a side-by-side image of original and processed frames.
        Left: original, Right: processed (or placeholder if None).
        """
        # Ensure we don't modify original in-place
        original_disp = original.copy()

        h, w = original_disp.shape[:2]

        if processed is not None:
            # Resize processed to match original height
            ph, pw = processed.shape[:2]
            scale = h / float(ph)
            new_w = max(1, int(pw * scale))
            processed_resized = cv2.resize(processed, (new_w, h))
        else:
            # If preprocessing failed, show a black image with text
            processed_resized = np.zeros_like(original_disp)
            cv2.putText(processed_resized,
                        "Preprocessing failed / no hand",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

        # Label both halves
        cv2.putText(original_disp, "Original",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(processed_resized, "Processed",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Pad narrower image so they have equal widths for hstack
        w1 = original_disp.shape[1]
        w2 = processed_resized.shape[1]

        if w1 > w2:
            pad = np.zeros((h, w1 - w2, 3), dtype=np.uint8)
            processed_resized = np.hstack([processed_resized, pad])
        elif w2 > w1:
            pad = np.zeros((h, w2 - w1, 3), dtype=np.uint8)
            original_disp = np.hstack([original_disp, pad])

        combined = np.hstack([original_disp, processed_resized])
        return combined

    def update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed

    def run(self):
        """Run the live preprocessor viewer."""
        print("\n" + "=" * 60)
        print("STARTING LIVE PREPROCESSOR VIEWER")
        print("=" * 60)
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("\nShowing camera feed...\n")

        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        screenshot_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            # Mirror the frame for better UX
            frame = cv2.flip(frame, 1)

            # Preprocess the frame
            processed = self.preprocess_frame(frame)

            # Compose side-by-side display
            display_frame = self.compose_display(frame, processed)

            # Update FPS and draw it
            self.update_fps()
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}",
                        (10, display_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show result
            cv2.imshow('Hand Gesture Preprocessing - Live', display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n\n👋 Quitting...")
                break
            elif key == ord('s'):
                # Save screenshot of combined view
                screenshot_name = f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(screenshot_name, display_frame)
                print(f"\n📸 Screenshot saved: {screenshot_name}")
                screenshot_count += 1

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("✅ Live viewer ended")
        print(f"📊 Average FPS: {self.fps:.1f}")
        print("=" * 60)


def main():
    """Main function."""
    print("\n" + "🤖 " * 20)
    print("HAND GESTURE PREPROCESSING VIEWER")
    print("🤖 " * 20 + "\n")

    viewer = LivePreprocessorViewer()

    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
