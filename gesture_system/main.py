# gesture_system.py
"""
Gesture-based Crazyflie control with debug viewer.

Pipeline:
    Camera -> GestureClassifier -> Smoothed gesture -> Command
    (Optionally) Command -> Crazyflie hover_setpoint

Features:
    - DRY_RUN mode (no drone, just a debug window)
    - Majority vote smoothing over recent predictions
    - On-screen overlay: raw gesture, smoothed gesture, mapped command, confidence

Requirements:
    - camera_module.camera.Camera
    - gesture_classifier.classifier.GestureClassifier
    - best_model.keras (or adjust MODEL_PATH)
"""

import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from camera_module.camera import Camera
from gesture_classifier.classifier import GestureClassifier

# ---------------------- CONFIG ----------------------

# If True, do NOT connect to drone, just show window + print commands.
DRY_RUN = True

# Model path
MODEL_PATH = "best_model.keras"

# Crazyflie connection URI (only used when DRY_RUN = False)
URI = "radio://0/80/2M/E7E7E7E7E7"

# Gesture smoothing
SMOOTHING_WINDOW = 7          # how many recent labels to consider
CONFIDENCE_THRESHOLD = 0.75   # minimum confidence to use a prediction

# Takeoff/land behavior
TAKEOFF_ALT = 0.5             # meters
TAKEOFF_RATE = 0.4            # m/s
LAND_RATE = 0.25              # m/s

# Motion speeds
FORWARD_SPEED = 0.3           # m/s
BACKWARD_SPEED = 0.3          # m/s
YAW_SPEED = 30.0              # deg/s

CONTROL_HZ = 30.0             # main loop frequency
COMMAND_DEBOUNCE = 0.3        # seconds between updating motion command
TOGGLE_COOLDOWN = 1.0         # seconds between takeoff/land toggles

# -------------------------------------------------------
#  Gesture → high-level command mapping
# -------------------------------------------------------
#
# Your classifier uses: ['five','four','one','three','two','zero']
# You MUST adjust this to match what each gesture should do.
#
# Commands:
#   - "HOVER"          : no translational motion, keep altitude
#   - "FORWARD"        : move forward
#   - "BACK"           : move backward
#   - "YAW_LEFT"       : rotate left
#   - "YAW_RIGHT"      : rotate right
#   - "TAKEOFF_LAND"   : toggle takeoff/landing
#
GESTURE_TO_COMMAND = {
    "zero": "HOVER",        # neutral / no motion
    "one": "FORWARD",
    "two": "BACK",
    "three": "YAW_LEFT",
    "four": "YAW_RIGHT",
    "five": "TAKEOFF_LAND",
}


# ---------------------- UTILS ----------------------


def majority_vote(labels: deque) -> Optional[str]:
    """
    Majority vote over a deque of labels (strings).

    Returns:
        Most common label or None if deque is empty.
    """
    if not labels:
        return None
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]


def draw_overlay(
    frame: np.ndarray,
    raw_label: Optional[str],
    raw_conf: float,
    smooth_label: Optional[str],
    command: Optional[str],
) -> np.ndarray:
    """
    Draw text overlays on frame for debug.

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    color_raw = (0, 255, 255)
    color_smooth = (0, 255, 0)
    color_cmd = (255, 255, 0)

    y = 30
    dy = 25

    cv2.putText(
        annotated,
        f"Raw: {raw_label or 'None'}  conf={raw_conf:.2f}",
        (10, y),
        font,
        scale,
        color_raw,
        thickness,
        cv2.LINE_AA,
    )
    y += dy

    cv2.putText(
        annotated,
        f"Smoothed: {smooth_label or 'None'}",
        (10, y),
        font,
        scale,
        color_smooth,
        thickness,
        cv2.LINE_AA,
    )
    y += dy

    cv2.putText(
        annotated,
        f"Command: {command or 'None'}",
        (10, y),
        font,
        scale,
        color_cmd,
        thickness,
        cv2.LINE_AA,
    )
    y += dy

    mode_str = "DRY RUN (no drone)" if DRY_RUN else "LIVE: controlling drone"
    cv2.putText(
        annotated,
        mode_str,
        (10, annotated.shape[0] - 20),
        font,
        0.6,
        (0, 0, 255) if DRY_RUN else (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return annotated


# ---------------------- CORE LOOP ----------------------


def run_gesture_control():
    """
    Main gesture control loop.

    - Always shows a debug window.
    - If DRY_RUN = True: no drone connection, just prints and visualizes.
    - If DRY_RUN = False: connects to Crazyflie and sends hover_setpoint commands.
    """
    # --- Init camera & classifier ---
    camera = Camera()
    classifier = GestureClassifier(model_path=MODEL_PATH)

    recent_labels = deque(maxlen=SMOOTHING_WINDOW)
    smooth_label: Optional[str] = None
    last_smooth_label: Optional[str] = None
    last_command_update = 0.0
    last_toggle_time = 0.0
    current_command: Optional[str] = "HOVER"

    # Drone state (only used if not DRY_RUN)
    flight_mode = "IDLE"   # IDLE, TAKEOFF, FLYING, LANDING
    z_ref = 0.0

    # Optional: import and set up Crazyflie
    if not DRY_RUN:
        import logging
        import warnings
        import cflib.crtp
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
        from cflib.utils import uri_helper

        logging.basicConfig(level=logging.ERROR)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        cflib.crtp.init_drivers()
        uri = uri_helper.uri_from_env(default=URI)

        print(f"[DRONE] Connecting to {uri}...")
        scf_context = SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache"))
    else:
        scf_context = None

    # We wrap the main loop in a context manager if drone is used
    if DRY_RUN:
        # No drone; simple loop
        try:
            _gesture_loop_body(
                camera=camera,
                classifier=classifier,
                recent_labels=recent_labels,
                smooth_label_holder=[smooth_label],
                last_smooth_label_holder=[last_smooth_label],
                last_command_update_holder=[last_command_update],
                last_toggle_time_holder=[last_toggle_time],
                current_command_holder=[current_command],
                flight_mode_holder=[flight_mode],
                z_ref_holder=[z_ref],
                cf=None,
            )
        finally:
            camera.stop()
            cv2.destroyAllWindows()
    else:
        # With drone
        from cflib.crazyflie.log import LogConfig

        with scf_context as scf:
            cf = scf.cf
            # Arm platform
            cf.platform.send_arming_request(True)
            print("[DRONE] Armed. Gesture control active.")
            # Optional: set mode to high-level / hover if needed, but here we use hover setpoints directly.

            try:
                _gesture_loop_body(
                    camera=camera,
                    classifier=classifier,
                    recent_labels=recent_labels,
                    smooth_label_holder=[smooth_label],
                    last_smooth_label_holder=[last_smooth_label],
                    last_command_update_holder=[last_command_update],
                    last_toggle_time_holder=[last_toggle_time],
                    current_command_holder=[current_command],
                    flight_mode_holder=[flight_mode],
                    z_ref_holder=[z_ref],
                    cf=cf,
                )
            finally:
                print("[DRONE] Stopping, sending stop setpoint and disarming.")
                try:
                    cf.commander.send_stop_setpoint()
                except Exception:
                    pass
                cf.platform.send_arming_request(False)
                camera.stop()
                cv2.destroyAllWindows()


def _gesture_loop_body(
    camera: Camera,
    classifier: GestureClassifier,
    recent_labels: deque,
    smooth_label_holder: list,
    last_smooth_label_holder: list,
    last_command_update_holder: list,
    last_toggle_time_holder: list,
    current_command_holder: list,
    flight_mode_holder: list,
    z_ref_holder: list,
    cf,
):
    """
    Internal loop shared by DRY_RUN and live modes.

    We use 1-element lists to allow modification of "holder" values in place.
    """
    smooth_label = smooth_label_holder[0]
    last_smooth_label = last_smooth_label_holder[0]
    last_command_update = last_command_update_holder[0]
    last_toggle_time = last_toggle_time_holder[0]
    current_command = current_command_holder[0]
    flight_mode = flight_mode_holder[0]
    z_ref = z_ref_holder[0]

    running = True
    dt_target = 1.0 / CONTROL_HZ
    last_loop_time = time.time()

    print("[SYSTEM] Press 'q' in the window to quit.")

    while running:
        loop_start = time.time()

        # --- Get frame ---
        frame = camera.get_frame()
        if frame is None:
            # If camera hiccups, just wait a bit
            time.sleep(0.01)
            continue

        # --- Classify gesture ---
        raw_label, raw_conf, _ = classifier.classify(frame)

        if raw_label is None or raw_conf < CONFIDENCE_THRESHOLD:
            # Treat low-confidence as a neutral "HOVER"
            raw_label = None

        # Update smoothing buffer
        if raw_label is not None:
            recent_labels.append(raw_label)

        # Compute smoothed label
        smooth_label = majority_vote(recent_labels) if recent_labels else None

        # Map to command
        now = time.time()
        if (now - last_command_update) > COMMAND_DEBOUNCE:
            cmd = None
            if smooth_label is not None:
                cmd = GESTURE_TO_COMMAND.get(smooth_label, None)

            # Default to hover if unknown
            if cmd is None:
                cmd = "HOVER"

            # Handle TAKEOFF_LAND as edge-triggered toggle
            if cmd == "TAKEOFF_LAND":
                if (
                    smooth_label != last_smooth_label
                    and (now - last_toggle_time) > TOGGLE_COOLDOWN
                ):
                    # Toggle flight mode
                    if flight_mode in ("IDLE", "LANDING"):
                        print("[CMD] Takeoff requested.")
                        flight_mode = "TAKEOFF"
                    else:
                        print("[CMD] Landing requested.")
                        flight_mode = "LANDING"

                    last_toggle_time = now
                # Command remains whatever it was before for motion
                cmd = current_command if current_command != "TAKEOFF_LAND" else "HOVER"

            current_command = cmd
            last_command_update = now
            last_smooth_label = smooth_label

        # --- Drone motion (if not DRY_RUN) ---
        vx = 0.0
        yawrate = 0.0

        # Flight mode & altitude
        now = time.time()
        dt = now - last_loop_time
        last_loop_time = now

        if cf is None:
            # DRY_RUN: no altitude logic needed, just keep z_ref at 0
            flight_mode = "IDLE"
            z_ref = 0.0
        else:
            # Simple finite-state altitude control
            if flight_mode == "TAKEOFF":
                z_ref += TAKEOFF_RATE * dt
                if z_ref >= TAKEOFF_ALT:
                    z_ref = TAKEOFF_ALT
                    flight_mode = "FLYING"
                    print(f"[STATE] Reached takeoff altitude {z_ref:.2f} m, now FLYING.")
            elif flight_mode == "LANDING":
                z_ref -= LAND_RATE * dt
                if z_ref <= 0.05:
                    z_ref = 0.0
                    flight_mode = "IDLE"
                    print("[STATE] Landed.")
            elif flight_mode == "FLYING":
                # In this simple version, we keep constant altitude during flight
                pass
            elif flight_mode == "IDLE":
                z_ref = 0.0

            # Decide motion only when flying
            if flight_mode == "FLYING":
                if current_command == "FORWARD":
                    vx = FORWARD_SPEED
                elif current_command == "BACK":
                    vx = -BACKWARD_SPEED
                elif current_command == "YAW_LEFT":
                    yawrate = YAW_SPEED
                elif current_command == "YAW_RIGHT":
                    yawrate = -YAW_SPEED
                elif current_command == "HOVER":
                    vx = 0.0
                    yawrate = 0.0

            # Send command
            try:
                cf.commander.send_hover_setpoint(vx, 0.0, yawrate, z_ref)
            except Exception as e:
                print(f"[DRONE] Error sending setpoint: {e}")
                running = False

        # --- Overlay + show window ---
        display_cmd = current_command
        disp_frame = draw_overlay(
            frame=frame,
            raw_label=raw_label,
            raw_conf=raw_conf if raw_label is not None else 0.0,
            smooth_label=smooth_label,
            command=display_cmd,
        )

        cv2.imshow("Gesture Control", disp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[SYSTEM] 'q' pressed, exiting.")
            running = False

        # --- Loop timing ---
        elapsed = time.time() - loop_start
        sleep_time = dt_target - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Save back holder values (for completeness)
    smooth_label_holder[0] = smooth_label
    last_smooth_label_holder[0] = last_smooth_label
    last_command_update_holder[0] = last_command_update
    last_toggle_time_holder[0] = last_toggle_time
    current_command_holder[0] = current_command
    flight_mode_holder[0] = flight_mode
    z_ref_holder[0] = z_ref


# ---------------------- ENTRYPOINT ----------------------


if __name__ == "__main__":
    run_gesture_control()

