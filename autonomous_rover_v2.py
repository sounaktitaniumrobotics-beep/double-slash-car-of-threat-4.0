"""
Autonomous CCTV Rover — Path Following via Grayscale + Threshold + Contours
=============================================================================
HOW IT WORKS:
  1. Camera frame → Grayscale → Gaussian blur → Binary threshold (B&W)
  2. The ROAD/FLOOR is the dominant bright (or dark) region in the bottom half
  3. Find the largest contour in the region of interest (ROI) → that is the path
  4. Compute the centroid X of that path contour
  5. Compare centroid X to frame center → steer LEFT / RIGHT / STRAIGHT
  6. If no path contour found (obstacle or edge) → STOP

Keyboard:
    SPACE   — Pause / Resume autonomous driving
    Q / ESC — Emergency stop and quit
    T       — Toggle debug view (B&W threshold overlay)
    +/-     — Raise / Lower brightness threshold

Requirements:
    pip install paho-mqtt opencv-python keyboard
"""

import cv2
import time
import threading
import keyboard
import paho.mqtt.client as mqtt
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  MQTT Configuration
# ══════════════════════════════════════════════════════════════════════════════
BROKER   = "1887637a30c2432683efd36d80813f78.s1.eu.hivemq.cloud"
PORT     = 8883
USERNAME = "titaniumrobotics"
PASSWORD = "Sapt090059#"
TOPIC    = "rccar/control"

# ══════════════════════════════════════════════════════════════════════════════
#  Speed Config  (150 RPM motors — keep it slow)
# ══════════════════════════════════════════════════════════════════════════════
THROTTLE_FORWARD  = 15    # % forward speed — safe for 150 RPM
THROTTLE_TURN     = 0     # throttle while correcting steering (0 = turn in place)
STEERING_CORRECT  = 25    # % steering when path drifts off-center
STEERING_STRAIGHT = 0     # % steering when centred

# ══════════════════════════════════════════════════════════════════════════════
#  Path Detection Config
# ══════════════════════════════════════════════════════════════════════════════
# ROI = Region Of Interest: bottom portion of frame where the path/road is visible
ROI_TOP_FRAC      = 0.55  # ROI starts at 55% from top (ignore ceiling/far distance)
ROI_BOT_FRAC      = 0.95  # ROI ends at 95% from top  (ignore bumper edge)

# Binary threshold value (0-255). Pixels ABOVE this → white (road), below → black
# For a LIGHT floor: ~120-160. For a DARK road on light bg: invert below.
THRESHOLD_VALUE   = 127
THRESHOLD_STEP    = 10

# If the path centroid is within this fraction of center, go straight
CENTRE_DEADBAND   = 0.15  # 15% of half-width — fine-tune to reduce wobble

# Minimum contour area (pixels²) to be considered a valid path
MIN_CONTOUR_AREA  = 3000

# If the largest path region covers less than this % of ROI → treat as BLOCKED
MIN_PATH_FILL_PCT = 5.0   # percent

# How long path must be absent before declaring obstacle (avoids flicker stops)
BLOCK_HOLD_SECS   = 0.4

# ══════════════════════════════════════════════════════════════════════════════
#  Camera Config
# ══════════════════════════════════════════════════════════════════════════════
MAX_CAMERAS  = 6
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ══════════════════════════════════════════════════════════════════════════════
#  Shared State
# ══════════════════════════════════════════════════════════════════════════════
mqtt_client       = None
auto_mode         = True
show_debug        = True   # T key toggles B&W debug view
threshold_val     = THRESHOLD_VALUE
state_lock        = threading.Lock()

# ══════════════════════════════════════════════════════════════════════════════
#  MQTT
# ══════════════════════════════════════════════════════════════════════════════

def on_connect(client, userdata, flags, rc, props):
    print(f"[MQTT] {'Connected OK' if rc == 0 else f'Failed rc={rc}'}")

def send_command(throttle: int, steering: int):
    if mqtt_client:
        mqtt_client.publish(TOPIC, f"{throttle},{steering}", qos=1)

def connect_mqtt():
    global mqtt_client
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    c.username_pw_set(USERNAME, PASSWORD)
    c.tls_set()
    c.on_connect = on_connect
    mqtt_client = c
    print("[MQTT] Connecting…")
    c.connect(BROKER, PORT, 60)
    c.loop_start()
    time.sleep(2)

# ══════════════════════════════════════════════════════════════════════════════
#  Camera
# ══════════════════════════════════════════════════════════════════════════════

def find_best_camera():
    available = []
    for i in range(MAX_CAMERAS):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"  [Camera {i}] {cap.getBackendName()}")
                available.append(i)
        cap.release()
    if not available:
        raise RuntimeError("No cameras found. Start OBS Virtual Camera first.")
    preferred = [i for i in available if i > 0]
    idx = preferred[0] if preferred else available[0]
    print(f"[CAM]  Using index {idx}")
    return idx

def open_camera(idx):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap

# ══════════════════════════════════════════════════════════════════════════════
#  Path Detection Core
# ══════════════════════════════════════════════════════════════════════════════

def detect_path(frame, thresh_val):
    """
    Returns:
        binary_full  : full-frame B&W image for debug display
        centroid_x   : X position of path centroid in FRAME coords (or None)
        path_fill    : % of ROI covered by path (0-100)
        contour_pts  : contour points for drawing (or None)
    """
    h, w = frame.shape[:2]

    # ── Step 1: Greyscale ────────────────────────────────────────────────────
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Step 2: Gaussian blur (kills noise / texture) ────────────────────────
    blurred = cv2.GaussianBlur(grey, (11, 11), 0)

    # ── Step 3: Binary threshold → B&W ──────────────────────────────────────
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    # Full-frame binary for debug view
    binary_full = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ── Step 4: Crop to ROI (bottom band = floor/path region) ───────────────
    roi_y1 = int(ROI_TOP_FRAC * h)
    roi_y2 = int(ROI_BOT_FRAC * h)
    roi    = binary[roi_y1:roi_y2, :]

    roi_area = roi.shape[0] * roi.shape[1]

    # ── Step 5: Find contours of white regions ───────────────────────────────
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return binary_full, None, 0.0, None

    # Largest contour = the dominant floor/road region
    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    if area < MIN_CONTOUR_AREA:
        return binary_full, None, 0.0, None

    path_fill = (area / roi_area) * 100.0

    # ── Step 6: Centroid of largest contour ──────────────────────────────────
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return binary_full, None, path_fill, None

    cx_roi = int(M["m10"] / M["m00"])   # X in ROI coords
    cy_roi = int(M["m01"] / M["m00"])   # Y in ROI coords

    # Convert back to full-frame coords for drawing
    cx_frame = cx_roi
    cy_frame = cy_roi + roi_y1

    # Shift contour points to full-frame coords for drawing
    contour_pts = largest + np.array([[[0, roi_y1]]])

    return binary_full, cx_frame, path_fill, contour_pts


def steer_decision(cx_frame, frame_w):
    """
    Returns (throttle, steering, direction_label)
    based on where the path centroid is relative to frame centre.
    """
    centre     = frame_w // 2
    half_width = frame_w // 2
    deviation  = (cx_frame - centre) / half_width   # -1.0 (left) to +1.0 (right)

    if abs(deviation) <= CENTRE_DEADBAND:
        return THROTTLE_FORWARD, STEERING_STRAIGHT, "▲ STRAIGHT"
    elif deviation > 0:
        # Path centroid is RIGHT of centre → steer right
        return THROTTLE_FORWARD, STEERING_CORRECT, "↗ STEER RIGHT"
    else:
        # Path centroid is LEFT of centre → steer left
        return THROTTLE_FORWARD, -STEERING_CORRECT, "↖ STEER LEFT"

# ══════════════════════════════════════════════════════════════════════════════
#  Overlay Drawing
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(frame, cx_frame, path_fill, contour_pts,
                 throttle, steering, direction, blocked, auto, fps, thresh_val, show_dbg):
    h, w = frame.shape[:2]

    roi_y1 = int(ROI_TOP_FRAC * h)
    roi_y2 = int(ROI_BOT_FRAC * h)

    # ── ROI boundary ─────────────────────────────────────────────────────────
    roi_color = (0, 80, 255) if blocked else (0, 255, 120)
    cv2.line(frame, (0, roi_y1), (w, roi_y1), roi_color, 1)
    cv2.line(frame, (0, roi_y2), (w, roi_y2), roi_color, 1)
    cv2.putText(frame, "ROI", (8, roi_y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)

    # ── Path contour fill ─────────────────────────────────────────────────────
    if contour_pts is not None:
        cv2.drawContours(frame, [contour_pts], -1, (0, 200, 255), 2)

    # ── Centroid marker + centre line ────────────────────────────────────────
    centre_x = w // 2
    cv2.line(frame, (centre_x, roi_y1), (centre_x, roi_y2), (255, 255, 0), 1)

    if cx_frame is not None:
        cy_draw = (roi_y1 + roi_y2) // 2
        cv2.circle(frame, (cx_frame, cy_draw), 10, (0, 255, 0), -1)
        cv2.line(frame, (centre_x, cy_draw), (cx_frame, cy_draw), (255, 200, 0), 2)

    # ── Status banner (bottom) ───────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 46), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    if not auto:
        st, sc = "■  PAUSED  (SPACE to resume)", (100, 100, 255)
    elif blocked:
        st, sc = "●  STOPPED — PATH LOST / BLOCKED", (0, 60, 255)
    else:
        st, sc = f"▶  {direction}", (60, 230, 80)

    cv2.putText(frame, st, (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, sc, 2, cv2.LINE_AA)

    # ── Top-left stats ────────────────────────────────────────────────────────
    stats = [
        f"FPS: {fps:.0f}",
        f"Throttle: {throttle}%  Steering: {steering}%",
        f"Path fill: {path_fill:.1f}%",
        f"Threshold: {thresh_val}  (+/- to adjust)",
        f"Debug B&W: {'ON' if show_dbg else 'OFF'}  (T to toggle)",
    ]
    for i, s in enumerate(stats):
        cv2.putText(frame, s, (12, 26 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════════════
#  Main Loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global auto_mode, show_debug, threshold_val

    connect_mqtt()
    send_command(0, 0)

    cam_idx = find_best_camera()
    cap     = open_camera(cam_idx)

    prev_t           = time.time()
    path_lost_since  = None   # timestamp when path first disappeared

    # Key edge-detection state
    k_space = k_quit = k_debug = k_plus = k_minus = False

    print("\n" + "="*60)
    print("  AUTONOMOUS PATH-FOLLOWING ROVER")
    print("="*60)
    print("  Camera → Greyscale → B&W threshold → Contour path")
    print(f"  Forward throttle : {THROTTLE_FORWARD}%")
    print(f"  Threshold value  : {threshold_val}  (use +/- to tune)")
    print("  SPACE=pause/resume   T=debug view   +/-=threshold   Q=quit")
    print("="*60 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] No frame — is OBS Virtual Camera running?")
                time.sleep(0.3)
                continue

            h, w = frame.shape[:2]
            now  = time.time()
            fps  = 1.0 / max(now - prev_t, 1e-9)
            prev_t = now

            # ── Path detection ────────────────────────────────────────────
            binary_view, cx_frame, path_fill, contour_pts = detect_path(frame, threshold_val)

            # ── Obstacle / path-lost logic ────────────────────────────────
            path_present = (cx_frame is not None and path_fill >= MIN_PATH_FILL_PCT)

            if not path_present:
                if path_lost_since is None:
                    path_lost_since = now   # start counting
                blocked = (now - path_lost_since) >= BLOCK_HOLD_SECS
            else:
                path_lost_since = None
                blocked = False

            # ── Drive decision ────────────────────────────────────────────
            if auto_mode and not blocked and path_present:
                throttle, steering, direction = steer_decision(cx_frame, w)
            else:
                throttle, steering = 0, 0
                direction = "● STOPPED"

            send_command(throttle, steering)

            # ── Key handling (edge-detected) ──────────────────────────────
            # SPACE — pause / resume
            if keyboard.is_pressed('space'):
                if not k_space:
                    k_space = True
                    auto_mode = not auto_mode
                    if not auto_mode:
                        send_command(0, 0)
                    print(f"[KEY] Auto mode: {'ON' if auto_mode else 'PAUSED'}")
            else:
                k_space = False

            # T — toggle debug B&W view
            if keyboard.is_pressed('t'):
                if not k_debug:
                    k_debug = True
                    show_debug = not show_debug
                    print(f"[KEY] Debug view: {'ON' if show_debug else 'OFF'}")
            else:
                k_debug = False

            # + / = — raise threshold
            if keyboard.is_pressed('+') or keyboard.is_pressed('='):
                if not k_plus:
                    k_plus = True
                    threshold_val = min(threshold_val + THRESHOLD_STEP, 245)
                    print(f"[KEY] Threshold → {threshold_val}")
            else:
                k_plus = False

            # - — lower threshold
            if keyboard.is_pressed('-'):
                if not k_minus:
                    k_minus = True
                    threshold_val = max(threshold_val - THRESHOLD_STEP, 10)
                    print(f"[KEY] Threshold → {threshold_val}")
            else:
                k_minus = False

            # Q / ESC — quit
            if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
                if not k_quit:
                    k_quit = True
                    print("[KEY] Quit.")
                    break

            # ── Build display frame ───────────────────────────────────────
            # Blend B&W debug view over original when debug is ON
            if show_debug:
                display = cv2.addWeighted(frame, 0.55, binary_view, 0.45, 0)
            else:
                display = frame.copy()

            draw_overlay(display, cx_frame, path_fill, contour_pts,
                         throttle, steering, direction, blocked,
                         auto_mode, fps, threshold_val, show_debug)

            cv2.imshow("Autonomous Rover  |  SPACE=pause  T=debug  +/-=thresh  Q=quit", display)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        print("[INFO] Sending stop…")
        send_command(0, 0)
        time.sleep(0.4)
        cap.release()
        cv2.destroyAllWindows()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        print("[INFO] Done.")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Autonomous Rover — Grayscale Path Detection")
    print("="*60)
    print("\nChecklist:")
    print("  1. OBS open + Virtual Camera STARTED")
    print("  2. CCTV feed showing the floor/road/path")
    print("  3. ESP32 rover powered ON")
    print("\nTip: Press T to see the B&W view and use +/- to tune")
    print("     the threshold until the floor turns WHITE and")
    print("     obstacles/walls turn BLACK (or vice versa).\n")
    time.sleep(2)
    main()
