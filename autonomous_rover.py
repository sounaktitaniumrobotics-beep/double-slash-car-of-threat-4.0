"""
Autonomous CCTV Rover — YOLOv8 + MQTT
======================================
The rover drives itself forward continuously at a slow safe speed.
It stops automatically when an obstacle is detected in the "danger zone"
(the central region of the frame) and resumes when the path is clear.

Keyboard controls (only two):
    SPACE  — Toggle autonomous mode ON / OFF (pause / resume)
    Q / ESC — Quit and stop the rover safely

Hardware:
    - 150 RPM tank-style bot → slow speed (15% throttle) to avoid accidents
    - OBS Virtual Camera → YOLOv8 → obstacle detection
    - MQTT over TLS → ESP32 motor controller

Requirements:
    pip install paho-mqtt ultralytics opencv-python keyboard
"""

import cv2
import time
import threading
import paho.mqtt.client as mqtt
from ultralytics import YOLO
import keyboard

# ══════════════════════════════════════════════════════════════════════════════
#  MQTT Configuration
# ══════════════════════════════════════════════════════════════════════════════
BROKER   = "1887637a30c2432683efd36d80813f78.s1.eu.hivemq.cloud"
PORT     = 8883
USERNAME = "titaniumrobotics"
PASSWORD = "Sapt090059#"
TOPIC    = "rccar/control"

# ══════════════════════════════════════════════════════════════════════════════
#  Rover Speed / Safety Config
# ══════════════════════════════════════════════════════════════════════════════
# 150 RPM motors — keep throttle LOW to avoid accidents
FORWARD_THROTTLE   = 15     # 15% forward speed (safe for 150 RPM)
STEERING_TRIM      = 0      # 0 = straight; adjust if bot drifts

# ══════════════════════════════════════════════════════════════════════════════
#  Obstacle Detection Config
# ══════════════════════════════════════════════════════════════════════════════
MODEL_NAME        = "yolov8n.pt"   # Nano = fastest inference
CONF_THRESHOLD    = 0.45

# Danger zone: central band of the frame where obstacles trigger a stop
# Values are fractions of frame width/height  (0.0 → 1.0)
DANGER_ZONE_X1    = 0.25   # left boundary   (25% from left)
DANGER_ZONE_X2    = 0.75   # right boundary  (75% from left)
DANGER_ZONE_Y1    = 0.30   # top boundary    (30% from top)
DANGER_ZONE_Y2    = 0.90   # bottom boundary (90% from top) — ignore sky

# Minimum fraction of the frame a bounding box must occupy to count as obstacle
# (filters out tiny detections that might be noise)
MIN_BOX_AREA_FRAC = 0.005  # 0.5% of frame area

# Classes to IGNORE as obstacles (things that are fine to drive near)
IGNORE_CLASSES    = {}      # e.g. {"sky", "ceiling"} — empty = block everything

# How long (seconds) to keep stopped after an obstacle disappears
# (hysteresis — avoids rapid start/stop oscillation)
CLEAR_HOLD_SECS   = 0.8

# ══════════════════════════════════════════════════════════════════════════════
#  Camera Config
# ══════════════════════════════════════════════════════════════════════════════
MAX_CAMERAS       = 6
FRAME_WIDTH       = 1280
FRAME_HEIGHT      = 720

# ══════════════════════════════════════════════════════════════════════════════
#  Shared State
# ══════════════════════════════════════════════════════════════════════════════
mqtt_client         = None
obstacle_detected   = False     # set by vision thread
auto_mode           = True      # SPACE toggles this
last_clear_time     = 0.0       # when obstacle last disappeared
state_lock          = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  MQTT Helpers
# ══════════════════════════════════════════════════════════════════════════════

def on_connect(client, userdata, flags, reason_code, properties):
    status = "OK" if reason_code == 0 else f"FAILED ({reason_code})"
    print(f"[MQTT] Connected: {status}")


def send_command(throttle: int, steering: int):
    """Publish a throttle,steering command."""
    if mqtt_client:
        mqtt_client.publish(TOPIC, f"{throttle},{steering}", qos=1)


def connect_mqtt():
    global mqtt_client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()
    client.on_connect = on_connect
    mqtt_client = client
    print("[MQTT] Connecting to broker…")
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    time.sleep(2)


# ══════════════════════════════════════════════════════════════════════════════
#  Camera Helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_best_camera():
    print("[CAM]  Scanning for cameras…")
    available = []
    for i in range(MAX_CAMERAS):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                backend = cap.getBackendName()
                print(f"  [Camera {i}] Found — {backend}")
                available.append(i)
        cap.release()
    if not available:
        raise RuntimeError("No cameras found. Is OBS Virtual Camera running?")
    preferred = [i for i in available if i > 0]
    idx = preferred[0] if preferred else available[0]
    print(f"[CAM]  Using camera index {idx}")
    return idx


def open_camera(index: int):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


# ══════════════════════════════════════════════════════════════════════════════
#  Obstacle Detection Helpers
# ══════════════════════════════════════════════════════════════════════════════

def boxes_in_danger_zone(results, frame_h: int, frame_w: int) -> list:
    """Return list of (label, x1,y1,x2,y2) detections that overlap the danger zone."""
    dz_x1 = int(DANGER_ZONE_X1 * frame_w)
    dz_x2 = int(DANGER_ZONE_X2 * frame_w)
    dz_y1 = int(DANGER_ZONE_Y1 * frame_h)
    dz_y2 = int(DANGER_ZONE_Y2 * frame_h)
    frame_area = frame_h * frame_w

    hits = []
    names = results.names
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = names[int(box.cls[0])]

        if label in IGNORE_CLASSES:
            continue

        # Box must be large enough to count
        box_area = (x2 - x1) * (y2 - y1)
        if box_area / frame_area < MIN_BOX_AREA_FRAC:
            continue

        # Check overlap with danger zone (intersection)
        ix1 = max(x1, dz_x1);  iy1 = max(y1, dz_y1)
        ix2 = min(x2, dz_x2);  iy2 = min(y2, dz_y2)
        if ix2 > ix1 and iy2 > iy1:
            hits.append((label, x1, y1, x2, y2))
    return hits


# ══════════════════════════════════════════════════════════════════════════════
#  Overlay Drawing
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10),  (146, 204, 23), (61, 219, 134),
    (26, 147, 52),  (0, 212, 187),  (44, 153, 168), (0, 194, 255),
    (52, 69, 147),  (100, 115, 255),(0, 24, 236),   (132, 56, 255),
]

def draw_boxes(frame, results):
    names = results.names
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        score  = float(box.conf[0])
        label  = names[cls_id]
        c = PALETTE[cls_id % len(PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        text = f"{label} {score:.0%}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - bl - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_danger_zone(frame, obstacle: bool):
    h, w = frame.shape[:2]
    x1 = int(DANGER_ZONE_X1 * w);  y1 = int(DANGER_ZONE_Y1 * h)
    x2 = int(DANGER_ZONE_X2 * w);  y2 = int(DANGER_ZONE_Y2 * h)
    color = (0, 0, 255) if obstacle else (0, 255, 100)
    # Semi-transparent fill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = "DANGER ZONE — OBSTACLE" if obstacle else "DANGER ZONE — CLEAR"
    cv2.putText(frame, label, (x1 + 6, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_hud(frame, fps: float, auto: bool, obstacle: bool, obj_count: int):
    # Status banner at bottom
    h, w = frame.shape[:2]
    banner_h = 42
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    if not auto:
        state_txt   = "■ PAUSED (SPACE to resume)"
        state_color = (80, 80, 255)
    elif obstacle:
        state_txt   = "⚠  STOPPED — OBSTACLE DETECTED"
        state_color = (0, 60, 255)
    else:
        state_txt   = "▶  MOVING FORWARD"
        state_color = (50, 220, 80)

    cv2.putText(frame, state_txt, (16, h - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2, cv2.LINE_AA)

    # Top-right mini stats
    cv2.putText(frame, f"FPS {fps:.0f}  |  Objects: {obj_count}  |  Throttle: {FORWARD_THROTTLE}%",
                (w - 380, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # SPACE hint
    cv2.putText(frame, "SPACE=pause/resume  |  Q=quit",
                (w - 310, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global obstacle_detected, auto_mode, last_clear_time

    # ── Connect MQTT ────────────────────────────────────────────────────────
    connect_mqtt()

    # ── Load YOLO ───────────────────────────────────────────────────────────
    print(f"[YOLO] Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    print("[YOLO] Model ready.")

    # ── Open Camera ─────────────────────────────────────────────────────────
    cam_idx = find_best_camera()
    cap     = open_camera(cam_idx)

    prev_t  = time.time()

    # Key state tracking (edge detection)
    space_was_pressed = False
    quit_was_pressed  = False

    print("\n" + "="*60)
    print("  AUTONOMOUS ROVER — RUNNING")
    print("="*60)
    print(f"  Forward throttle : {FORWARD_THROTTLE}%  (safe for 150 RPM)")
    print(f"  Danger zone      : X {DANGER_ZONE_X1:.0%}→{DANGER_ZONE_X2:.0%}  "
          f"Y {DANGER_ZONE_Y1:.0%}→{DANGER_ZONE_Y2:.0%}")
    print(f"  SPACE            : Pause / Resume autonomous mode")
    print(f"  Q / ESC          : Emergency stop & quit")
    print("="*60 + "\n")

    # Make sure bot is stopped at start
    send_command(0, 0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] No frame — is OBS Virtual Camera running?")
                time.sleep(0.5)
                continue

            h, w = frame.shape[:2]

            # ── YOLO Inference ───────────────────────────────────────────
            results   = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            hits      = boxes_in_danger_zone(results, h, w)
            obj_count = len(results.boxes)

            # ── Update obstacle state ────────────────────────────────────
            now = time.time()
            with state_lock:
                if hits:
                    obstacle_detected = True
                    last_clear_time   = 0.0   # reset clear timer
                else:
                    if obstacle_detected:
                        # Start the clear hold timer
                        if last_clear_time == 0.0:
                            last_clear_time = now
                        elif now - last_clear_time >= CLEAR_HOLD_SECS:
                            obstacle_detected = False
                    # else: already clear, nothing to do

            # ── Drive Decision ───────────────────────────────────────────
            with state_lock:
                should_move = auto_mode and not obstacle_detected

            send_command(FORWARD_THROTTLE if should_move else 0, STEERING_TRIM)

            # ── Key Handling (edge-detected) ─────────────────────────────
            # SPACE — toggle auto mode
            if keyboard.is_pressed('space'):
                if not space_was_pressed:
                    space_was_pressed = True
                    with state_lock:
                        auto_mode = not auto_mode
                    if not auto_mode:
                        send_command(0, 0)
                    print(f"[KEY]  Auto mode: {'ON' if auto_mode else 'PAUSED'}")
            else:
                space_was_pressed = False

            # Q / ESC — quit
            if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
                if not quit_was_pressed:
                    quit_was_pressed = True
                    print("\n[KEY]  Quit requested — stopping rover…")
                    break

            # ── Draw Overlay ─────────────────────────────────────────────
            draw_danger_zone(frame, bool(hits))
            draw_boxes(frame, results)

            fps = 1.0 / max(now - prev_t, 1e-9)
            prev_t = now

            with state_lock:
                _auto = auto_mode
                _obs  = obstacle_detected

            draw_hud(frame, fps, _auto, bool(hits), obj_count)

            # ── Display ──────────────────────────────────────────────────
            cv2.imshow("Autonomous Rover  |  SPACE=pause  Q=quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):   # 27 = ESC in OpenCV
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        print("[INFO] Emergency stop sent.")
        send_command(0, 0)
        time.sleep(0.5)
        cap.release()
        cv2.destroyAllWindows()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Autonomous CCTV Rover — YOLOv8 + MQTT")
    print("="*60)
    print("\nPre-flight checklist:")
    print("  1. OBS is open and Virtual Camera is STARTED")
    print("  2. CCTV/webcam feed is visible in OBS")
    print("  3. ESP32 rover is powered ON and connected to WiFi")
    print("  4. MQTT broker is reachable")
    print("\nStarting in 3 seconds…\n")
    time.sleep(3)
    main()
