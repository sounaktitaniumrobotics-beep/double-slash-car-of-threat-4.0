"""
Real-Time Object Detection — OBS Virtual Camera + YOLOv8 + Claude Vision
=========================================================================
1. Open OBS → add your True Cloud / CCTV feed as a source
2. Click "Start Virtual Camera" in OBS
3. Run this script — it will auto-detect the OBS virtual camera

Requirements:
    pip install ultralytics opencv-python anthropic

Controls:
    Q        - Quit
    S        - Save screenshot
    +/-      - Adjust confidence threshold
    N        - Switch to next camera device
    I        - Toggle Claude AI identification panel
"""

import cv2
import time
import base64
import threading
import anthropic
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME           = "yolov8n.pt"   # nano=fast; s/m/l/x=more accurate
CONF_THRESHOLD       = 0.45
CONF_STEP            = 0.05
WINDOW_TITLE         = "CCTV Object Detection (OBS)  |  Q=quit  S=save  +/-=conf  N=next cam  I=AI info"
MAX_CAMERAS          = 5              # how many indices to scan
AI_IDENTIFY_COOLDOWN = 3.0           # seconds between Claude calls per object label
SHOW_AI_PANEL        = True          # toggle with I key
# ────────────────────────────────────────────────────────────────────────────

PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10),  (146, 204, 23), (61, 219, 134),
    (26, 147, 52),  (0, 212, 187),  (44, 153, 168), (0, 194, 255),
    (52, 69, 147),  (100, 115, 255),(0, 24, 236),   (132, 56, 255),
    (82, 0, 133),   (203, 56, 255), (255, 149, 200),(255, 55, 199),
]

# ── Shared AI state ──────────────────────────────────────────────────────────
ai_info_cache:  dict[str, str]   = {}   # label  →  Claude description
ai_last_called: dict[str, float] = {}   # label  →  last API call timestamp
ai_lock = threading.Lock()

client = anthropic.Anthropic()


# ── Helpers ──────────────────────────────────────────────────────────────────

def colour_for(class_id: int):
    return PALETTE[class_id % len(PALETTE)]


def draw_box(frame, x1, y1, x2, y2, label, conf, class_id):
    c = colour_for(class_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
    text = f"{label} {conf:.0%}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 6), (x1 + tw + 4, y1), c, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def encode_crop(crop_bgr) -> str:
    """Encode a BGR numpy array to a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buf).decode("utf-8")


def _identify_worker(label: str, b64: str):
    """Background thread: ask Claude to identify the cropped object."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"This crop comes from a live CCTV feed. "
                            f"A YOLO detector labelled it '{label}'. "
                            "Identify the object as specifically as possible: "
                            "its precise name, notable characteristics (color, brand, "
                            "model, activity, condition, etc.), and any useful context. "
                            "Be concise — 2-3 sentences maximum."
                        ),
                    },
                ],
            }],
        )
        info = response.content[0].text.strip()
    except Exception as exc:
        info = f"[Claude error: {exc}]"

    with ai_lock:
        ai_info_cache[label] = info
    print(f"[AI] {label}: {info}")


def maybe_identify(label: str, crop_bgr):
    """Fire an async Claude Vision call if the per-label cooldown has elapsed."""
    now = time.time()
    with ai_lock:
        if now - ai_last_called.get(label, 0) < AI_IDENTIFY_COOLDOWN:
            return
        ai_last_called[label] = now

    b64 = encode_crop(crop_bgr)
    threading.Thread(target=_identify_worker, args=(label, b64), daemon=True).start()


def draw_ai_panel(frame, detections: list[tuple[str, str]]):
    """Render a semi-transparent panel on the right side with AI descriptions."""
    if not detections:
        return

    PANEL_W   = 420
    LINE_H    = 18
    PAD       = 8
    WRAP_COLS = 55
    panel_x   = frame.shape[1] - PANEL_W - PAD

    # Build text lines
    lines: list[tuple[str, bool]] = []   # (text, is_header)
    for label, info in detections:
        lines.append((label.upper(), True))
        words, row = info.split(), ""
        for w in words:
            if len(row) + len(w) + 1 > WRAP_COLS:
                lines.append((row.strip(), False))
                row = w + " "
            else:
                row += w + " "
        if row.strip():
            lines.append((row.strip(), False))
        lines.append(("", False))   # spacer

    panel_h = len(lines) * LINE_H + PAD * 2

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x - PAD,        10),
                  (panel_x + PANEL_W,    10 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 10 + PAD + LINE_H
    for text, is_header in lines:
        if not text:
            y += LINE_H // 2
            continue
        color = (0, 220, 255) if is_header else (220, 220, 220)
        scale = 0.48 if is_header else 0.43
        thick = 2   if is_header else 1
        cv2.putText(frame, text, (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += LINE_H


# ── Camera helpers ────────────────────────────────────────────────────────────

def find_obs_camera():
    print("[INFO] Scanning for camera devices...")
    available = []
    for i in range(MAX_CAMERAS):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"  [Camera {i}] Found — backend: {cap.getBackendName()}")
                available.append(i)
        cap.release()

    if not available:
        raise RuntimeError("No camera devices found. Is OBS Virtual Camera running?")

    preferred = [i for i in available if i > 0]
    chosen = preferred[0] if preferred else available[0]
    print(f"[INFO] Using camera index: {chosen}")
    return chosen, available


def open_camera(index: int):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    global SHOW_AI_PANEL

    print(f"[INFO] Loading YOLO model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    names = model.names

    cam_index, _ = find_obs_camera()
    cap = open_camera(cam_index)

    conf   = CONF_THRESHOLD
    prev_t = time.time()

    print("[INFO] Running — Q=quit  S=save  +/-=confidence  N=next cam  I=toggle AI")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No frame received. Is OBS Virtual Camera running?")
            cv2.waitKey(500)
            continue

        # ── YOLO inference ───────────────────────────────────────────────
        results    = model(frame, conf=conf, verbose=False)[0]
        obj_count  = 0
        timestamp  = int(time.time() * 1000)
        seen_labels: dict[str, str] = {}   # deduplicated for panel

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            score  = float(box.conf[0])
            label  = names[cls_id]

            draw_box(frame, x1, y1, x2, y2, label, score, cls_id)

            # Save crop
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(f"crop_{label}_{score:.0%}_{timestamp}_{obj_count}.jpg", crop)
                # Trigger Claude Vision identification (background, rate-limited)
                maybe_identify(label, crop.copy())

            # Collect AI description for panel
            if label not in seen_labels:
                with ai_lock:
                    seen_labels[label] = ai_info_cache.get(label, "Identifying…")

            obj_count += 1

        # ── AI info panel ────────────────────────────────────────────────
        if SHOW_AI_PANEL:
            draw_ai_panel(frame, list(seen_labels.items()))

        # ── HUD overlay ──────────────────────────────────────────────────
        now    = time.time()
        fps    = 1.0 / (now - prev_t + 1e-9)
        prev_t = now

        cv2.putText(frame, f"FPS: {fps:.1f}",         (10, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.8,  (0, 255, 0),     2)
        cv2.putText(frame, f"Objects: {obj_count}",    (10, 58),  cv2.FONT_HERSHEY_SIMPLEX, 0.8,  (0, 220, 255),   2)
        cv2.putText(frame, f"Conf >= {conf:.0%}",      (10, 88),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(frame, f"Cam: {cam_index}",        (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        ai_color = (100, 255, 100) if SHOW_AI_PANEL else (100, 100, 100)
        cv2.putText(frame, "AI: ON" if SHOW_AI_PANEL else "AI: OFF",
                    (10, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ai_color, 1)

        cv2.imshow(WINDOW_TITLE, frame)

        # ── Key handling ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"cctv_frame_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Saved {fname}")
        elif key in (ord('+'), ord('=')):
            conf = min(conf + CONF_STEP, 0.95)
            print(f"[INFO] Confidence → {conf:.0%}")
        elif key == ord('-'):
            conf = max(conf - CONF_STEP, 0.05)
            print(f"[INFO] Confidence → {conf:.0%}")
        elif key == ord('i'):
            SHOW_AI_PANEL = not SHOW_AI_PANEL
            print(f"[INFO] AI panel {'ON' if SHOW_AI_PANEL else 'OFF'}")
        elif key == ord('n'):
            cap.release()
            cam_index = (cam_index + 1) % MAX_CAMERAS
            for _ in range(MAX_CAMERAS):
                test = cv2.VideoCapture(cam_index)
                if test.isOpened():
                    test.release()
                    break
                test.release()
                cam_index = (cam_index + 1) % MAX_CAMERAS
            cap = open_camera(cam_index)
            print(f"[INFO] Switched to camera index: {cam_index}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    main()
