import base64
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.txt")
INFERENCE_SIZE = (320, 240)
DISPLAY_SIZE = (640, 480)
CONFIDENCE = 0.5
CACHE_TTL_SECONDS = 1.0
USE_HALF = bool(torch is not None and torch.cuda.is_available())
JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 75]

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)

model = YOLO(MODEL_PATH)
LABELS = {}
with open(LABEL_MAP_PATH, encoding="utf-8") as f:
    LABELS = {int(l.split(":")[0]): l.split(":")[1].strip() for l in f}

model_lock = threading.Lock()
camera_lock = threading.Lock()
camera = None
stream_running = False
last_prediction = None
last_confidence = None
last_prediction_at = 0.0
history = []


def _run_model(img):
    return model(
        img,
        imgsz=320,
        conf=CONFIDENCE,
        half=USE_HALF,
        verbose=False,
    )


def _warmup_model():
    start = time.perf_counter()
    blank = np.zeros((INFERENCE_SIZE[1], INFERENCE_SIZE[0], 3), dtype=np.uint8)
    with model_lock:
        _run_model(blank)
    return round((time.perf_counter() - start) * 1000, 2)


warmup_ms = _warmup_model()
print(
    f"Model loaded - {len(LABELS)} classes - "
    f"half={USE_HALF} - warmup={warmup_ms}ms"
)


def _open_camera():
    global camera

    with camera_lock:
        if camera is not None and camera.isOpened():
            return camera

        backend = cv2.CAP_DSHOW if os.name == "nt" else 0
        camera = cv2.VideoCapture(0, backend)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_SIZE[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_SIZE[1])
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not camera.isOpened():
            camera.release()
            camera = None
            raise RuntimeError("Could not open webcam")

        return camera


def _release_camera():
    global camera

    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None


def _detections_from_results(results):
    boxes = results[0].boxes
    dets = []
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        dets.append(
            {
                "label": LABELS.get(cls, str(cls)),
                "confidence": round(conf, 3),
            }
        )
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return dets


def _remember_prediction(label, confidence):
    global last_confidence, last_prediction, last_prediction_at, history

    last_prediction = label
    last_confidence = confidence
    last_prediction_at = time.perf_counter()

    if not history or history[-1] != label:
        history = (history + [label])[-7:]


def _cached_detection():
    if (
        last_prediction is None
        or time.perf_counter() - last_prediction_at > CACHE_TTL_SECONDS
    ):
        return None

    return {
        "label": last_prediction,
        "confidence": last_confidence,
        "cached": True,
    }


def infer_frame(frame):
    small = cv2.resize(frame, INFERENCE_SIZE)
    with model_lock:
        results = _run_model(small)

    dets = _detections_from_results(results)
    if dets:
        top = dets[0]
        _remember_prediction(top["label"], top["confidence"])
        return dets

    cached = _cached_detection()
    return [cached] if cached else []


def infer_bytes(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")

    return infer_frame(img)


def _draw_label(frame, label):
    if not label:
        return

    text = str(label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    box_w = text_size[0] + 34
    box_h = text_size[1] + 24
    x = max((frame.shape[1] - box_w) // 2, 10)
    y = 24

    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (207, 79, 91), -1)
    cv2.putText(
        frame,
        text,
        (x + 17, y + box_h - 14),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _draw_live_badge(frame):
    x2 = frame.shape[1] - 18
    y1 = 20
    x1 = x2 - 86
    y2 = y1 + 34
    cv2.rectangle(frame, (x1, y1), (x2, y2), (55, 68, 245), -1)
    cv2.circle(frame, (x1 + 18, y1 + 17), 5, (255, 255, 255), -1)
    cv2.putText(
        frame,
        "LIVE",
        (x1 + 32, y1 + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_history(frame):
    text = "  |  ".join(history) if history else "Waiting for signs..."
    h, w = frame.shape[:2]
    y1 = h - 66
    y2 = h - 26
    x1 = 14
    x2 = w - 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(
        frame,
        text[:95],
        (x1 + 14, y1 + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _annotate_frame(frame, detections):
    top = detections[0] if detections else None
    _draw_label(frame, top["label"] if top else "")
    _draw_live_badge(frame)
    _draw_history(frame)
    return frame


def _error_frame(message):
    frame = np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (40, DISPLAY_SIZE[1] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _encode_frame(frame):
    ok, encoded = cv2.imencode(".jpg", frame, JPEG_PARAMS)
    if not ok:
        return None
    return encoded.tobytes()


def generate_frames():
    global stream_running

    stream_running = True
    try:
        cap = _open_camera()
        while stream_running:
            ok, frame = cap.read()
            if not ok:
                frame = _error_frame("Camera frame unavailable")
                time.sleep(0.05)
            else:
                frame = cv2.flip(frame, 1)
                detections = infer_frame(frame)
                frame = _annotate_frame(frame, detections)

            jpg = _encode_frame(frame)
            if jpg is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + jpg
                + b"\r\n"
            )
    except GeneratorExit:
        pass
    except Exception as e:
        jpg = _encode_frame(_error_frame(str(e)))
        if jpg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpg
                + b"\r\n"
            )
    finally:
        stream_running = False
        _release_camera()


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "warmup_ms": warmup_ms,
            "stream_running": stream_running,
        }
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.route("/stop_stream", methods=["POST"])
def stop_stream():
    global stream_running

    stream_running = False
    _release_camera()
    return jsonify({"status": "stopped"})


@app.route("/predict_base64", methods=["POST"])
def predict_b64():
    try:
        data = request.get_json(silent=True) or {}
        image = data.get("image")
        if not image:
            return jsonify({"success": False, "error": "Missing image"}), 400

        img_bytes = base64.b64decode(image)
        dets = infer_bytes(img_bytes)
        top = dets[0] if dets else None
        return jsonify(
            {
                "success": True,
                "detections": dets,
                "top_prediction": top["label"] if top else None,
                "top_confidence": top["confidence"] if top else None,
                "cached": bool(top and top.get("cached")),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
