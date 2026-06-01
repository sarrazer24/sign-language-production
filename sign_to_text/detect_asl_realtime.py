import cv2, time, os
from ultralytics import YOLO

# Chemins relatifs depuis ce fichier — marche sur tous les PCs
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "best.pt")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.txt")

CONF_THRESHOLD = 0.5
STABILITY_N    = 5   # frames consécutives pour confirmer un signe

# Load label map
with open(LABEL_MAP_PATH) as f:
    LABELS = {int(l.split(":")[0]): l.split(":")[1].strip() for l in f}

# Load model
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

history      = []
frame_buffer = []
prev_time    = time.time()

print("ASL Detector running")
print("Q = quit  |  C = clear history  |  S = screenshot")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results    = model(frame, imgsz=320, conf=CONF_THRESHOLD, verbose=False)
    boxes      = results[0].boxes
    best_label = None
    best_conf  = 0.0

    for box in boxes:
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = LABELS.get(cls, str(cls))
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.0%}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        if conf > best_conf:
            best_conf  = conf
            best_label = label

    # Stability filter
    frame_buffer.append(best_label)
    if len(frame_buffer) > STABILITY_N:
        frame_buffer.pop(0)
    if (len(frame_buffer) == STABILITY_N
            and len(set(frame_buffer)) == 1
            and best_label is not None
            and best_conf >= 0.7):
        if not history or history[-1] != best_label:
            history.append(best_label)
            if len(history) > 7:
                history.pop(0)

    # FPS
    now       = time.time()
    fps       = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # History bar
    hf = frame.shape[0]
    cv2.rectangle(frame, (0, hf - 50), (frame.shape[1], hf), (0, 0, 0), -1)
    txt = "  |  ".join(history) if history else "No sign detected"
    cv2.putText(frame, txt, (10, hf - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("ASL Real-Time Detection", frame)
    key = cv2.waitKey(30) & 0xFF
    if   key == ord("q"): break
    elif key == ord("c"): history.clear(); frame_buffer.clear()
    elif key == ord("s"):
        fname = os.path.join(BASE_DIR, f"screenshot_{int(time.time())}.jpg")
        cv2.imwrite(fname, frame)
        print(f"Screenshot: {fname}")

cap.release()
cv2.destroyAllWindows()