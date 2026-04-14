# -*- coding: utf-8 -*-
"""Detection




from google.colab import drive
drive.mount('/content/drive')

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="")
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(11)
dataset = version.download("yolo26")

# Install required libraries

!pip install  ultralytics opencv-python easyocr

from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data=f"{dataset.location}/data.yaml",

    epochs=80,
    imgsz=640,
    batch=8,

    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    patience=15,

    augment=True,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5
)

# =========================
# 4. Auto Save to Drive
# =========================
import os
import shutil

base_path = "runs/detect"
train_folders = sorted(os.listdir(base_path))
last_train = train_folders[-1]

best_path = f"{base_path}/{last_train}/weights/best.pt"
last_path = f"{base_path}/{last_train}/weights/last.pt"

# نسخ للـ Drive
shutil.copy(best_path, "/content/drive/MyDrive/best.pt")
shutil.copy(last_path, "/content/drive/MyDrive/last.pt")

print("✅ Model saved to Google Drive!")

from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(8501)"))

# Install all required packages
!pip install ultralytics easyocr opencv-python-headless filterpy -q
!pip install numpy pandas -q
!git clone https://github.com/abewley/sort.git /content/sort 2>/dev/null || echo "SORT already cloned"

from google.colab import drive
drive.mount('/content/drive')

import os
model_path = "/content/drive/MyDrive/best.pt"
print("✅ Model found:" if os.path.exists(model_path) else "❌ Model NOT found!")

import matplotlib
matplotlib.use('Agg')

import cv2, os, sys, csv, time
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import defaultdict, Counter
from google.colab import files
from IPython.display import HTML, display
from base64 import b64encode

# ──────────────── Settings ────────────────
MODEL_PATH   = "/content/drive/MyDrive/best.pt"
CONF_THRESH  = 0.4
OCR_CONF     = 0.25
TRAIL_LEN    = 30
VOTE_FRAMES  = 8
LANGUAGES    = ['ar', 'en']

# ──────────────── Load Models ────────────────
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("Loading EasyOCR (ar + en)...")
reader = easyocr.Reader(LANGUAGES, gpu=False)
print("All models loaded!")

# ──────────────── Upload Video ────────────────
print("Upload your video...")
uploaded = files.upload()
video_path = list(uploaded.keys())[0]

# ──────────────── Video Setup ────────────────
cap    = cv2.VideoCapture(video_path)
W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS    = int(cap.get(cv2.CAP_PROP_FPS))
TOTAL  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter('output_pro.mp4', fourcc, FPS, (W, H))

# ──────────────── Tracking Variables ────────────────
trails      = defaultdict(list)   # movement trails per ID
plate_votes = defaultdict(list)   # OCR voting buffer per ID
best_plates = {}                  # best plate text per ID
seen_ids    = set()               # all vehicle IDs seen
csv_rows    = []                  # data for CSV export
track_history = defaultdict(list) # ultralytics track history
frame_count = 0

# ──────────────── Colors ────────────────
COLORS = [
    (255,100,100), (100,255,100), (100,100,255),
    (255,255,100), (100,255,255), (255,100,255),
    (255,180,50),  (50,255,180),  (180,50,255),
]

def get_color(track_id):
    return COLORS[track_id % len(COLORS)]

def draw_text(img, text, pos, color=(0,255,0), scale=0.75):
    # Draw text with black outline for readability
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 4)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def draw_dashboard(frame, frame_num, vehicle_count, plates_found):
    # Draw semi-transparent dashboard in top-left corner
    overlay = frame.copy()
    cv2.rectangle(overlay, (10,10), (270,125), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (10,10), (270,125), (80,80,80), 1)
    t = frame_num / FPS
    draw_text(frame, f"Vehicles : {vehicle_count}",       (20, 38),  (100,255,100))
    draw_text(frame, f"Plates   : {plates_found}",        (20, 65),  (100,200,255))
    draw_text(frame, f"Frame    : {frame_num}/{TOTAL}",   (20, 92),  (200,200,200))
    draw_text(frame, f"Time     : {t:.1f}s",              (20, 119), (200,200,200))

def run_ocr(plate_img):
    # Preprocess plate image then run OCR
    if plate_img is None or plate_img.size == 0:
        return "", 0.0
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results = reader.readtext(binary, detail=1)
    if not results:
        return "", 0.0
    best = max(results, key=lambda x: x[2])
    return best[1].strip(), best[2]

# ──────────────── Main Processing Loop ────────────────
print(f"Processing {TOTAL} frames at {FPS} FPS...")
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Run YOLO tracking (built-in tracker, no external deps)
    results = model.track(frame, conf=CONF_THRESH, persist=True, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy().astype(int)

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            color = get_color(tid)
            seen_ids.add(tid)

            # Update and draw trailing path
            trails[tid].append((cx, cy))
            if len(trails[tid]) > TRAIL_LEN:
                trails[tid].pop(0)
            trail = trails[tid]
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                thickness = max(1, int(3 * alpha))
                c = tuple(int(v * alpha) for v in color)
                cv2.line(frame, trail[i-1], trail[i], c, thickness)

            # Draw bounding box and ID label
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.rectangle(frame, (x1, y1-28), (x1+85, y1), color, -1)
            draw_text(frame, f"ID:{tid}", (x1+4, y1-8), (255,255,255))

            # Run OCR every 5 frames to save time
            if frame_count % 5 == 0:
                plate_img = frame[y1:y2, x1:x2]
                text, conf = run_ocr(plate_img)
                if text and conf > OCR_CONF:
                    plate_votes[tid].append(text)
                    if len(plate_votes[tid]) > VOTE_FRAMES:
                        plate_votes[tid] = plate_votes[tid][-VOTE_FRAMES:]
                    # Pick the most common OCR result (voting system)
                    voted = Counter(plate_votes[tid]).most_common(1)
                    if voted:
                        best_plates[tid] = voted[0][0]
                        csv_rows.append({
                            'frame'      : frame_count,
                            'time_sec'   : round(frame_count / FPS, 2),
                            'vehicle_id' : tid,
                            'plate_text' : voted[0][0],
                            'confidence' : round(conf, 3)
                        })

            # Draw best plate text below the box
            if tid in best_plates:
                draw_text(frame, best_plates[tid], (x1, y2+22), color)

    # Draw dashboard overlay
    draw_dashboard(frame, frame_count, len(seen_ids), len(best_plates))
    out.write(frame)

    if frame_count % 30 == 0:
        elapsed  = time.time() - start_time
        fps_proc = frame_count / elapsed
        print(f"  Frame {frame_count}/{TOTAL} | FPS: {fps_proc:.1f} | Vehicles: {len(seen_ids)} | Plates: {len(best_plates)}")

cap.release()
out.release()

# ──────────────── Save CSV ────────────────
if csv_rows:
    csv_path = 'plates_log.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV saved: {len(csv_rows)} records")
    files.download(csv_path)

# ──────────────── Summary ────────────────
elapsed = time.time() - start_time
print(f"""
════════════════════════════════
Done!
  Total frames  : {frame_count}
  Total vehicles: {len(seen_ids)}
  Plates found  : {len(best_plates)}
  Time elapsed  : {elapsed:.1f}s
════════════════════════════════""")

# ──────────────── Display & Download Video ────────────────
mp4      = open('output_pro.mp4', 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
display(HTML(f"""
<video width='100%' controls>
  <source src='{data_url}' type='video/mp4'>
</video>
"""))
files.download('output_pro.mp4')

