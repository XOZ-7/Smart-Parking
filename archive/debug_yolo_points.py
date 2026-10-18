import cv2
import json
import numpy as np
import os
from ultralytics import YOLO

# Load image and model
img_path = "data/cropped_frame_0.png"
if not os.path.exists(img_path):
    img_path = "data/frame_0.png"

img = cv2.imread(img_path)
h_img, w_img = img.shape[:2]

with open("data/roi_registry.json", "r") as f:
    roi_registry = json.load(f)

model = YOLO("yolov8n.pt")

# Run YOLO with ZERO class restriction to see all raw detections
results = model(img, verbose=True, conf=0.05)[0]

print(f"\n--- DEBUG RESULTS ---")
print(f"Image Resolution: {w_img}x{h_img}")
print(f"Total Raw Detections Found: {len(results.boxes)}")

preview = img.copy()

# Draw all raw bounding boxes and ground points
for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    cls_id = int(box.cls[0].cpu().numpy())
    conf = float(box.conf[0].cpu().numpy())

    # Calculate center point
    c_x, c_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

    # Draw box and center dot
    cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)
    cv2.circle(preview, (c_x, c_y), 3, (0, 0, 255), -1)

# Overlay slot quads
match_count = 0
for spot_id, info in roi_registry.items():
    pts = (np.array(info["polygon_norm"]) * [w_img, h_img]).astype(np.int32)
    cv2.polylines(preview, [pts], True, (0, 255, 0), 1)

cv2.imwrite("data/yolo_debug_output.png", preview)
print("Saved debug frame to data/yolo_debug_output.png")