import cv2
import json
import numpy as np
import os

# 1. Load Frame
frame_path_png = "data/frame_0.png"
video_path = "data/parking_1920_1080_loop.mp4"

frame = None
if os.path.exists(frame_path_png):
    frame = cv2.imread(frame_path_png)
elif os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

if frame is None:
    raise FileNotFoundError("Could not find frame_0.png or parking_1920_1080_loop.mp4!")

# 2. Extract INDIVIDUAL slots from mask (RETR_LIST prevents merging adjacent slots)
mask_path = "data/mask_1920_1080.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Binary threshold
_, thresh_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# RETR_LIST gets all individual shapes without grouping parent/child columns
contours, _ = cv2.findContours(thresh_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

slots = []
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    
    # Strict bounding box filtering to isolate single parking spaces
    # Single slots in this 1080p mask are roughly 15x30 to 30x60 pixels
    if 200 < area < 3500 and w < 100 and h < 100:
        slots.append({
            "slot_id": f"A{len(slots)+1:03d}",
            "bbox": [int(x), int(y), int(w), int(h)]
        })

print(f"Successfully extracted {len(slots)} individual slots from mask!")

# Save clean geometry
with open("data/slots.json", "w") as f:
    json.dump(slots, f, indent=4)

# 3. Pre-process Frame for Occupancy Check
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 1)
thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
)
median = cv2.medianBlur(thresh, 5)
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(median, kernel, iterations=1)

# 4. Occupancy Check (Calibrated Threshold = 150 pixels)
occupied_count = 0
PIXEL_THRESHOLD = 150

for slot in slots:
    x, y, w, h = slot["bbox"]
    crop = dilated[y:y+h, x:x+w]
    count = cv2.countNonZero(crop)
    
    is_occupied = count > PIXEL_THRESHOLD
    slot["occupied"] = is_occupied
    
    if is_occupied:
        occupied_count += 1

print(f"--- Occupancy Summary ---")
print(f"Total Slots: {len(slots)}")
print(f"Occupied: {occupied_count}")
print(f"Free: {len(slots) - occupied_count}")

# 5. Save Visual
for slot in slots:
    x, y, w, h = slot["bbox"]
    color = (0, 0, 255) if slot["occupied"] else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

cv2.imwrite("data/occupancy_result_cv.png", frame)
print("Saved fixed visual to data/occupancy_result_cv.png")