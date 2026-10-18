import cv2
import json
import numpy as np
import os

# 1. Load cropped frame snapshot to get exact resolution
frame_path = "data/cropped_frame_0.png"
mask_path = "data/mask_1920_1080.png"

if not os.path.exists(frame_path):
    raise FileNotFoundError("Missing data/cropped_frame_0.png!")
if not os.path.exists(mask_path):
    raise FileNotFoundError("Missing data/mask_1920_1080.png!")

frame = cv2.imread(frame_path)
h_crop, w_crop = frame.shape[:2]

# 2. Load original mask and crop it matching the left 3 aisles
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Crop coordinates matching your video crop region
Y1, Y2 = 30, 1020
X1, X2 = 80, 950

cropped_mask = mask[Y1:Y2, X1:X2]

# Resize mask if there's any 1-2 pixel rounding difference
if cropped_mask.shape[1] != w_crop or cropped_mask.shape[0] != h_crop:
    cropped_mask = cv2.resize(cropped_mask, (w_crop, h_crop))

# 3. OpenCV contour extraction on the white slot mask
_, thresh = cv2.threshold(cropped_mask, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

raw_slots = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    # Filter for single slot sizes in this crop resolution
    if 250 < area < 4000 and w < 120 and h < 120:
        raw_slots.append({"bbox": [int(x), int(y), int(w), int(h)]})

# Sort spatially: column-by-column (left to right), then row-by-row (top to bottom)
sorted_slots = sorted(raw_slots, key=lambda s: (s["bbox"][0] // 110, s["bbox"][1]))

# Limit strictly to 69 spots
final_slots = sorted_slots[:69]

roi_registry = {}
for idx, s in enumerate(final_slots):
    spot_id = f"P{idx + 1}"
    roi_registry[spot_id] = {
        "spot_id": spot_id,
        "bbox": s["bbox"]
    }

os.makedirs("data", exist_ok=True)
with open("data/roi_registry.json", "w") as f:
    json.dump(roi_registry, f, indent=2)

# 4. Save visual check
for spot_id, info in roi_registry.items():
    x, y, w, h = info["bbox"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, spot_id, (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

cv2.imwrite("data/cropped_alignment_test.png", frame)
print(f"Successfully extracted {len(roi_registry)} slots (P1 -> P69) using OpenCV mask alignment!")