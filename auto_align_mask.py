import cv2
import json
import numpy as np
import os

frame_path = "data/cropped_frame_0.png"
mask_path = "data/mask_1920_1080.png"

if not os.path.exists(frame_path) or not os.path.exists(mask_path):
    raise FileNotFoundError("Missing cropped_frame_0.png or mask_1920_1080.png in data/!")

frame = cv2.imread(frame_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
h_crop, w_crop = frame.shape[:2]

# Convert frame to grayscale for template matching
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Match cropped_frame_0.png template inside mask_1920_1080.png
# (Resizing gray_frame down slightly to match feature scale if needed)
res = cv2.matchTemplate(mask, cv2.resize(gray_frame, (w_crop, h_crop)), cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

crop_x, crop_y = max_loc
print(f"Auto-detected exact crop origin in full mask: X={crop_x}, Y={crop_y}")

# Slice the exact sub-mask region
aligned_mask = mask[crop_y:crop_y+h_crop, crop_x:crop_x+w_crop]

if aligned_mask.shape[1] != w_crop or aligned_mask.shape[0] != h_crop:
    aligned_mask = cv2.resize(aligned_mask, (w_crop, h_crop))

# Extract contours from perfectly aligned sub-mask
_, thresh = cv2.threshold(aligned_mask, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

slots = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if 200 < area < 4000 and 15 < w < 120 and 15 < h < 120:
        slots.append({"bbox": [int(x), int(y), int(w), int(h)]})

# Sort left-to-right (aisles), top-to-bottom (rows)
sorted_slots = sorted(slots, key=lambda s: (s["bbox"][0] // 100, s["bbox"][1]))[:69]

roi_registry = {}
for idx, s in enumerate(sorted_slots):
    spot_id = f"P{idx + 1}"
    roi_registry[spot_id] = {
        "spot_id": spot_id,
        "bbox": s["bbox"]
    }

with open("data/roi_registry.json", "w") as f:
    json.dump(roi_registry, f, indent=2)

# Save alignment check
for spot_id, info in roi_registry.items():
    x, y, w, h = info["bbox"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, spot_id, (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

cv2.imwrite("data/cropped_alignment_test.png", frame)
print("Saved aligned result to data/cropped_alignment_test.png!")