import cv2
import json
import os

frame_path = "data/cropped_frame_0.png"
if not os.path.exists(frame_path):
    raise FileNotFoundError("Missing data/cropped_frame_0.png!")

frame = cv2.imread(frame_path)
img_h, img_w = frame.shape[:2]

# --- ADJUST THESE 4 NUMBERS TO SHIFT THE ENTIRE GRID IN 1 SECOND ---
START_X = int(img_w * 0.05)   # Starting X position (5% from left edge)
START_Y = int(img_h * 0.03)   # Starting Y position (3% from top edge)
SLOT_W  = int(img_w * 0.11)   # Width of one parking space
SLOT_H  = int(img_h * 0.038)  # Height of one parking space

# Column X-Offsets (Column 1, Column 2, Column 3)
col_offsets = [
    START_X, 
    START_X + int(img_w * 0.12), 
    START_X + int(img_w * 0.38)
]

slots = []
spot_counter = 1

# Build 3 double-columns matching your reference UI layout
for col_x in col_offsets:
    for row in range(23):  # 23 slots per column section = 69 total
        x = col_x
        y = START_Y + int(row * (SLOT_H + 4))
        
        if y + SLOT_H < img_h:
            slots.append({
                "spot_id": f"P{spot_counter}",
                "bbox": [x, y, SLOT_W, SLOT_H]
            })
            spot_counter += 1

slots = slots[:69]

roi_registry = {s["spot_id"]: s for s in slots}

os.makedirs("data", exist_ok=True)
with open("data/roi_registry.json", "w") as f:
    json.dump(roi_registry, f, indent=2)

# Save test visual
for spot_id, info in roi_registry.items():
    x, y, w, h = info["bbox"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, spot_id, (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

cv2.imwrite("data/cropped_alignment_test.png", frame)
print(f"Generated {len(roi_registry)} grid slots!")
print("Open data/cropped_alignment_test.png to check!")