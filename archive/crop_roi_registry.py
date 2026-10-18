import json
import os

slots_path = "data/slots.json"
if not os.path.exists(slots_path):
    raise FileNotFoundError("Run detect_occupancy.py first to generate slots.json!")

with open(slots_path, "r") as f:
    raw_slots = json.load(f)

# Crop bounds (Leftmost 3 double-aisles)
CROP_X1, CROP_Y1 = 80, 20
CROP_X2, CROP_Y2 = 1100, 1050

filtered_slots = []
for s in raw_slots:
    x, y, w, h = s["bbox"]
    # Keep slots inside our crop area
    if CROP_X1 <= x <= CROP_X2 and CROP_Y1 <= y <= CROP_Y2:
        # Shift coordinates relative to the cropped frame origin (0,0)
        shifted_bbox = [x - CROP_X1, y - CROP_Y1, w, h]
        filtered_slots.append({
            "bbox": shifted_bbox,
            "orig_x": x,
            "orig_y": y
        })

# Sort spatially: Column by column, top to bottom
filtered_slots = sorted(filtered_slots, key=lambda s: (s["orig_x"] // 140, s["orig_y"]))

# Limit strictly to 69 spots to match reference layout
filtered_slots = filtered_slots[:69]

roi_registry = {}
for idx, slot in enumerate(filtered_slots):
    spot_id = f"P{idx + 1}"
    roi_registry[spot_id] = {
        "spot_id": spot_id,
        "bbox": slot["bbox"],
        "zone": "ROI-A"
    }

with open("data/roi_registry.json", "w") as f:
    json.dump(roi_registry, f, indent=2)

print(f"Successfully aligned {len(roi_registry)} ROI slots (P1 -> P69)!")