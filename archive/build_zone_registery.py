import json
import os

slots_path = "data/slots.json"
registry_path = "data/parking_registry.json"

if not os.path.exists(slots_path):
    raise FileNotFoundError("Run detect_occupancy.py first to generate data/slots.json!")

with open(slots_path, "r") as f:
    raw_slots = json.load(f)

# Sort slots spatially by X coordinate (Columns) then Y coordinate (Rows)
sorted_slots = sorted(raw_slots, key=lambda s: (s["bbox"][0], s["bbox"][1]))

# Divide into 4 geographic zones across the lot grid
num_slots = len(sorted_slots)
zone_size = num_slots // 4

registry = {}
zones = ["Zone-A", "Zone-B", "Zone-C", "Zone-D"]

for idx, slot in enumerate(sorted_slots):
    zone_idx = min(idx // zone_size, 3)
    zone_name = zones[zone_idx]
    
    # Generate structured slot ID (e.g., A-001, B-012)
    slot_num = (idx % zone_size) + 1
    structured_id = f"{zone_name[5]}-{slot_num:03d}"
    
    x, y, w, h = slot["bbox"]
    polygon = [
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ]

    registry[structured_id] = {
        "slot_id": structured_id,
        "raw_id": slot["slot_id"],
        "zone": zone_name,
        "bbox": [x, y, w, h],
        "polygon": polygon,
        "camera_id": "CAM_01"
    }

with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)

print(f"Successfully generated Parking Registry for {len(registry)} slots across 4 zones!")