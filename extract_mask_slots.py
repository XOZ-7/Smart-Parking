import cv2
import json

mask = cv2.imread("data/mask_1920_1080.png", cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

slots = []
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    # Filter out tiny noise artifacts
    if w * h > 150:
        slots.append({
            "slot_id": f"A{i+1:03d}",
            "bbox": [int(x), int(y), int(w), int(h)]
        })

print(f"Extracted {len(slots)} clean parking slot rectangles.")
with open("data/slots.json", "w") as f:
    json.dump(slots, f, indent=4)