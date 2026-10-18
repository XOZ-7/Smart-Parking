import cv2
import json
import numpy as np
import os

def evaluate_slot_occupancy(frame, quad_norm):
    h_f, w_f = frame.shape[:2]
    pts = (np.array(quad_norm) * [w_f, h_f]).astype(np.int32)

    mask = np.zeros((h_f, w_f), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Standard Deviation of Color Intensity
    _, stddev = cv2.meanStdDev(gray, mask=mask)
    val_std = stddev[0][0]

    # 2. Canny Edge Pixel Density
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 160)
    slot_edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    quad_area = cv2.countNonZero(mask)
    if quad_area == 0:
        return False

    edge_density = (cv2.countNonZero(slot_edges) / quad_area) * 100.0

    # Calibrated thresholds: Vehicles require strong edge density AND high color variance
    return (val_std > 28.0) and (edge_density > 12.0)

img_path = "data/cropped_frame_0.png"
if not os.path.exists(img_path):
    img_path = "data/frame_0.png"

img = cv2.imread(img_path)
with open("data/roi_registry.json", "r") as f:
    roi_registry = json.load(f)

preview = img.copy()
occ_count = 0

for sid, info in roi_registry.items():
    is_occ = evaluate_slot_occupancy(img, info["polygon_norm"])
    if is_occ:
        occ_count += 1
    
    pts = (np.array(info["polygon_norm"]) * [img.shape[1], img.shape[0]]).astype(np.int32)
    color = (0, 0, 255) if is_occ else (0, 255, 0)
    cv2.polylines(preview, [pts], True, color, 2)
    cv2.putText(preview, sid, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

cv2.imwrite("data/crop_classifier_test.png", preview)
print(f"Evaluated {len(roi_registry)} slots! Occupied: {occ_count} | Vacant: {len(roi_registry) - occ_count}")