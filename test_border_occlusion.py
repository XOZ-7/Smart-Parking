import cv2
import json
import numpy as np
import os

def check_border_occlusion(frame, quad_norm):
    h_f, w_f = frame.shape[:2]
    pts = (np.array(quad_norm) * [w_f, h_f]).astype(np.int32)

    # 1. Internal Border Ring Mask (3px thick)
    border_mask = np.zeros((h_f, w_f), dtype=np.uint8)
    cv2.polylines(border_mask, [pts], isClosed=True, color=255, thickness=3)

    # 2. Convert to HSV to differentiate white paint lines from metallic glare
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # White paint in HSV: Low saturation, high value
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 45, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 3. Combine with Canny Edge detector along borders
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Only look at edges that align with white paint along the border line
    paint_and_edges = cv2.bitwise_and(white_mask, edges)
    border_paint_pixels = cv2.bitwise_and(paint_and_edges, paint_and_edges, mask=border_mask)

    total_border_area = cv2.countNonZero(border_mask)
    visible_paint_area = cv2.countNonZero(border_paint_pixels)

    if total_border_area == 0:
        return False

    paint_ratio = (visible_paint_area / total_border_area) * 100.0

    # Vacant slot = Intact white boundary lines visible
    # Occupied slot = Vehicle body occludes border lines
    return paint_ratio < 2.5

img_path = "data/cropped_frame_0.png"
if not os.path.exists(img_path):
    img_path = "data/frame_0.png"

img = cv2.imread(img_path)
with open("data/roi_registry.json", "r") as f:
    roi_registry = json.load(f)

preview = img.copy()
occ_count = 0

for sid, info in roi_registry.items():
    is_occ = check_border_occlusion(img, info["polygon_norm"])
    if is_occ:
        occ_count += 1
    
    pts = (np.array(info["polygon_norm"]) * [img.shape[1], img.shape[0]]).astype(np.int32)
    color = (0, 0, 255) if is_occ else (0, 255, 0)
    cv2.polylines(preview, [pts], True, color, 2)
    cv2.putText(preview, sid, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

cv2.imwrite("data/border_occlusion_test.png", preview)
print(f"\n--- RECALIBRATED OCCLUSION RESULTS ---")
print(f"Total Slots: {len(roi_registry)} | Occupied: {occ_count} | Available: {len(roi_registry) - occ_count}")