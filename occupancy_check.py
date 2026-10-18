"""
occupancy_check.py - per-slot occupancy from the slot quads in data/roi_registry.json.

Why not paint visibility: stall paint stays visible between parked cars, so an occupied slot
looks the same as an empty one along its border. Instead, look INSIDE the slot:
  * how much of it differs from bare asphalt (Lab distance to the lot's asphalt colour)
  * how much edge texture it contains (empty asphalt is smooth, cars are not)
"""
import json
import os

import cv2
import numpy as np


def asphalt_reference(img):
    """Median Lab colour of the open drive aisles (no stalls there). Edit the strips if your crop differs."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    strips = np.zeros((h, w), np.uint8)
    for x0, x1 in ((0.30, 0.385), (0.59, 0.67)):           # aisles between the stall blocks
        strips[int(0.11 * h):int(0.93 * h), int(x0 * w):int(x1 * w)] = 1
    return lab, np.median(lab[strips > 0], axis=0)


def slot_features(img, quad_norm, lab, ref, edges, gray):
    h, w = img.shape[:2]
    p = (np.array(quad_norm) * [w, h]).astype(np.float32)
    c = p.mean(0)
    q = p.copy()
    q[:, 0] = c[0] + (p[:, 0] - c[0]) * 0.70               # shrink 15% per side: ignore paint and neighbours
    q[:, 1] = c[1] + (p[:, 1] - c[1]) * 0.76
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [q.astype(np.int32)], 1)
    dE = np.linalg.norm(lab[m > 0] - ref, axis=1)
    return float((dE > 28).mean()), float((edges[m > 0] > 0).mean())


def classify(fg_frac, edge_density):
    return fg_frac > 0.50 or edge_density > 0.028          # True = occupied


def run(image_path, registry_path="data/roi_registry.json", out_path="data/occupancy_preview.png"):
    img = cv2.imread(image_path)
    reg = json.load(open(registry_path))
    lab, ref = asphalt_reference(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)

    h, w = img.shape[:2]
    prev, state = img.copy(), {}
    for sid, info in reg.items():
        fg, ed = slot_features(img, info["polygon_norm"], lab, ref, edges, gray)
        state[sid] = classify(fg, ed)
        pts = (np.array(info["polygon_norm"]) * [w, h]).astype(np.int32)
        cv2.polylines(prev, [pts], True, (0, 0, 255) if state[sid] else (0, 255, 0), 2)
        cv2.putText(prev, sid, (pts[0][0] + 4, pts[0][1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, prev)
    occ = sum(state.values())
    print(f"Total Slots: {len(state)} | Occupied: {occ} | Available: {len(state) - occ}")
    return state


if __name__ == "__main__":
    path = "data/cropped_frame_0.png" if os.path.exists("data/cropped_frame_0.png") else "data/frame_0.png"
    run(path)