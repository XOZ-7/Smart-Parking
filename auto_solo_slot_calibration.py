import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

def calibrate_slots_from_yolo(image_path="data/cropped_frame_0.png", output_path="data/roi_registry.json"):
    # 1. Fallback to extracting frame 0 from cropped MP4 if PNG missing
    if not os.path.exists(image_path):
        video_path = "data/cropped_parking_69.mp4"
        if not os.path.exists(video_path):
            video_path = "data/parking_1920_1080_loop.mp4"
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            image_path = "data/cropped_frame_0.png"
            cv2.imwrite(image_path, frame)
        else:
            raise FileNotFoundError("Could not find cropped_frame_0.png or video source!")

    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]

    # 2. Run YOLO to detect vehicle ground contact points (lower confidence for maximum detection)
    print("Scanning frame for vehicles with YOLOv8...")
    model = YOLO("yolov8n.pt")
    results = model(img, verbose=False, conf=0.10)[0] # COCO classes: car, motorcycle, bus, truck

    centers_x = []
    for box in results.boxes:
        cls_id = int(box.cls[0].cpu().numpy())
        if cls_id in [2, 3, 5, 7]: # Vehicle classes
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            centers_x.append((x1 + x2) / 2.0)

    # 3. K-Means clustering or Geometric 6-Column Spatial Partition
    if len(centers_x) >= 6:
        print(f"Detected {len(centers_x)} vehicles! Clustering into 6 single-slot columns...")
        data = np.float32(centers_x).reshape(-1, 1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(data, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        col_centers = sorted([c[0] for c in centers])
    else:
        print("Fallback: Using uniform spatial partitioning for 6 single-slot columns...")
        col_centers = [w_img * pct for pct in [0.08, 0.22, 0.40, 0.54, 0.72, 0.86]]

    # 4. Calculate column widths dynamically based on distance between aisles
    half_widths = []
    for i in range(len(col_centers)):
        if i == 0:
            hw = (col_centers[1] - col_centers[0]) * 0.40
        elif i % 2 == 1:  # Right side of a double aisle
            hw = (col_centers[i] - col_centers[i-1]) * 0.40
        else:  # Left side of a double aisle
            hw = (col_centers[i+1] - col_centers[i]) * 0.40
        half_widths.append(hw)

    # 5. Generate 6 distinct single-slot columns (11-12 rows each)
    auto_registry = {}
    spot_counter = 1
    stalls_per_col = 11

    # Y-bounds avoiding islands/curbs (Y: 5% to 92%)
    y_start, y_end = h_img * 0.05, h_img * 0.92

    for col_idx, center_x in enumerate(col_centers):
        hw = half_widths[col_idx]
        x_left = max(0.0, (center_x - hw) / w_img)
        x_right = min(1.0, (center_x + hw) / w_img)

        for row in range(stalls_per_col):
            y_top = (y_start + (row / stalls_per_col) * (y_end - y_start)) / h_img
            y_bot = (y_start + ((row + 1) / stalls_per_col) * (y_end - y_start)) / h_img

            quad_norm = [
                [x_left, y_top],
                [x_right, y_top],
                [x_right, y_bot],
                [x_left, y_bot]
            ]

            spot_id = f"P{spot_counter}"
            auto_registry[spot_id] = {
                "spot_id": spot_id,
                "polygon_norm": quad_norm
            }
            spot_counter += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(auto_registry, f, indent=2)

    # Preview verification image
    preview = img.copy()
    for sid, info in auto_registry.items():
        pts = (np.array(info["polygon_norm"]) * [w_img, h_img]).astype(np.int32)
        cv2.polylines(preview, [pts], True, (0, 255, 0), 2)
        cv2.putText(preview, sid, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    cv2.imwrite("data/auto_yolo_calibration_test.png", preview)
    print(f"Successfully generated {len(auto_registry)} single-slot quads!")
    print("Check data/auto_yolo_calibration_test.png!")

if __name__ == "__main__":
    calibrate_slots_from_yolo()