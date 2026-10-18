import cv2
import numpy as np
import json
import os

def generate_precision_grid(image_path="data/cropped_frame_0.png", output_path="data/roi_registry.json"):
    if not os.path.exists(image_path):
        image_path = "data/frame_0.png"

    if not os.path.exists(image_path):
        video_path = "data/cropped_parking_69.mp4"
        if not os.path.exists(video_path):
            video_path = "data/parking_1920_1080_loop.mp4"
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            image_path = "data/cropped_frame_0.png"
            os.makedirs("data", exist_ok=True)
            cv2.imwrite(image_path, frame)
        else:
            raise FileNotFoundError("Could not find cropped_frame_0.png or any video source!")

    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]

    # Exactly calibrated X-bounds to lock ONTO white lines and stay OFF the sidewalk
    col_x_bounds = [
        (0.095, 0.185),  # Column 1 (Far Left)
        (0.195, 0.285),  # Column 2 (Left Inner)
        (0.380, 0.470),  # Column 3 (Mid Left)
        (0.480, 0.570),  # Column 4 (Mid Right)
        (0.670, 0.760),  # Column 5 (Right Inner)
        (0.765, 0.855)   # Column 6 (Right Outer - Shifted left off sidewalk!)
    ]

    auto_registry = {}
    spot_counter = 1
    stalls_per_col = 11

    # Y-bounds avoiding curb islands at top and bottom
    y_start_ratio = 0.05
    y_end_ratio = 0.93

    for col_idx, (x_left, x_right) in enumerate(col_x_bounds):
        for row in range(stalls_per_col):
            y_top = y_start_ratio + (row / stalls_per_col) * (y_end_ratio - y_start_ratio)
            y_bot = y_start_ratio + ((row + 1) / stalls_per_col) * (y_end_ratio - y_start_ratio)

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

    print(f"Shifted {len(auto_registry)} slots off sidewalk and onto white stall lines!")

if __name__ == "__main__":
    generate_precision_grid()