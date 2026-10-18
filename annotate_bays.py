import cv2
import numpy as np
import json
import os

frame_path = "data/cropped_frame_0.png"
if not os.path.exists(frame_path):
    frame_path = "data/frame_0.png"

frame = cv2.imread(frame_path)
H_img, W_img = frame.shape[:2]
clone = frame.copy()

current_corners = []
all_bays = {}

def bay_slots(corners, n_slots):
    src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    dst = np.float32(corners)
    H_mat = cv2.getPerspectiveTransform(src, dst)
    
    quads = []
    for i in range(n_slots):
        y0, y1 = i / n_slots, (i + 1) / n_slots
        quad = np.float32([[[0, y0], [1, y0], [1, y1], [0, y1]]])
        warped_quad = cv2.perspectiveTransform(quad, H_mat)[0]
        quads.append(warped_quad)
    return quads

def click_event(event, x, y, flags, param):
    global current_corners, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        current_corners.append((x, y))
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        if len(current_corners) > 1:
            cv2.line(clone, current_corners[-2], current_corners[-1], (0, 255, 255), 2)
        if len(current_corners) == 4:
            cv2.line(clone, current_corners[3], current_corners[0], (0, 255, 255), 2)
        cv2.imshow("Annotator", clone)

cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Annotator", click_event)

print("\n--- INSTRUCTIONS ---")
print("Click 4 corners of each bay column: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
print("Enter the EXACT stall count (e.g. 13 stalls per column). Press 's' to Save.\n")

bay_idx = 1
while True:
    cv2.imshow("Annotator", clone)
    key = cv2.waitKey(1) & 0xFF

    if len(current_corners) == 4:
        bay_name = f"Bay_{bay_idx}"
        n_stalls = int(input(f"Enter exact number of stalls for {bay_name}: "))
        
        quads = bay_slots(current_corners, n_stalls)
        all_bays[bay_name] = [(q / [W_img, H_img]).tolist() for q in quads]
        
        for q in quads:
            pts = q.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(clone, [pts], True, (0, 255, 0), 2)
        
        current_corners = []
        bay_idx += 1

    elif key == ord('s'):
        flat_registry = {}
        global_id = 1
        for b_name, quad_list in all_bays.items():
            for q in quad_list:
                spot_id = f"P{global_id}"
                flat_registry[spot_id] = {"spot_id": spot_id, "polygon_norm": q}
                global_id += 1

        os.makedirs("data", exist_ok=True)
        with open("data/roi_registry.json", "w") as f:
            json.dump(flat_registry, f, indent=2)
        print(f"\nSaved {len(flat_registry)} quads to data/roi_registry.json!")
        break

cv2.destroyAllWindows()