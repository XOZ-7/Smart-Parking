import cv2
import json
import numpy as np
import os
from datetime import datetime, timedelta

# 1. Paths & Setup
video_path = "data/parking_1920_1080_loop.mp4"
slots_path = "data/slots.json"
output_log_path = "data/parking_state_history.json"

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Missing video at {video_path}")
if not os.path.exists(slots_path):
    raise FileNotFoundError(f"Missing slot geometry at {slots_path}. Run detect_occupancy.py first!")

# Load pre-extracted slot geometry
with open(slots_path, "r") as f:
    slots = json.load(f)

# Open Video Stream
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"--- Video Processing Engine Started ---")
print(f"Total Video Frames: {total_frames} | FPS: {fps}")

# Process 1 frame every second of video time
SAMPLE_INTERVAL_SEC = 1
frame_step = int(fps * SAMPLE_INTERVAL_SEC)

PIXEL_THRESHOLD = 150
simulated_start_time = datetime.now()
history_log = []

frame_idx = 0

while cap.isOpened():
    # Set video position to next sample frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video stream

    # Pre-process Frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
    )
    median = cv2.medianBlur(thresh, 5)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=1)

    # Evaluate Occupancy for All Slots
    occupied_count = 0
    current_timestamp = (simulated_start_time + timedelta(seconds=frame_idx / fps)).isoformat()
    
    slot_states = []
    for slot in slots:
        x, y, w, h = slot["bbox"]
        crop = dilated[y:y+h, x:x+w]
        count = cv2.countNonZero(crop)
        is_occupied = count > PIXEL_THRESHOLD
        
        if is_occupied:
            occupied_count += 1

        slot_states.append({
            "slot_id": slot["slot_id"],
            "occupied": is_occupied,
            "bbox": slot["bbox"]
        })

    total_slots = len(slots)
    free_count = total_slots - occupied_count
    occupancy_rate = round((occupied_count / total_slots) * 100, 2)

    # Construct Temporal State Payload
    state_payload = {
        "timestamp": current_timestamp,
        "frame_index": frame_idx,
        "metrics": {
            "total_slots": total_slots,
            "occupied_slots": occupied_count,
            "free_slots": free_count,
            "occupancy_rate_percent": occupancy_rate
        },
        "slots": slot_states
    }

    history_log.append(state_payload)
    print(f"[{current_timestamp[:19]}] Frame {frame_idx}/{total_frames} -> Occupied: {occupied_count}/{total_slots} ({occupancy_rate}%)")

    frame_idx += frame_step

cap.release()

# 2. Save Temporal Dataset
with open(output_log_path, "w") as f:
    json.dump(history_log, f, indent=2)

print(f"\nSuccessfully logged {len(history_log)} temporal state frames to {output_log_path}!")