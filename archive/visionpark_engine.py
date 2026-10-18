import cv2
import json
import sqlite3
import numpy as np
import os
from datetime import datetime, timedelta

# 1. Setup Paths & Load Registry
registry_path = "data/parking_registry.json"
video_path = "data/parking_1920_1080_loop.mp4"
db_path = "data/visionpark.db"
events_log_path = "data/parking_events.json"

if not os.path.exists(registry_path):
    raise FileNotFoundError("Run build_zone_registry.py first!")

with open(registry_path, "r") as f:
    registry = json.load(f)

# 2. Initialize SQLite Database for Live State & Event History
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Table 1: Live State Snapshot
cursor.execute("""
CREATE TABLE IF NOT EXISTS live_parking_state (
    slot_id TEXT PRIMARY KEY,
    zone TEXT,
    status TEXT,
    last_updated TEXT,
    confidence REAL
)
""")

# Table 2: Event Stream Log
cursor.execute("""
CREATE TABLE IF NOT EXISTS state_change_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    slot_id TEXT,
    zone TEXT,
    previous_state TEXT,
    new_state TEXT
)
""")
conn.commit()

# Populate initial DB snapshot as VACANT
for slot_id, info in registry.items():
    cursor.execute("""
    INSERT OR REPLACE INTO live_parking_state (slot_id, zone, status, last_updated, confidence)
    VALUES (?, ?, 'vacant', ?, 0.90)
    """, (slot_id, info["zone"], datetime.now().isoformat()))
conn.commit()

# 3. Process Video Stream & Detect State Changes
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Track previous state in memory
previous_slot_states = {slot_id: "vacant" for slot_id in registry}

PIXEL_THRESHOLD = 150
SAMPLE_INTERVAL_SEC = 1
frame_step = int(fps * SAMPLE_INTERVAL_SEC)

simulated_time = datetime.now()
event_history = []
frame_idx = 0

print(f"\n--- VISIONPARK Event Engine Online ---")
print(f"Monitoring {len(registry)} slots | Stream Resolution: 1920x1080\n")

while cap.isOpened():
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV Image Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
    )
    median = cv2.medianBlur(thresh, 5)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=1)

    current_timestamp = (simulated_time + timedelta(seconds=frame_idx / fps)).isoformat()
    state_changes_in_frame = 0

    occupied_count = 0

    for slot_id, info in registry.items():
        x, y, w, h = info["bbox"]
        crop = dilated[y:y+h, x:x+w]
        pixel_count = cv2.countNonZero(crop)
        
        current_status = "occupied" if pixel_count > PIXEL_THRESHOLD else "vacant"
        if current_status == "occupied":
            occupied_count += 1

        prev_status = previous_slot_states[slot_id]

        # DETECT STATE TRANSITION (Event Trigger)
        if current_status != prev_status:
            state_changes_in_frame += 1
            
            # Emit Event Payload
            event_payload = {
                "timestamp": current_timestamp,
                "slot_id": slot_id,
                "zone": info["zone"],
                "previous_state": prev_status,
                "new_state": current_status,
                "camera_id": info["camera_id"]
            }
            event_history.append(event_payload)

            # Insert into SQLite Database
            cursor.execute("""
            INSERT INTO state_change_events (timestamp, slot_id, zone, previous_state, new_state)
            VALUES (?, ?, ?, ?, ?)
            """, (current_timestamp, slot_id, info["zone"], prev_status, current_status))

            cursor.execute("""
            UPDATE live_parking_state
            SET status = ?, last_updated = ?
            WHERE slot_id = ?
            """, (current_status, current_timestamp, slot_id))

            # Update Memory Tracker
            previous_slot_states[slot_id] = current_status

    conn.commit()

    total_slots = len(registry)
    occupancy_rate = (occupied_count / total_slots) * 100

    print(f"[{current_timestamp[11:19]}] Frame {frame_idx:04d}/{total_frames} | "
          f"Occupied: {occupied_count:03d}/{total_slots} ({occupancy_rate:.1f}%) | "
          f"State Deltas: {state_changes_in_frame}")

    frame_idx += frame_step

cap.release()
conn.close()

# Save Event Log
with open(events_log_path, "w") as f:
    json.dump(event_history, f, indent=2)

print(f"\nEngine Execution Complete.")
print(f"Recorded {len(event_history)} state-change events into SQLite database & {events_log_path}!")