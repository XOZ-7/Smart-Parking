from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import json
import numpy as np
import os
import time
import threading
from collections import deque
import occupancy_check

app = FastAPI(title="VISIONPARK Production Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REGISTRY_PATH = "data/roi_registry.json"
VIDEO_PATH = "data/cropped_parking_69.mp4"
if not os.path.exists(VIDEO_PATH):
    VIDEO_PATH = "data/parking_1920_1080_loop.mp4"

with open(REGISTRY_PATH, "r") as f:
    roi_registry = json.load(f)

# --- THREAD-SAFE SHARED STATE & SMOOTHING QUEUES ---
shared = {
    "raw_frame": None,
    "annotated_frame": None,
    "states": {sid: False for sid in roi_registry}
}
history = {sid: deque(maxlen=5) for sid in roi_registry}

# 1. Real-Time Video Reader Thread (30 FPS Playback)
def video_reader_thread():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    t0 = time.time()
    n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            t0 = time.time()
            n = 0
            continue

        n += 1
        shared["raw_frame"] = frame

        h_f, w_f = frame.shape[:2]
        display_frame = frame.copy()

        # Draw overlays using latest evaluated state
        for spot_id, info in roi_registry.items():
            pts = (np.array(info["polygon_norm"]) * [w_f, h_f]).astype(np.int32)
            is_occ = shared["states"].get(spot_id, False)
            color = (0, 0, 255) if is_occ else (0, 255, 0)
            cv2.polylines(display_frame, [pts], True, color, 2)
            cv2.putText(display_frame, spot_id, (pts[0][0] + 4, pts[0][1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        shared["annotated_frame"] = display_frame

        elapsed = time.time() - t0
        target = n / fps
        if target > elapsed:
            time.sleep(target - elapsed)

# 2. Asynchronous Occupancy Evaluator Thread (2 Hz)
def occupancy_evaluator_thread():
    while True:
        frame = shared["raw_frame"]
        if frame is not None:
            # Extract asphalt color reference & feature maps
            lab, ref = occupancy_check.asphalt_reference(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)

            for spot_id, info in roi_registry.items():
                fg_frac, edge_density = occupancy_check.slot_features(
                    frame, info["polygon_norm"], lab, ref, edges, gray
                )
                raw_occ = occupancy_check.classify(fg_frac, edge_density)

                # Temporal Majority Voting (3 out of 5 checks)
                history[spot_id].append(raw_occ)
                shared["states"][spot_id] = sum(history[spot_id]) >= 3

        time.sleep(0.3) # Runs twice per second

# Launch Threads
threading.Thread(target=video_reader_thread, daemon=True).start()
threading.Thread(target=occupancy_evaluator_thread, daemon=True).start()

@app.get("/")
def read_root():
    return FileResponse("dashboard.html") if os.path.exists("dashboard.html") else {"status": "ok"}

def stream_mjpeg():
    while True:
        frame = shared["annotated_frame"]
        if frame is not None:
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(stream_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            states = shared["states"]
            total = len(states)
            occ = sum(1 for v in states.values() if v)
            payload = {
                "total": total,
                "occupied": occ,
                "available": total - occ,
                "spots": {sid: {"occupied": is_occ, "spot_id": sid} for sid, is_occ in states.items()}
            }
            await websocket.send_json(payload)
            await time.sleep(0.1)
    except WebSocketDisconnect:
        pass