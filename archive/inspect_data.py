import cv2
import os

video_path = "data/parking_1920_1080_loop.mp4"
mask_path = "data/mask_1920_1080.png"

# Check if files exist
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Missing video file at {video_path}")
if not os.path.exists(mask_path):
    raise FileNotFoundError(f"Missing mask file at {mask_path}")

# Inspect Video
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"--- Video Info ---")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")

# Extract first frame
ret, frame = cap.read()
if ret:
    cv2.imwrite("data/frame_0.png", frame)
    print("Successfully saved data/frame_0.png")

cap.release()