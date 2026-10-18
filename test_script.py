"""
import cv2
import json
import numpy as np

frame = cv2.imread("data/frame_0.png")
with open("data/slots.json", "r") as f:
    slots = json.load(f)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 1)
thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
)
median = cv2.medianBlur(thresh, 5)
dilated = cv2.dilate(median, np.ones((3, 3), np.uint8), iterations=1)

# Inspect pixel counts
counts = []
for slot in slots:
    x, y, w, h = slot["bbox"]
    crop = dilated[y:y+h, x:x+w]
    counts.append(cv2.countNonZero(crop))

print(f"Min pixel count: {min(counts)}")
print(f"Max pixel count: {max(counts)}")
print(f"Average pixel count: {sum(counts) / len(counts)}")
"""
import cv2

# Load your manually cropped video
cap = cv2.VideoCapture("data/cropped_parking_69.mp4")

# Read the very first frame
ret, frame = cap.read()

if ret:
    cv2.imwrite("data/cropped_frame_0.png", frame)
    print("Success! Saved first frame to data/cropped_frame_0.png")
else:
    print("Error: Could not read data/cropped_parking_69.mp4. Check the file name and folder path!")

cap.release()