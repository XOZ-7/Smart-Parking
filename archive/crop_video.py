import cv2
import os

input_video = "data/parking_1920_1080_loop.mp4"
output_video = "data/cropped_parking_69.mp4"

if not os.path.exists(input_video):
    raise FileNotFoundError(f"Missing {input_video}!")

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Define crop coordinates [Y1:Y2, X1:X2] for the 69-slot region
# Adjust these 4 numbers if you want to frame the vehicles tighter/looser
Y1, Y2 = 30, 1020
X1, X2 = 80, 950

crop_w = X2 - X1
crop_h = Y2 - Y1

out = cv2.VideoWriter(output_video, fourcc, fps, (crop_w, crop_h))

print(f"Cropping video stream to {crop_w}x{crop_h} resolution...")

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Slice frame directly
    cropped_frame = frame[Y1:Y2, X1:X2]
    out.write(cropped_frame)
    count += 1

cap.release()
out.release()

# Save a single reference frame from the new video for geometry extraction
cv2.imwrite("data/cropped_frame_0.png", cv2.imread(input_video)[Y1:Y2, X1:X2])

print(f"Done! Saved {count} frames to {output_video}")
print("Saved reference frame to data/cropped_frame_0.png")