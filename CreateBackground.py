import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture('VideoInput/sample.mkv')

# Read all frames into a list
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

# Convert list to NumPy array
frames_np = np.stack(frames, axis=0)

# Compute median across time axis
background = np.median(frames_np, axis=0).astype(np.uint8)

# Save the result
cv2.imwrite('VideoOutput/background_image.jpg', background)
