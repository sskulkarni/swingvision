import cv2
import pandas as pd
import sys

# Paths
video_path = 'data/raw_videos/sample.mkv'
csv_path = 'outputs/tracked_centroids_all.csv'
output_path = 'outputs/player_trajectory_overlay.mkv'

# Load trajectory data
df = pd.read_csv(csv_path)
# Filter only for player (adjust if needed)
player_df = df[df['Object'] == 'person']

# Open the video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_index = 0
trajectory = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get player coords for this frame
    rows = player_df[player_df['Frame'] == frame_index]


    for _, row in rows.iterrows():
        x, y = int(row['X']), int(row['Y'])
        trajectory.append((x, y))
        # Draw current point
        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

        coord_text = f"X2: {x}, Y2: {y}"
        #cv2.putText(frame, coord_text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


    # Draw all previous trajectory points
    for point in trajectory:

        coord_text = f"X2: {point[0]}, Y2: {point[1]}"

        #cv2.putText(frame, coord_text, (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 2)

        cv2.circle(frame, point, 2, (0, 255, 255), -1)





    out.write(frame)
    frame_index += 1

cap.release()
out.release()
print("✅ Trajectory overlay video saved at:", output_path)
