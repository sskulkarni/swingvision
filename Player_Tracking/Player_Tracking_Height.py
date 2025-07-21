# Player_Tracking.py

from ultralytics import YOLO
import cv2
import os
import sys
import csv
import numpy as np
from pose_utils import classify_shot, draw_corner_info_box
from player_speed import PlayerSpeedTracker
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# ////
# Output directory
os.makedirs("VideoOutput", exist_ok=True)

# Load model and video
model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture("VideoInput/video_input4.mp4")

# Save Co-oridinates in the file
csv_file = open("VideoOutput/tracked_centroids.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'ID', 'Object', 'X', 'Y'])

# save Graph with player co-orfinates
plt.figure(figsize=(10, 6))
plt.title("Player Trajectories")
plt.xlabel("X Position")
plt.ylabel("Y Position")

# frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print(frameCount)
# sys.exit()



if not cap.isOpened():
    print("Failed to open input video.")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_size = (width, height)

# Initialize player speed trackers
playerA_speed_tracker = PlayerSpeedTracker(fps)
playerB_speed_tracker = PlayerSpeedTracker(fps)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("VideoOutput/output.mp4", fourcc, fps, output_size)

# Display window
cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose Estimation", 800, 600)

# Define court area (customize as needed)
court_top = int(height * 0.05)
court_bottom = int(height * 0.95)
court_left = int(width * 0.10)
court_right = int(width * 0.90)


frame_count = 0

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.15, verbose=False)

    


    r = results[0]


    #for r in results:


    # loop

    keypoints_list = r.keypoints.xy
    bboxes = r.boxes.xyxy if r.boxes is not None else []

    # print(len(bboxes))
    # sys.exit()

    locked_player_index = None  # To track only one player

    #print(keypoints_list)
    #print(bboxes)
    #sys.exit()

    for i, person in enumerate(keypoints_list):


        
        if i >= len(bboxes):
            continue

        keypoint_array = person.cpu().numpy()
        x1, y1, x2, y2 = bboxes[i].cpu().numpy()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Assign Player A or Player B based on horizontal position
        # if cx < width / 2:
        #     player_id = 0  # Player A
        #     player_name = "Player A"
        #     info_box_position = (20, 20)
        #     speed_tracker = playerA_speed_tracker
        # else:
        #     player_id = 1  # Player B
        #     player_name = "Player B"
        #     info_box_position = (width - 300, 20)
        #     speed_tracker = playerB_speed_tracker



        # Lock onto a player once
        if locked_player_index is None:
            # if cy > (height / 2):  # Player on bottom half of the court

            if cy > (height / 2) and (y2 - y1) > 150:
                locked_player_index = i  # Lock onto this person
            else:
                continue
    
        if i != locked_player_index:
            continue  # Only track the locked player



        # Now everything below is for the locked player only
        player_id = 0
        player_name = "Tracked Player"
        info_box_position = (20, 20)
        speed_tracker = playerA_speed_tracker



        # Pose classification
        label = classify_shot(keypoint_array)

        # Draw keypoints
        for x, y in keypoint_array:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        # Draw Player A / Player B label above head
        cv2.putText(frame, player_name, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Update and display player speed
        norm_cx = cx / width
        norm_cy = cy / height
        current_position = (cx, cy)
        speed = speed_tracker.update_position(current_position)

        # Draw info box for each player
        frame = draw_corner_info_box(frame, player_id, label, norm_cx, norm_cy, speed=speed, top_left=info_box_position)


        csv_writer.writerow([frame_count, int(player_id), player_name, norm_cx, norm_cy])

    # loop

    # Write and display
    out.write(frame)
    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Finalize
cap.release()
out.release()
cv2.destroyAllWindows()




df = pd.read_csv("VideoOutput/tracked_centroids.csv")
players = df[df['Object'] == 'Player B']
for pid in players['ID'].unique():
    data = players[players['ID'] == pid]
    plt.plot(data['X'], data['Y'], label=f"Player {pid}")




plt.gca().invert_yaxis()
plt.legend()
plt.grid(True)
#plt.tight_layout()
plt.savefig("VideoOutput/trajectory_plot.png")
plt.show()




print("Final video saved to: VideoOutput/output.mp4")
