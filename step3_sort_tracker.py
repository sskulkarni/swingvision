import cv2
import csv
from ultralytics import YOLO
from sort import Sort  # Make sure sort.py is in the same directory
import sys

# Initialize model and tracker
model = YOLO("yolov8n.pt")
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

cap = cv2.VideoCapture("data/raw_videos/sample.mkv")

csv_file = open("outputs/tracked_centroids_player_1_auto.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'ID', 'Object', 'X', 'Y'])

frame_count = 0
locked_player_id = None  # ID of the one player you want to track

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = []
    results = model(frame)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name not in ['person', 'sports ball', 'tennis racket']:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            detections.append([x1, y1, x2, y2, score])



    # Convert to NumPy for SORT
    import numpy as np
    dets_np = np.array(detections)

    


    # Update tracker
    tracks = tracker.update(dets_np)

    #print(tracks)
    #sys.exit()



    for track in tracks:
        x1, y1, x2, y2, track_id = track
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Optionally: check if this track came from a person or ball (approximate)
        import torch

        obj_type = 'unknown'
        best_iou = 0.0

        for box in results[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0]
            iou_x1 = max(x1, bx1)
            iou_y1 = max(y1, by1)
            iou_x2 = min(x2, bx2)
            iou_y2 = min(y2, by2)

            inter_area = max(0, iou_x2 - iou_x1) * max(0, iou_y2 - iou_y1)
            box_area = (x2 - x1) * (y2 - y1)
            det_area = (bx2 - bx1) * (by2 - by1)
            union_area = box_area + det_area - inter_area

            iou = inter_area / union_area if union_area != 0 else 0

            if iou > best_iou and iou > 0.2:
                best_iou = iou
                obj_type = model.names[int(box.cls[0])]

        if track_id == 1:
            csv_writer.writerow([frame_count, int(track_id), obj_type, cx, cy])
        

        # Only lock person
        if obj_type == "person":
            if locked_player_id is None:
                # Lock player with the largest Y (closer to bottom of screen)
                if cy > 400:  # Adjust threshold based on your video resolution
                    locked_player_id = int(track_id)
            
            # Track only locked player
            if int(track_id) == locked_player_id:
                # Do something with this one player's track
                csv_writer.writerow([frame_count, int(track_id), obj_type, cx, cy])


    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed frame {frame_count}")

cap.release()
csv_file.close()
print("✅ Tracked centroids saved in outputs/tracked_centroids.csv")
