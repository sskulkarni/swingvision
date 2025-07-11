import cv2
import csv
from ultralytics import YOLO
import sys

# Initialize model
model = YOLO("yolov8n.pt")  # You can change to yolov8s.pt for better accuracy

# Video path
video_path = "data/raw_videos/sample.mkv"
cap = cv2.VideoCapture(video_path)

# CSV output file
csv_file = open("outputs/centroids.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Object', 'X', 'Y'])

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    
    for box in results[0].boxes:
        # Get class ID and name
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Only keep 'person' and 'sports ball'
        if class_name not in ['person', 'sports ball', 'tennis racket']:
            continue

        #print(box)
        #sys.exit()

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        #print("%s %s" % (x2, y2))


        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Write to CSV
        csv_writer.writerow([frame_count, class_name, cx, cy])

    

    frame_count += 1

    # Optional: show progress every 30 frames
    if frame_count % 30 == 0:
        print(f"Processed frame {frame_count}")

cap.release()
csv_file.close()
print("✅ Centroid extraction complete. Output saved to outputs/centroids.csv")
