import sys
import os
import numpy as np
import cv2
import csv
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from court_detector import CourtDetector
from pose_utils import classify_shot, draw_corner_info_box
from player_speed import PlayerSpeedTracker

PIXEL_TO_METER = 0.01

os.makedirs("../VideoOutput", exist_ok=True)
os.makedirs("../CSVOutput", exist_ok=True)

csv_file_path = "CSVOutput/player_tracking.csv"
csv_file = open(csv_file_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Player", "Court_X(m)", "Court_Y(m)", "Speed(m/s)", "Shot_Type"])

video_in = "VideoInput/video9.mp4"
video_out = "VideoOutput/output_court_tracking9.mp4"

model = YOLO('yolov8s-pose.pt')
cap = cv2.VideoCapture(video_in)

if not cap.isOpened():
    print("Failed to open input video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_size = (width, height)

playerA_speed_tracker = PlayerSpeedTracker(fps)
playerB_speed_tracker = PlayerSpeedTracker(fps)

court_detector = CourtDetector()
ret, first_frame = cap.read()
if not ret:
    print("Failed to read first frame for court detection.")
    exit()
court_detector.detect(first_frame)
H = court_detector.court_warp_matrix[-1]

# For filtering, get a mask of the court area
court_mask = court_detector.get_warped_court()
court_area = np.argwhere(court_mask > 0)
court_polygon = cv2.convexHull(court_area[:, [1, 0]]) if len(court_area) > 0 else None

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_out, fourcc, fps, output_size)
if not out.isOpened():
    print("Failed to open VideoWriter. Check codec and output path.")
    exit()

cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose Estimation", 800, 600)

frame_count = 1

def warp_point(point, H):
    pt = np.array([point[0], point[1], 1.0]).reshape(3, 1)
    dst = np.dot(H, pt)
    dst /= dst[2]
    return (dst[0][0], dst[1][0])

def is_point_in_court(point, court_polygon):
    if court_polygon is None:
        return False
    return cv2.pointPolygonTest(court_polygon, (int(point[0]), int(point[1])), False) >= 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = court_detector.add_court_overlay(frame, homography=H, overlay_color=(0, 0, 255))

    results = model.predict(source=frame, conf=0.05, verbose=False)
    for r in results:
        keypoints_list = r.keypoints.xy
        bboxes = r.boxes.xyxy if r.boxes is not None else []

        # Filter detections whose bbox center is inside court area
        candidates = []
        for i, person in enumerate(keypoints_list):
            if i >= len(bboxes):
                continue
            x1, y1, x2, y2 = bboxes[i].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if is_point_in_court((cx, cy), court_polygon):
                area = (x2 - x1) * (y2 - y1)
                candidates.append({
                    'idx': i,
                    'area': area,
                    'cx': cx,
                    'cy': cy,
                    'bbox': (x1, y1, x2, y2),
                    'keypoints': person.cpu().numpy()
                })

        # Pick up to 2 largest area bboxes in court
        candidates = sorted(candidates, key=lambda d: d['area'], reverse=True)[:2]
        # Assign left/right based on cx
        candidates = sorted(candidates, key=lambda d: d['cx'])

        for player_id, candidate in enumerate(candidates):
            if player_id == 0:
                player_name = "Player A"
                info_box_position = (20, 20)
                speed_tracker = playerA_speed_tracker
                box_color = (255, 0, 0)
            else:
                player_name = "Player B"
                info_box_position = (width - 320, 20)
                speed_tracker = playerB_speed_tracker
                box_color = (0, 255, 255)

            keypoint_array = candidate['keypoints']
            x1, y1, x2, y2 = candidate['bbox']
            cx, cy = candidate['cx'], candidate['cy']

            label = classify_shot(keypoint_array)
            for x, y in keypoint_array:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.putText(frame, player_name, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            norm_cx = cx / width
            norm_cy = cy / height
            current_position = (cx, cy)
            speed_px_per_s = speed_tracker.update_position(current_position)
            speed_m_per_s = speed_px_per_s * PIXEL_TO_METER

            court_x, court_y = warp_point((cx, cy), H)
            court_x_m = court_x * PIXEL_TO_METER
            court_y_m = court_y * PIXEL_TO_METER

            frame = draw_corner_info_box(
                frame, player_id, label, norm_cx, norm_cy,
                speed=speed_m_per_s, top_left=info_box_position
            )

            csv_writer.writerow([frame_count, player_name, f"{court_x_m:.2f}", f"{court_y_m:.2f}", f"{speed_m_per_s:.2f}", label])

    out.write(frame)
    cv2.imshow("Pose Estimation", frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

if os.path.exists(video_out) and os.path.getsize(video_out) > 1000:
    print("Final video saved to:", video_out)
else:
    print("ERROR: Output video file was not written (file missing or empty). Try a different codec or check output path.")

print("CSV log saved to:", csv_file_path)
