import sys
import os
import numpy as np
import cv2
import csv
import torch
from ultralytics import YOLO
from sort import Sort

# Ensure model directory is in sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model import BallTrackerNet
from court_detector import CourtDetector
from pose_utils import classify_shot, draw_corner_info_box
from player_speed import PlayerSpeedTracker

PIXEL_TO_METER = 0.01

os.makedirs("VideoOutput", exist_ok=True)
os.makedirs("CSVOutput", exist_ok=True)

csv_file_path = "CSVOutput/player_tracking.csv"
csv_file = open(csv_file_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Frame", "Player_name", "Court_X(m)", "Court_Y(m)",
    "Speed(m/s)", "Shot_Type", "Ball_X", "Ball_Y"
])

video_in = "VideoInput/video_input6.mp4"
video_out = "VideoOutput/output_tracking2_with_ball6.mp4"

posemodel = YOLO('yolov8s-pose.pt')
cap = cv2.VideoCapture(video_in)
if not cap.isOpened():
    print("Failed to open input video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_size = (width, height)

court_detector = CourtDetector()
ret, first_frame = cap.read()
if not ret:
    print("Failed to read first frame for court detection.")
    exit()
court_detector.detect(first_frame)
H = court_detector.court_warp_matrix[-1]

court_mask = court_detector.get_warped_court()
court_area = np.argwhere(court_mask > 0)
court_polygon = cv2.convexHull(court_area[:, [1, 0]]) if len(court_area) > 0 else None

tracker = Sort()
speed_trackers = {}

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_out, fourcc, fps, output_size)
if not out.isOpened():
    print("Failed to open VideoWriter. Check codec and output path.")
    exit()

cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose Estimation", 800, 600)

frame_count = 1

# === BallTrackerNet Setup ===
TRACKNET_WEIGHTS = os.path.join(os.path.dirname(__file__), 'model', 'model_best.pt')
input_height, input_width = 360, 640
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ball_model = BallTrackerNet().to(device)
ball_model.load_state_dict(torch.load(TRACKNET_WEIGHTS, map_location=device))
ball_model.eval()
ball_frame_buffer = []

def preprocess_for_tracknet(frame):
    frame = cv2.resize(frame, (input_width, input_height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))  # CHW
    return frame

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

    # === Ball detection (TrackNet) ===
    frame_for_ball = preprocess_for_tracknet(frame)
    ball_xy = None
    ball_frame_buffer.append(frame_for_ball)
    if len(ball_frame_buffer) > 3:
        ball_frame_buffer.pop(0)
    if len(ball_frame_buffer) == 3:
        tracknet_input = np.concatenate(ball_frame_buffer, axis=0)[None, ...]
        tracknet_input = torch.from_numpy(tracknet_input).to(device)
        with torch.no_grad():
            ball_pred = ball_model(tracknet_input)
            heatmap = ball_pred[0, 1].reshape(input_height, input_width).cpu().numpy()
        ball_y, ball_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        draw_x = int(ball_x * width / input_width)
        draw_y = int(ball_y * height / input_height)
        ball_xy = (draw_x, draw_y)
        cv2.rectangle(frame, (draw_x-12, draw_y-12), (draw_x+12, draw_y+12), (0, 255, 255), 2)
        cv2.putText(frame, "Ball", (draw_x-10, draw_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # === Player detection and tracking ===
    results = posemodel.predict(source=frame, conf=0.05, verbose=False)
    detections = []
    keypoints_map = {}
    for r in results:
        keypoints_list = r.keypoints.xy
        bboxes = r.boxes.xyxy if r.boxes is not None else []
        scores = r.boxes.conf if r.boxes is not None else []
        for i, person in enumerate(keypoints_list):
            if i >= len(bboxes):
                continue
            x1, y1, x2, y2 = bboxes[i].cpu().numpy()
            score = float(scores[i].cpu().numpy())
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if is_point_in_court((cx, cy), court_polygon):
                detections.append([x1, y1, x2, y2, score])
                keypoints_map[(x1, y1, x2, y2)] = person.cpu().numpy()

    dets_np = np.array(detections) if len(detections) > 0 else np.empty((0,5))
    tracks = tracker.update(dets_np)

    tracked_players = []
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        min_dist = 1e6
        best_kps = None
        for bbox, kps in keypoints_map.items():
            dist = np.linalg.norm(np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]) - np.array([(x1+x2)/2, (y1+y2)/2]))
            if dist < min_dist:
                min_dist = dist
                best_kps = kps
        if best_kps is not None:
            tracked_players.append({'track_id': int(track_id), 'bbox': (x1,y1,x2,y2), 'cx': (x1+x2)/2, 'cy': (y1+y2)/2, 'keypoints': best_kps})

    tracked_players = sorted(tracked_players, key=lambda d: d['cx'])
    if tracked_players:
        # Only process Player A (the first in list)
        player = tracked_players[0]
        track_id = player['track_id']
        x1, y1, x2, y2 = map(int, player['bbox'])
        cx, cy = player['cx'], player['cy']
        keypoint_array = player['keypoints']

        player_name = "Player A"
        info_box_position = (20, 20)
        box_color = (255, 0, 0)

        cv2.putText(frame, player_name, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if track_id not in speed_trackers:
            speed_trackers[track_id] = PlayerSpeedTracker(fps)
        speed_tracker = speed_trackers[track_id]

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
            frame, 0, label, norm_cx, norm_cy,
            speed=speed_m_per_s, top_left=info_box_position
        )

        csv_writer.writerow([
            frame_count, player_name, f"{court_x_m:.2f}", f"{court_y_m:.2f}", f"{speed_m_per_s:.2f}", label,
            ball_xy[0] if ball_xy else '', ball_xy[1] if ball_xy else ''
        ])

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
