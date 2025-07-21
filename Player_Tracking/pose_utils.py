import numpy as np
import cv2

def compute_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    cosine = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def classify_shot(keypoints):
    try:
        # Right arm
        r_shoulder = keypoints[5]
        r_elbow = keypoints[7]
        r_wrist = keypoints[9]
        r_angle = compute_angle(r_shoulder, r_elbow, r_wrist)

        # Left arm
        l_shoulder = keypoints[6]
        l_elbow = keypoints[8]
        l_wrist = keypoints[10]
        l_angle = compute_angle(l_shoulder, l_elbow, l_wrist)

        # Average angles
        avg_angle = (r_angle + l_angle) / 2

        if avg_angle < 110:
            return "Forehand or Aggressive"
        elif avg_angle > 130:
            return "Backhand or Defensive"
        else:
            return "Neutral"
    except:
        return "Unknown"


def draw_corner_info_box(frame, player_id, pose, norm_cx, norm_cy, speed=None, top_left=(20, 20)):
    x, y = top_left
    w, h = 280, 140  # Extended height for speed display
    player_name = f"Player {chr(65 + player_id)}"

    # Draw background box and border
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 2)

    # Text lines inside the box
    cv2.putText(frame, player_name, (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Pose: {pose}", (x + 10, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"X Yolo: {norm_cx:.4f}, Y Yolo: {norm_cy:.4f}", (x + 10, y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    if speed is not None:
        cv2.putText(frame, f"Speed: {speed:.2f} px/s", (x + 10, y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    return frame
