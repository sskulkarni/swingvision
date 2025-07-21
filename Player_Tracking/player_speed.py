# player_speed.py

import math

class PlayerSpeedTracker:
    def __init__(self, fps):
        self.previous_position = None
        self.fps = fps
        self.time_per_frame = 1 / fps

    def update_position(self, current_position):
        """
        Update speed based on any provided (x, y) position.
        """
        if self.previous_position is None:
            self.previous_position = current_position
            return 0

        dx = current_position[0] - self.previous_position[0]
        dy = current_position[1] - self.previous_position[1]
        distance = math.sqrt(dx**2 + dy**2)
        speed = distance / self.time_per_frame

        self.previous_position = current_position
        return speed

    def update_position_from_keypoints(self, keypoints):
        """
        Use hip keypoints to compute body center and update speed.
        keypoints: list of (x, y) points from pose estimation.
        Assumes keypoints[11] = left hip, keypoints[12] = right hip.
        """
        try:
            hip_left = keypoints[11]
            hip_right = keypoints[12]
            center_x = (hip_left[0] + hip_right[0]) / 2
            center_y = (hip_left[1] + hip_right[1]) / 2
            center_point = (center_x, center_y)
            return self.update_position(center_point)
        except:
            # Fall back to zero speed if keypoints missing
            return 0
