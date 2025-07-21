import numpy as np
import cv2
from sympy import Line
from itertools import combinations
from court_reference import CourtReference

class CourtDetector:
    """
    Detect and overlay tennis court lines on video frames using a template and homography.
    """
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.colour_threshold = 200
        self.dist_tau = 3
        self.intensity_threshold = 40
        self.court_reference = CourtReference()
        self.v_width = 0
        self.v_height = 0
        self.frame = None
        self.gray = None
        self.court_warp_matrix = []
        self.game_warp_matrix = []
        self.court_score = 0
        self.success_flag = False
        self.best_conf = None
        self.frame_points = None
        self.dist = 5

    def detect(self, frame, verbose=0):
        """
        Detect the court in the frame and estimate homography.
        """
        self.verbose = verbose
        self.frame = frame
        self.v_height, self.v_width = frame.shape[:2]
        # 1. Threshold to binary court lines
        self.gray = self._threshold(frame)
        # 2. Filter out non-court pixels
        filtered = self._filter_pixels(self.gray)
        # 3. Detect lines with Hough transform
        horizontal_lines, vertical_lines = self._detect_lines(filtered)
        # 4. Find homography from template to frame
        court_warp_matrix, game_warp_matrix, self.court_score = self._find_homography(horizontal_lines, vertical_lines)
        self.court_warp_matrix.append(court_warp_matrix)
        self.game_warp_matrix.append(game_warp_matrix)
        return court_warp_matrix  # Return H for main code

    def _threshold(self, frame):
        """Simple thresholding for white pixels (court lines)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    def _filter_pixels(self, gray):
        """Remove noise, keep court lines only."""
        for i in range(self.dist_tau, len(gray) - self.dist_tau):
            for j in range(self.dist_tau, len(gray[0]) - self.dist_tau):
                if gray[i, j] == 0:
                    continue
                if (gray[i, j] - gray[i + self.dist_tau, j] > self.intensity_threshold and gray[i, j] - gray[i - self.dist_tau, j] > self.intensity_threshold):
                    continue
                if (gray[i, j] - gray[i, j + self.dist_tau] > self.intensity_threshold and gray[i, j] - gray[i, j - self.dist_tau] > self.intensity_threshold):
                    continue
                gray[i, j] = 0
        return gray

    def _detect_lines(self, gray):
        """Find horizontal and vertical court lines using Hough transform."""
        minLineLength = 100
        maxLineGap = 20
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80, minLineLength=minLineLength, maxLineGap=maxLineGap)
        if lines is None:
            return [], []
        lines = np.squeeze(lines)
        horizontal, vertical = self._classify_lines(lines)
        horizontal, vertical = self._merge_lines(horizontal, vertical)
        return horizontal, vertical

    def _classify_lines(self, lines):
        """Classify lines as horizontal or vertical by their slope."""
        horizontal = []
        vertical = []
        highest_vertical_y = np.inf
        lowest_vertical_y = 0
        for line in lines:
            x1, y1, x2, y2 = line
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            if dx > 2 * dy:
                horizontal.append(line)
            else:
                vertical.append(line)
                highest_vertical_y = min(highest_vertical_y, y1, y2)
                lowest_vertical_y = max(lowest_vertical_y, y1, y2)
        # Simple filtering by location (optional, can comment/remove)
        clean_horizontal = []
        h = lowest_vertical_y - highest_vertical_y
        lowest_vertical_y += h / 15
        highest_vertical_y -= h * 2 / 15
        for line in horizontal:
            x1, y1, x2, y2 = line
            if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
                clean_horizontal.append(line)
        return clean_horizontal, vertical

    def _merge_lines(self, horizontal_lines, vertical_lines):
        """Merge lines that belong to the same court line (for Hough output)."""
        # Merge horizontal lines
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
        mask = [True] * len(horizontal_lines)
        new_horizontal_lines = []
        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        dy = abs(y3 - y2)
                        if dy < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_horizontal_lines.append(line)
        # Merge vertical lines
        vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
        xl, yl, xr, yr = (0, self.v_height * 6 / 7, self.v_width, self.v_height * 6 / 7)
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []
        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        xi, yi = line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                        xj, yj = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))
                        dx = abs(xi - xj)
                        if dx < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_vertical_lines.append(line)
        return new_horizontal_lines, new_vertical_lines

    def _find_homography(self, horizontal_lines, vertical_lines):
        """
        Finds transformation from reference court to frame's court using 4 pairs of matching points.
        """
        max_score = -np.inf
        max_mat = None
        max_inv_mat = None
        for horizontal_pair in list(combinations(horizontal_lines, 2)):
            for vertical_pair in list(combinations(vertical_lines, 2)):
                h1, h2 = horizontal_pair
                v1, v2 = vertical_pair
                i1 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v1[:2]), tuple(v1[2:])))
                i2 = line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v2[:2]), tuple(v2[2:])))
                i3 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v1[:2]), tuple(v1[2:])))
                i4 = line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v2[:2]), tuple(v2[2:])))
                intersections = [i1, i2, i3, i4]
                intersections = sort_intersection_points(intersections)
                for i, configuration in self.court_reference.court_conf.items():
                    matrix, _ = cv2.findHomography(np.float32(configuration), np.float32(intersections), method=0)
                    inv_matrix = cv2.invert(matrix)[1]
                    confi_score = self._get_confi_score(matrix)
                    if max_score < confi_score:
                        max_score = confi_score
                        max_mat = matrix
                        max_inv_mat = inv_matrix
                        self.best_conf = i
        return max_mat, max_inv_mat, max_score

    def _get_confi_score(self, matrix):
        """Scoring function for homography."""
        court = cv2.warpPerspective(self.court_reference.court, matrix, self.frame.shape[1::-1])
        court[court > 0] = 1
        gray = self.gray.copy()
        gray[gray > 0] = 1
        correct = court * gray
        wrong = court - correct
        c_p = np.sum(correct)
        w_p = np.sum(wrong)
        return c_p - 0.5 * w_p

    def add_court_overlay(self, frame, homography=None, overlay_color=(0, 0, 255)):
        """
        Overlay the court template lines in the frame using the provided homography and color.
        """
        if homography is None and len(self.court_warp_matrix) > 0:
            homography = self.court_warp_matrix[-1]
        court_mask = cv2.warpPerspective(self.court_reference.court, homography, frame.shape[1::-1])
        frame[court_mask > 0] = overlay_color
        return frame

    def get_warped_court(self):
        """
        Returns a binary mask of the court region in the current frame.
        """
        if not self.court_warp_matrix or self.frame is None:
            return None
        court_mask = cv2.warpPerspective(
            self.court_reference.court,
            self.court_warp_matrix[-1],
            self.frame.shape[1::-1]
        )
        court_mask[court_mask > 0] = 255  # Make sure mask is binary (0 or 255)
        return court_mask

def line_intersection(line1, line2):
    """
    Find intersection point of two lines (as points).
    """
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])
    intersection = l1.intersection(l2)
    return intersection[0].coordinates

def sort_intersection_points(intersections):
    """
    Sort rectangle points from top-left to bottom-right.
    """
    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34
