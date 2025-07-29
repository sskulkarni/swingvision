import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


# === Load tracking data ===
csv_path = 'CSVOutput/player_tracking.csv'
df = pd.read_csv(csv_path)
player_id_to_plot = "Player A"  # change if needed

df = df[df['Player_name'] == player_id_to_plot]


# === Load tennis court image ===
court_img = cv2.imread('CSVOutput/court.jpg')  # Make sure this is a full-court image
#court_img = cv2.resize(court_img, (1920, 1080))

court_img_rgb = cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)

court_img = cv2.convertScaleAbs(court_img_rgb, alpha=1.3, beta=30)  # alpha > 1 and beta > 0 for brightness




# === Get image size ===
img_height, img_width = court_img.shape[:2]

# === Normalize tracking coordinates to image size ===
x = df['X'].values
y = df['Y'].values


ball_x = df['Ball_X'].values
ball_y = df['Ball_Y'].values



# Normalize (rescale x, y into image width/height)
x_scaled = ((x - x.min()) / (x.max() - x.min())) * img_width
y_scaled = ((y - y.min()) / (y.max() - y.min())) * img_height
# Optional: Flip y to match image origin (if needed)
y_scaled = img_height - y_scaled


# === Create heatmap using 2D histogram ===
heatmap, xedges, yedges = np.histogram2d(x, y, bins=[64, 64], range=[[0, img_width], [0, img_height]])
heatmap = heatmap.T  # Transpose for correct orientation
heatmap = heatmap / np.max(heatmap)
heatmap = heatmap ** 0.5  # Optional enhancement
# === Plot heatmap on top of court image ===



# === Ball Create heatmap using 2D histogram ===
heatmap_ball, xedges_ball, yedges_ball = np.histogram2d(ball_x, ball_y, bins=[64, 64], range=[[0, img_width], [0, img_height]])
heatmap_ball = heatmap_ball.T  # Transpose for correct orientation
heatmap_ball = heatmap_ball / np.max(heatmap_ball)
heatmap_ball = heatmap_ball ** 0.5  # Optional enhancement
# === Plot heatmap on top of court image ===



# Normalize and enhance contrast


# Plot
plt.figure(figsize=(10, 6))
plt.imshow(court_img_rgb, extent=[0, img_width, 0, img_height])
plt.imshow(heatmap, cmap='hot', alpha=0.6, extent=[0, img_width, 0, img_height])  # increased alpha
plt.title("Player Position Heatmap")
plt.axis('off')
plt.savefig('CSVOutput/player_heatmap.png', bbox_inches='tight', pad_inches=0)



# Ball
# plt.figure(figsize=(10, 6))
# plt.imshow(court_img_rgb, extent=[0, img_width, 0, img_height])
# plt.imshow(heatmap_ball, cmap='hot', alpha=0.6, extent=[0, img_width, 0, img_height])  # increased alpha
# plt.title("Ball Position Heatmap")
# plt.axis('off')
# plt.savefig('CSVOutput/ball_heatmap.png', bbox_inches='tight', pad_inches=0)
