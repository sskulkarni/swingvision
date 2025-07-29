import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load tracking data ===
csv_path = 'CSVOutput/player_tracking.csv'
df = pd.read_csv(csv_path)
player_id_to_plot = "Player A"  # change if needed

df = df[df['Player'] == player_id_to_plot]


# === Load tennis court image ===
court_img = cv2.imread('CSVOutput/court.jpg')  # Make sure this is a full-court image
court_img_rgb = cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)

# === Get image size ===
img_height, img_width = court_img.shape[:2]

# === Normalize tracking coordinates to image size ===
x = df['X'].values
y = df['Y'].values

# Normalize (rescale x, y into image width/height)
x_scaled = ((x - x.min()) / (x.max() - x.min())) * img_width
y_scaled = ((y - y.min()) / (y.max() - y.min())) * img_height

# Optional: Flip y to match image origin (if needed)
y_scaled = img_height - y_scaled

# === Create heatmap using 2D histogram ===
heatmap, xedges, yedges = np.histogram2d(x_scaled, y_scaled, bins=[64, 64], range=[[0, img_width], [0, img_height]])
heatmap = heatmap.T  # Transpose for correct orientation

# === Plot heatmap on top of court image ===
plt.figure(figsize=(20, 12))
plt.imshow(court_img_rgb, extent=[0, img_width, 0, img_height])
plt.imshow(heatmap, cmap='hot', alpha=0.5, extent=[0, img_width, 0, img_height])
plt.title("Player Position Heatmap")
plt.axis('off')
plt.show()
