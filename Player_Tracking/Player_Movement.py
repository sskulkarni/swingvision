import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
csv_path = 'CSVOutput/player_tracking.csv'
df = pd.read_csv(csv_path)
df = df[df['Player_name'] == 'Player A']

fps = 60
video_width = 1920
video_height = 1080
court_center_x = video_width // 2

# Filter Player A
player_df = df[df['Player_name'] == 'Player A'].sort_values('Frame')

# Distance
dx = player_df['X'].diff()
dy = player_df['Y'].diff()
player_df['distance'] = np.sqrt(dx**2 + dy**2)
total_distance = player_df['distance'].sum()
print(f"Total distance: {total_distance:.2f} px")

# Speed
player_df['speed'] = player_df['distance'] * fps
avg_speed = player_df['speed'].mean()
max_speed = player_df['speed'].max()
print(f"Average speed: {avg_speed:.2f} px/s")
print(f"Max speed: {max_speed:.2f} px/s")

# Court side time
left_frames = player_df[player_df['X'] < court_center_x].shape[0]
right_frames = player_df[player_df['X'] >= court_center_x].shape[0]
total_time = player_df.shape[0] / fps
print(f"Left side time: {left_frames / fps:.2f}s")
print(f"Right side time: {right_frames / fps:.2f}s")

# Zone density
player_df['zone_x'] = pd.cut(player_df['X'], bins=3, labels=['Left', 'Center', 'Right'])
player_df['zone_y'] = pd.cut(player_df['Y'], bins=3, labels=['Top', 'Middle', 'Bottom'])
zone_counts = player_df.groupby(['zone_x', 'zone_y']).size().unstack(fill_value=0)
print("\nZone Densities:")
print(zone_counts)

# Plot movement over time
plt.figure(figsize=(10, 4))
plt.plot(player_df['Frame'], player_df['X'], label='X', color='blue')
plt.plot(player_df['Frame'], player_df['Y'], label='Y', color='green')
plt.title("Movement Over Time")
plt.xlabel("Frame")
plt.ylabel("Position (px)")
plt.legend()
plt.tight_layout()
plt.savefig("CSVOutput/movement_over_time.png")
plt.show()

# Plot speed distribution
plt.figure(figsize=(6, 4))
plt.hist(player_df['speed'].dropna(), bins=20, color='orange', edgecolor='black')
plt.title("Speed Distribution")
plt.xlabel("Speed (px/s)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("CSVOutput/speed_distribution.png")
plt.show()

# Plot speed over time
plt.figure(figsize=(10, 4))
plt.plot(player_df['Frame'], player_df['speed'], color='red')
plt.title("Speed Over Time")
plt.xlabel("Frame")
plt.ylabel("Speed (px/s)")
plt.tight_layout()
plt.savefig("CSVOutput/speed_over_time.png")
plt.show()
