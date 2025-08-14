import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load tracking data
csv_path = 'CSVOutput/player_tracking.csv'
df = pd.read_csv(csv_path)
df = df[df['Player_name'] == 'Player A']

# Set your actual video FPS
FPS = 15

# Ensure columns are there
assert all(col in df.columns for col in ['Frame', 'X', 'Y']), "CSV must have 'frame', 'x', 'y' columns."

# Calculate time from frame
df['time'] = df['Frame'] / FPS

# Calculate speed (pixels/sec)
df['dx'] = df['X'].diff()
df['dy'] = df['Y'].diff()
df['dt'] = df['time'].diff()
df['speed'] = np.sqrt(df['dx']**2 + df['dy']**2) / df['dt']

# Remove NaN (first row)
df = df.dropna()

# Stats
average_speed = df['speed'].mean()
max_speed = df['speed'].max()

print(f"✅ Average Speed: {average_speed:.2f} pixels/sec")
print(f"✅ Max Speed: {max_speed:.2f} pixels/sec")

# Line plot: speed over time
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['speed'], color='blue')
plt.title("Player Speed Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Speed (pixels/sec)")
plt.grid(True)
plt.tight_layout()
plt.savefig("CSVOutput/speed_over_time.png")
plt.show()

# Histogram: speed distribution
plt.figure(figsize=(8, 5))
plt.hist(df['speed'], bins=30, color='green', edgecolor='black')
plt.title("Speed Distribution")
plt.xlabel("Speed (pixels/sec)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("CSVOutput/speed_histogram.png")
plt.show()
