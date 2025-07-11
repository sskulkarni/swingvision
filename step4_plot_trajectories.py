import pandas as pd
import matplotlib.pyplot as plt
import sys


# Load tracked data
df = pd.read_csv("outputs/tracked_centroids_player_1_auto.csv")

# Separate by object type
players = df[df['Object'] == 'person']
ball = df[df['Object'] == 'sports ball']



# Plot setup
plt.figure(figsize=(10, 6))
plt.title("Player and Ball Trajectories")
plt.xlabel("X Position")
plt.ylabel("Y Position")

# Plot players with different colors
for pid in players['ID'].unique():
    data = players[players['ID'] == pid]
    plt.plot(data['X'], data['Y'], label=f"Player {pid}")

# Plot ball
# for bid in ball['ID'].unique():
#     data = ball[ball['ID'] == bid]
#     plt.plot(data['X'], data['Y'], 'o-', label=f"Ball {bid}", color='orange')

plt.gca().invert_yaxis()  # Match video coordinate system
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/trajectory_plot_player_1_auto.png")
plt.show()
