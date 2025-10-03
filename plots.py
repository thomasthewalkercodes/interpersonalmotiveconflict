import matplotlib.pyplot as plt
import numpy as np
from human_generated import (
    octants,
    df_satisfaction_history,
    history,
    df_influence_of_octants,
)
from matplotlib.colors import ListedColormap

colors = plt.cm.tab10(np.linspace(0, 1, 8))

# Plot 1: Satisfaction trajectories
plt.figure(figsize=(8, 5))
for i, octant in enumerate(octants):
    plt.plot(
        df_satisfaction_history[octant], label=octant, color=colors[i], linewidth=2
    )
plt.axhline(y=0, color="black", linestyle="--", alpha=0.7, linewidth=1)
plt.axhline(y=1, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Max")
plt.axhline(y=-1, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Min")
plt.xlabel("Step", fontsize=12)
plt.ylabel("Satisfaction Level", fontsize=12)
plt.title("Motive Satisfaction Over Time", fontsize=14, fontweight="bold")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt.ylim(-1.1, 1.1)
plt.tight_layout()
plt.savefig("plot1_satisfaction_trajectories.png")
plt.show()

# Plot 2: Active behavior timeline
plt.figure(figsize=(8, 5))
active_matrix = np.full((8, len(history["active_behavior"])), -1, dtype=float)
for round_idx, active in enumerate(history["active_behavior"]):
    if active is not None:
        active_idx = octants.index(active)
        active_matrix[active_idx, round_idx] = 1
cmap_custom = ListedColormap(["lightgray", "darkred"])
plt.imshow(active_matrix, cmap=cmap_custom, aspect="auto", vmin=-1, vmax=1)
plt.title("Active Behavior Pattern", fontsize=14, fontweight="bold")
plt.xlabel("Step", fontsize=12)
plt.ylabel("Motive", fontsize=12)
plt.yticks(range(8), octants)
plt.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
plt.tight_layout()
plt.savefig("plot2_active_behavior_timeline.png")
plt.show()

# Plot 3: Correlation matrix heatmap
plt.figure(figsize=(8, 5))
im3 = plt.imshow(df_influence_of_octants.values, cmap="RdBu_r", vmin=-1, vmax=1)
plt.title("Correlation Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Motive", fontsize=12)
plt.ylabel("Motive", fontsize=12)
plt.xticks(range(8), octants)
plt.yticks(range(8), octants)
for i in range(8):
    for j in range(8):
        plt.text(
            j,
            i,
            f"{df_influence_of_octants.values[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=8,
        )
cbar3 = plt.colorbar(im3)
cbar3.set_label("Correlation", fontsize=11)
plt.tight_layout()
plt.savefig("plot3_correlation_matrix.png")
plt.show()

# Plot 4: Activity frequency bar chart
plt.figure(figsize=(8, 5))
activity_counts = []
for octant in octants:
    count = sum(1 for x in history["active_behavior"] if x == octant)
    activity_counts.append(count)
bars = plt.bar(range(8), activity_counts, color=colors)
plt.xlabel("Motive", fontsize=12)
plt.ylabel("Steps Active", fontsize=12)
plt.title("Activity Frequency by Motive", fontsize=14, fontweight="bold")
plt.xticks(range(8), octants)
plt.grid(True, alpha=0.3, axis="y")
for i, (bar, count) in enumerate(zip(bars, activity_counts)):
    if count > 0:
        plt.text(i, count + 0.5, str(count), ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("plot4_activity_frequency.png")
plt.show()
