import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load arrays
sX = np.load('sX.npy')
sY = np.load('sY_augment.npy')
# ---------------------------------------------------------------------
# Helper: convert binary stream into consecutive (start, end, value) blocks
# ---------------------------------------------------------------------
def to_blocks(arr):
    blocks = []
    start = 0
    curr = arr[0]
    for i in range(1, len(arr)):
        if arr[i] != curr:
            blocks.append((start, i, curr))
            start = i
            curr = arr[i]
    blocks.append((start, len(arr), curr))
    return blocks

blocksX = to_blocks(sX)
blocksY = to_blocks(sY)

# where both are 1
both_one = (sX == 1) & (sY == 1)
align_blocks = to_blocks(both_one.astype(int))

# ---------------------------------------------------------------------
# Styling: Seaborn + Wong palette
# ---------------------------------------------------------------------
wong = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # light blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#0072B2",  # dark blue
    "#D55E00",  # red
    "#CC79A7",  # pink
]

color_on      = wong[3]      # light blue for 1
color_off     = "#f0f0f0"    # neutral grey for 0
color_align   = wong[6]      # orange band for X=Y=1

# sns.set_theme(style="white", font="sans-serif")
# sns.set_context("paper")

fig, ax = plt.subplots(figsize=(6.5, 2.0))  # good aspect for a paper figure

# ---------------------------------------------------------------------
# Plot X and Y blocks
# ---------------------------------------------------------------------
for start, end, val in blocksX:
    ax.fill_between(
        [start, end],
        1.1, 1.9,
        color=color_on if val == 1 else color_off,
        linewidth=0,
        zorder=1
    )

for start, end, val in blocksY:
    ax.fill_between(
        [start, end],
        0.1, 0.9,
        color=color_on if val == 1 else color_off,
        linewidth=0,
        zorder=1
    )

# ---------------------------------------------------------------------
# Highlight aligned 1-blocks ON TOP, spanning both rows
# ---------------------------------------------------------------------
for start, end, val in align_blocks:
    if val == 1:
        ax.axvspan(
            start,
            end,
            ymin=0.0,
            ymax=1.0,       # span over both X and Y rows
            color=color_align,
            alpha=0.2,      # requested opacity
            zorder=2,
            edgecolor='none', linewidth=0
        )

# ---------------------------------------------------------------------
# Axes formatting (publication style)
# ---------------------------------------------------------------------
ax.set_ylim(0, 2)
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(["Y", "X"])
ax.set_xlabel("Time index")

# Minimal spines
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_linewidth(0.8)

ax.tick_params(axis="x", width=0.8, length=3)
ax.tick_params(axis="y", length=0)

# Optional: subtle title
ax.set_title("Binary Streams with Aligned 1-Blocks", pad=8)

# Legend (compact, paper-friendly)
# handles = [
#     Patch(facecolor=color_on, edgecolor="none", label="State = 1"),
#     Patch(facecolor=color_off, edgecolor="none", label="State = 0"),
#     Patch(facecolor=color_align, edgecolor="none", alpha=0.2, label="X = Y = 1"),
# ]
# ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig("F:\\users\\prasetia\\data\\Children\\children_sync\\results\\bayesian_ttest\\statistical\\original_augment.pdf", format="pdf")
plt.show()