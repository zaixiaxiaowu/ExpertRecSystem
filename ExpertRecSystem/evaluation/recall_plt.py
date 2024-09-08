# Re-importing necessary packages after reset
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
font_prop = fm.FontProperties(fname=font_path)

plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

# Data from the provided test results
top_k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
hit_rate = [40.54, 54.73, 62.84, 66.89, 70.27, 72.97, 77.03, 77.70, 79.05, 79.05]
marginal_increase = np.diff(hit_rate, prepend=0)

# Define colors for the bar and line plots
hit_rate_color_new = (207 / 255, 218 / 255, 236 / 255)  # Bar chart color
marginal_increase_color_new = (191 / 255, 187 / 255, 186 / 255)  # Line plot color

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot hit rate as bar chart with thicker bars, and black edge color (border)
ax1.bar(
    top_k,
    hit_rate,
    color=hit_rate_color_new,
    edgecolor="black",
    label="命中率",
    width=6,
)
ax1.set_xlabel("Top-K")
ax1.set_ylabel("命中率 (%)")
ax1.set_xticks(top_k)
ax1.legend(loc="upper left")

# Create a second y-axis for the marginal increase curve with a thicker line and new color
ax2 = ax1.twinx()
ax2.plot(
    top_k,
    marginal_increase,
    color=marginal_increase_color_new,
    marker="o",
    linestyle="-",
    linewidth=3,
    label="边际增量",
)
ax2.set_ylabel("边际增量 (%)")

# Ensure that the right-hand side axis line is visible and solid
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_linewidth(1.5)
ax2.spines["right"].set_color("black")

# Set consistent border thickness for all four sides
border_thickness = 1.5
for spine in ["top", "bottom", "left", "right"]:
    ax1.spines[spine].set_visible(True)
    ax1.spines[spine].set_linewidth(border_thickness)
    ax1.spines[spine].set_color("black")

ax2.spines["top"].set_visible(True)
ax2.spines["top"].set_linewidth(border_thickness)
ax2.spines["top"].set_color("black")

ax2.legend(loc="upper right")

# Ensure gridlines are removed from both axes
ax1.tick_params(direction="in")
ax2.tick_params(direction="in")

ax1.grid(False)
ax2.grid(False)


# Save as SVG with Chinese labels and no title
chinese_labels_plot_path = "assets/hit_rate_and_marginal_increase.svg"
plt.savefig(chinese_labels_plot_path, format="svg")
