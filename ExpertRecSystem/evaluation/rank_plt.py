import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
font_prop = fm.FontProperties(fname=font_path)

plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

x = [3, 4, 5, 6, 7, 8, 9, 10]
narrow_map = [0.3983, 0.3594, 0.3123, 0.3230, 0.3259, 0.3322, 0.3332, 0.3354]
general_map = [0.9431, 0.9254, 0.9123, 0.8742, 0.8543, 0.8321, 0.8142, 0.7963]

color_narrow_map_alt = (207 / 255, 218 / 255, 236 / 255)
color_general_map_alt = (191 / 255, 187 / 255, 186 / 255)

plt.figure(figsize=(8, 6))

bar_width = 0.35
index = range(len(x))

plt.bar(
    [i - bar_width / 2 for i in index],
    narrow_map,
    width=bar_width,
    color=color_narrow_map_alt,
    label="狭义MAP",
)
plt.bar(
    [i + bar_width / 2 for i in index],
    general_map,
    width=bar_width,
    color=color_general_map_alt,
    label="广义MAP",
)


plt.xlabel("MAP@N")
plt.ylabel("MAP值")
plt.xticks(index, [f"MAP@{i}" for i in x])

plt.gca().spines["top"].set_visible(True)
plt.gca().spines["right"].set_visible(True)
plt.gca().spines["left"].set_visible(True)
plt.gca().spines["bottom"].set_visible(True)
plt.gca().tick_params(direction="in")
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
plt.gca().grid(False)

plt.legend()

plt.savefig("assets/map.svg", format="svg")
