import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (5, 5)
plt.style.use('seaborn-v0_8-whitegrid')

labels = [r'EM ($10^2$)', r'EM $(100^2)$', 'TraCS-C']
times = [1.64, 884.6, 0.001]
colors = ['green', 'blue', 'blue']  # one color per bar

x = np.arange(len(labels))

fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(5,5),
    gridspec_kw={'height_ratios': [1, 2]}
)

# Plot bars on BOTH axes
ax1.bar(x, times, color=colors)
ax2.bar(x, times, color=colors)

# Set y-limits to create break
ax1.set_ylim(500, 1000)   # upper part (big value)
ax2.set_ylim(0, 5)       # lower part (small values)

# Adjust bars
bars_1 = ax1.patches
bars_2 = ax2.patches
for bar in bars_1:
    bar.set_edgecolor('black')
    bar.set_linewidth(1)
    bar.set_alpha(0.7)
for bar in bars_2:
    bar.set_edgecolor('black')
    bar.set_linewidth(1)
    bar.set_alpha(0.7)

# Hide spines between axes
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)

ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

# Add diagonal break marks
d = .5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)

ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel("Time Cost")

for i, v in enumerate(times):
    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')

plt.savefig("./griding_time.pdf", bbox_inches='tight')
plt.show()
