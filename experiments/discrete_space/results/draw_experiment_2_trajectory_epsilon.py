import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (5, 5)



# open the csv file
df = pd.read_csv("./experiment_2_tky_trajectory_epsilon.csv")
# draw the plot
plt.ylim(0, 0.12)
# Use epsilon = 113 * [2, 4, 6, 8, 10] for epsilon
powers = [2, 4, 6, 8, 10]
plt.xticks([113 * p for p in powers], [2, 4, 6, 8, 10])
line_1, = plt.plot(df["epsilon"], df["lsrr"], label="L-SRR", linestyle="--", color="orange", marker="*")
line_2, = plt.plot(df["epsilon"], df["tp"], label="ATP", linestyle="--", color="black", marker="x")
line_3, = plt.plot(df["epsilon"], df["ngram"], label="NGram", linestyle="--", color="green", marker="^")
line_4, = plt.plot(df["epsilon"], df["tracs_d"], label="TraCS-D", linestyle="-.", color="red", marker="o")
line_5, = plt.plot(df["epsilon"], df["tracs_c"], label="TraCS-C", linestyle=":", color="blue", marker="s")

plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Average error")
legend_1 = plt.legend(handles=[line_1,line_2,line_3], loc="upper right")
plt.gca().add_artist(legend_1)
plt.legend(handles=[line_4,line_5], loc="lower left")
plt.savefig("./experiment_2_tky_trajectory_epsilon.pdf", bbox_inches='tight')
plt.show()


# open the csv file
df = pd.read_csv("./experiment_2_chi_trajectory_epsilon_new.csv")
# draw the plot
plt.ylim(0, 0.14)
plt.yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12])
# Use epsilon = 13 * [2, 4, 6, 8, 10] for epsilon
powers = [2, 4, 6, 8, 10]
plt.xticks([13 * p for p in powers], [2, 4, 6, 8, 10])
line_1, = plt.plot(df["epsilon"], df["srr"], label="L-SRR", linestyle="--", color="orange", marker="*")
line_2, = plt.plot(df["epsilon"], df["tp"], label="ATP", linestyle="--", color="black", marker="x")
line_3, = plt.plot(df["epsilon"], df["ngram"], label="NGram", linestyle="--", color="green", marker="^")
line_4, = plt.plot(df["epsilon"], df["tracs_d"], label="TraCS-D", linestyle="-.", color="red", marker="o")
line_5, = plt.plot(df["epsilon"], df["tracs_c"], label="TraCS-C", linestyle=":", color="blue", marker="s")

plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Average error")
legend_1 = plt.legend(handles=[line_1,line_2,line_3], loc="upper right")
plt.gca().add_artist(legend_1)
plt.legend(handles=[line_4,line_5], loc="lower left")

plt.savefig("./experiment_2_chi_trajectory_epsilon.pdf", bbox_inches='tight')
plt.show()

