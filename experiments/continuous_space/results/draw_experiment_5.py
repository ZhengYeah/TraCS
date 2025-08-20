import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (5, 5)


# open the csv file
df = pd.read_csv("./experiment_5_tky.csv")
# draw the plot
plt.ylim(0, 0.14)
plt.yticks(np.arange(0, 0.14, 0.04))
plt.xticks([2, 4, 6, 8, 10])
plt.plot(df["epsilon"], df["strawman"], label="Strawman", linestyle="--", color="black", marker="x")
plt.plot(df["epsilon"], df["tracs_d"], label="TraCS-D", linestyle="-.", color="red", marker="o")
plt.plot(df["epsilon"], df["tracs_c"], label="TraCS-C", linestyle=":", color="blue", marker="s")
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Average error")
plt.legend()
plt.savefig("./experiment_5_tky.eps")
plt.show()


# open the csv file
df = pd.read_csv("./experiment_5_chi.csv")
# draw the plot
plt.ylim(0, 0.28)
plt.xticks([2, 4, 6, 8, 10])
plt.plot(df["epsilon"], df["strawman"], label="Strawman", linestyle="--", color="black", marker="x")
plt.plot(df["epsilon"], df["tracs_d"], label="TraCS-D", linestyle="-.", color="red", marker="o")
plt.plot(df["epsilon"], df["tracs_c"], label="TraCS-C", linestyle=":", color="blue", marker="s")
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Average error")
plt.legend()
plt.savefig("./experiment_5_chi.eps")
plt.show()