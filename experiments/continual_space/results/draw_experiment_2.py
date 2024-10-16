import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (5, 5)


# open the csv file
df = pd.read_csv("./experiment_2.csv")
# draw the plot
plt.ylim(0, 0.9)
plt.xticks([2, 4, 6, 8, 10])
plt.plot(df["epsilon"], df["3-RR"], label="3-RR", linestyle=":", color="blue", marker="s")
plt.plot(df["epsilon"], df["6-RR"], label="6-RR", linestyle="--", color="black", marker="x")
plt.plot(df["epsilon"], df["12-RR"], label="12-RR", linestyle="--", color="green", marker="^")
plt.plot(df["epsilon"], df["tracs_d"], label="TraCS-D", linestyle="-.", color="red", marker="o")
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Error of the direction")
plt.legend()
plt.savefig("./experiment_2.eps")
plt.show()
