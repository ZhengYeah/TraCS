import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (5, 5)


# open the csv file
df = pd.read_csv("./experiment_3.csv")
# draw the plot
plt.ylim(0, 0.8)
plt.xticks([2, 4, 6, 8, 10])
plt.plot(df["epsilon"], df["sw"], label="Redesigned SW", linestyle="--", color="blue", marker="x")
plt.plot(df["epsilon"], df["tracs_d"], label="Def 3.1 & 3.3", linestyle="-.", color="red", marker="o")
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Average error")
plt.legend()
plt.savefig("./experiment_3.eps")
plt.show()
