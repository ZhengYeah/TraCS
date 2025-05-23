import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (5, 5)


# open the csv file
df = pd.read_csv("./experiment_1_3.csv")
# draw the plot
plt.ylim(0, 3.5)
plt.xticks([2, 4, 6, 8, 10])

plt.plot(df["epsilon"], df["tracs_c"], label="TraCS-C", linestyle=":", color="blue", marker="s")
# plt.plot(df["epsilon"], df["strawman"], label="Strawman", linestyle="--", color="black", marker="x")
plt.plot(df["epsilon"], df["tracs_d"], label="TraCS-D", linestyle="-.", color="red", marker="o")

plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Average error (AE)")
plt.legend()
plt.savefig("./experiment_1_3.pdf")
plt.show()
