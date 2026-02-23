import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (5, 5)


# open the csv file
df = pd.read_csv("./griding_em.csv")
# draw the plot
plt.ylim(0, 0.8)
plt.xticks([2, 4, 6, 8, 10])
plt.plot(df["epsilon"], df["results_10"], label="EM (100 Cells)", linestyle="--", color="green", marker="^")
plt.plot(df["epsilon"], df["results_100"], label="EM (10000 Cells)", linestyle="--", color="black", marker="x")
plt.plot(df["epsilon"], df["results_tracs_c"], label="TraCS-C", linestyle=":", color="blue", marker="s")
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Average error")
plt.legend()
plt.savefig("./griding.pdf", bbox_inches='tight')
plt.show()


