import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (5, 5)


# # open the csv file
# df = pd.read_csv("./experiment_2_tky.csv")
# # draw the plot
# plt.ylim(0, 0.15)
# plt.yticks([0, 0.05, 0.1, 0.15])
# plt.xticks([2, 4, 6, 8, 10])
# plt.plot(df["epsilon"], df["tp"], label="ATP", linestyle="--", color="black", marker="x")
# plt.plot(df["epsilon"], df["ngram"], label="NGram", linestyle="--", color="green", marker="^")
# plt.plot(df["epsilon"], df["tracs_d"], label="TraCS-D", linestyle="-.", color="red", marker="o")
# plt.plot(df["epsilon"], df["tracs_c"], label="TraCS-C", linestyle=":", color="blue", marker="s")
#
# plt.xlabel(r'Privacy parameter $\varepsilon$')
# plt.ylabel(r"Average error")
# plt.legend()
# plt.savefig("./experiment_2_tky.eps")
# plt.show()


# open the csv file
df = pd.read_csv("./experiment_2_chi.csv")
# draw the plot
plt.ylim(0, 0.28)
plt.xticks([2, 4, 6, 8, 10])
plt.plot(df["epsilon"], df["tp"], label="ATP", linestyle="--", color="black", marker="x")
plt.plot(df["epsilon"], df["ngram"], label="NGram", linestyle="--", color="green", marker="^")
plt.plot(df["epsilon"], df["tracs_d"], label="TraCS-D", linestyle="-.", color="red", marker="o")
plt.plot(df["epsilon"], df["tracs_c"], label="TraCS-C", linestyle=":", color="blue", marker="s")

plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Average error")
plt.legend()
plt.savefig("./experiment_2_chi.eps")
plt.show()
