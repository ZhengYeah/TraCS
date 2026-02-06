import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (5, 5)


# # open the csv file
# df = pd.read_csv("./appendix_range_query_chi.csv")
# # draw the plot
# plt.ylim(0, 95)
# plt.xticks([2, 4, 6, 8, 10])
# plt.plot(df["epsilon"], df["srr"] * 100, label="SRR", linestyle="--", color="orange", marker="*")
# plt.plot(df["epsilon"], df["tp"] * 100, label="ATP", linestyle="--", color="black", marker="x")
# plt.plot(df["epsilon"], df["ngram"] * 100, label="NGram", linestyle="--", color="green", marker="^")
# plt.plot(df["epsilon"], df["tracs_d"] * 100, label="TraCS-D", linestyle="-.", color="red", marker="o")
# plt.plot(df["epsilon"], df["tracs_c"] * 100, label="TraCS-C", linestyle=":", color="blue", marker="s")
#
# plt.xlabel(r'Privacy parameter $\varepsilon$')
# plt.ylabel(r"Percentage of preservation (%)")
# plt.legend()
# plt.savefig("./appendix_range_query.pdf", bbox_inches='tight')
# plt.show()

# open the csv file
df = pd.read_csv("appendix_hotpot_chi.csv")
# draw the plot
plt.ylim(0, 6.5)
plt.xticks([2, 4, 6, 8, 10])
plt.plot(df["epsilon"], df["srr"], label="L-SRR", linestyle="--", color="orange", marker="*")
plt.plot(df["epsilon"], df["tp"], label="ATP", linestyle="--", color="black", marker="x")
plt.plot(df["epsilon"], df["ngram"], label="NGram", linestyle="--", color="green", marker="^")
plt.plot(df["epsilon"], df["tracs_d"], label="TraCS-D", linestyle="-.", color="red", marker="o")
plt.plot(df["epsilon"], df["tracs_c"], label="TraCS-C", linestyle=":", color="blue", marker="s")

plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Error in hotpot preservation")
plt.legend()
plt.savefig("./appendix_hotpot.pdf", bbox_inches='tight')
plt.show()

