import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (5, 5)

# load location space
with open(f"./chi_location_space.pkl", "rb") as f:
    location_space = pickle.load(f)
# draw the location space
# CHI: x_min: 41.60015, x_max: 41.99822, y_min: -87.9952, y_max: -87.50765
# transform back to gps coordinates
x_min, x_max, y_min, y_max = 41.60015, 41.99822, -87.9952, -87.50765
location_space = location_space * [x_max - x_min, y_max - y_min] + [x_min, y_min]
plt.scatter(location_space[:, 1], location_space[:, 0], s=10, alpha=0.5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("CHI Location Space")
plt.savefig("./chi_location_space.pdf", bbox_inches='tight')
plt.show()