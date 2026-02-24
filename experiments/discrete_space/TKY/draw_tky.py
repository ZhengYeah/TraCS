import pickle
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["figure.figsize"] = (5, 5)

# load location space
with open(f"./tky_location_space.pkl", "rb") as f:
    location_space = pickle.load(f)
# draw the location space
# TKY: x_min: 35.51076909, x_max: 35.86407116, y_min: 139.47087765, y_max: 139.908359
# transform back to gps coordinates
x_min, x_max, y_min, y_max = 35.51076909, 35.86407116, 139.47087765, 139.908359
location_space = location_space * [x_max - x_min, y_max - y_min] + [x_min, y_min]
plt.scatter(location_space[:, 1], location_space[:, 0], s=10, alpha=0.5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("TKY Location Space")
plt.savefig("./tky_location_space.pdf", bbox_inches='tight')
plt.show()