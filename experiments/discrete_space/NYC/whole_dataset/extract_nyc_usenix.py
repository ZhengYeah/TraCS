import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('dataset_TSMC2014_NYC.csv', encoding='latin-1')
location_space = np.array(df[['Latitude', 'Longitude']])
# remove duplicates
location_space = np.unique(location_space, axis=0)
location_space = np.round(location_space, 8)
# normalize the location space
x_min, x_max = np.min(location_space[:, 0]), np.max(location_space[:, 0])
y_min, y_max = np.min(location_space[:, 1]), np.max(location_space[:, 1])
print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
location_space = (location_space - [x_min, y_min]) / [x_max - x_min, y_max - y_min]
print(f"Length of location space: {len(location_space)}")

# write to pickle
with open(f"nyc_location_space.pkl", "wb") as f:
    pickle.dump(location_space, f)


# extract trajectory
id_location = np.array(df[['User ID', 'Latitude', 'Longitude']])
rounded_location = np.round(id_location[:, 1:], 8)
# normalize the locations
x_min, x_max = np.min(rounded_location[:, 0]), np.max(rounded_location[:, 0])
y_min, y_max = np.min(rounded_location[:, 1]), np.max(rounded_location[:, 1])
rounded_location = (rounded_location - [x_min, y_min]) / [x_max - x_min, y_max - y_min]
id_location = np.hstack((id_location[:, 0].reshape(-1, 1), rounded_location))
id_location = np.unique(id_location, axis=0)

cur_id = id_location[0, 0]
trajectory = []
cur_id_trajectory = []
for i in range(len(id_location)):
    if id_location[i, 0] == cur_id:
        cur_id_trajectory.append(id_location[i, 1:])
    else:
        trajectory.append(cur_id_trajectory)
        cur_id_trajectory = [id_location[i, 1:]]
        cur_id = id_location[i, 0]
print(f"Length of trajectory: {len(trajectory)}")
with open(f"nyc_trajectory.pkl", "wb") as f:
    pickle.dump(trajectory, f)
