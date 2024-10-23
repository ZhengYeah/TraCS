import numpy as np
import pickle


def load_nyc():
    with open(f"./NYC/nyc_location_space.pkl", "rb") as f:
        location_space = pickle.load(f)
    with open(f"./NYC/nyc_trajectory.pkl", "rb") as f:
        trajectory = pickle.load(f)
    return location_space, trajectory



if __name__ == '__main__':
    location_space, trajectory = load_nyc()
    print(f"Length of location space: {len(location_space)}")
    print(trajectory[:3])
