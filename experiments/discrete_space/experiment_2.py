import numpy as np
import pickle
from src.methods.ngram import ngram_perturb
from src.methods.tp import tp_perturb
from src.methods.wrapped_tracs import tracs_d, tracs_c
from src.utilities.trajectory_distance import averaged_l2_distance
from src.utilities.gps_unit_transformation import unit_square_to_gps
from multiprocessing import Pool
pi = np.pi


def load_nyc():
    with open(f"./NYC/nyc_location_space.pkl", "rb") as f:
        location_space = pickle.load(f)
    with open(f"./NYC/nyc_trajectory.pkl", "rb") as f:
        trajectory = pickle.load(f)
    return location_space, trajectory


if __name__ == '__main__':
    location_space, trajectory = load_nyc()
    trajectory = trajectory[:100]
    error_list = np.zeros((len(trajectory), 4))
    epsilon = 2
    for k, traj in enumerate(trajectory):
        # perturbed trajectories
        perturbed_traj_tp = tp_perturb(traj, location_space, epsilon)
        perturbed_traj_ngram = ngram_perturb(traj, location_space, epsilon, theta=0.75)
        perturbed_traj_tracs_d = tracs_d(traj, epsilon, pi / (pi + 1) * epsilon)
        perturbed_traj_tracs_c = tracs_c(traj, epsilon, epsilon / 2)
        # round the perturbed locations to the nearest location in the location space (TraCS)
        for i in range(len(perturbed_traj_tracs_d)):
            nearest_location_d = location_space[np.argmin(np.linalg.norm(location_space - perturbed_traj_tracs_d[i], axis=1))]
            perturbed_traj_tracs_d[i] = nearest_location_d
            nearest_location_c = location_space[np.argmin(np.linalg.norm(location_space - perturbed_traj_tracs_c[i], axis=1))]
            perturbed_traj_tracs_c[i] = nearest_location_c
        # transform back to gps coordinates
        # NYC dataset: x_min: 40.56709, x_max: 40.97472, y_min: -74.26947, y_max: -73.69992
        x_min, x_max, y_min, y_max = 40.56709, 40.97472, -74.26947, -73.69992
        gps_traj = unit_square_to_gps(traj, x_min, x_max, y_min, y_max)
        gps_perturbed_traj_tp = unit_square_to_gps(perturbed_traj_tp, x_min, x_max, y_min, y_max)
        gps_perturbed_traj_ngram = unit_square_to_gps(perturbed_traj_ngram, x_min, x_max, y_min, y_max)
        gps_perturbed_traj_tracs_d = unit_square_to_gps(perturbed_traj_tracs_d, x_min, x_max, y_min, y_max)
        gps_perturbed_traj_tracs_c = unit_square_to_gps(perturbed_traj_tracs_c, x_min, x_max, y_min, y_max)
        # compute errors
        error_tp = averaged_l2_distance(gps_traj, gps_perturbed_traj_tp)
        error_ngram = averaged_l2_distance(gps_traj, gps_perturbed_traj_ngram)
        error_tracs_d = averaged_l2_distance(gps_traj, gps_perturbed_traj_tracs_d)
        error_tracs_c = averaged_l2_distance(gps_traj, gps_perturbed_traj_tracs_c)
        error_list[k] = [error_tp, error_ngram, error_tracs_d, error_tracs_c]
    print(f"epsilon: {epsilon}, tp: {np.mean(error_list[:, 0])}, ngram: {np.mean(error_list[:, 1])}, tracs-d: {np.mean(error_list[:, 2])}, tracs-c: {np.mean(error_list[:, 3])}")


