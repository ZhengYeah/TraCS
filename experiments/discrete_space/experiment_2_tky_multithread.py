import numpy as np
import pickle
from src.methods.ngram import ngram_perturb
from src.methods.tp import tp_perturb
from src.methods.srr_3_groups import srr_perturb
from src.methods.wrapped_tracs import tracs_d, tracs_c
from src.utilities.trajectory_distance import averaged_l2_distance
from src.utilities.gps_unit_transformation import unit_square_to_gps
from multiprocessing import Pool

pi = np.pi

def load_nyc():
    with open(f"./TKY/tky_location_space.pkl", "rb") as f:
        location_space = pickle.load(f)
    with open(f"./TKY/tky_trajectory.pkl", "rb") as f:
        trajectory = pickle.load(f)
    # print(f"Length of trajectory: {len(trajectory)}")
    # print(f"Length of location space: {len(location_space)}")
    return location_space, trajectory


def process_trajectory(traj):
    location_space, epsilon = load_nyc()[0], 10
    # perturbed trajectories
    perturbed_traj_tp = tp_perturb(traj, location_space, epsilon)
    theta = 0.75 * np.sqrt(2)
    perturbed_traj_ngram = ngram_perturb(traj, location_space, epsilon, theta)
    distance_list = [0.3, 0.6]
    perturbed_traj_srr = srr_perturb(traj, location_space, epsilon, distance_list)
    perturbed_traj_tracs_d = tracs_d(traj, epsilon, pi / (pi + 1) * epsilon)
    perturbed_traj_tracs_c = tracs_c(traj, epsilon, epsilon / 2)
    # round the perturbed locations to the nearest location in the location space (TraCS)
    for i in range(len(perturbed_traj_tracs_d)):
        nearest_location_d = location_space[np.argmin(np.linalg.norm(location_space - perturbed_traj_tracs_d[i], axis=1))]
        # print(f"perturbed_traj_tracs_d[i]: {perturbed_traj_tracs_d[i]}, nearest_location_d: {nearest_location_d}")
        perturbed_traj_tracs_d[i] = nearest_location_d
        nearest_location_c = location_space[np.argmin(np.linalg.norm(location_space - perturbed_traj_tracs_c[i], axis=1))]
        perturbed_traj_tracs_c[i] = nearest_location_c
    # transform back to gps coordinates
    # TKY: x_min: 35.51076909, x_max: 35.86407116, y_min: 139.47087765, y_max: 139.908359
    x_min, x_max, y_min, y_max = 35.51076909, 35.86407116, 139.47087765, 139.908359
    gps_traj = unit_square_to_gps(traj, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tp = unit_square_to_gps(perturbed_traj_tp, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_ngram = unit_square_to_gps(perturbed_traj_ngram, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_srr = unit_square_to_gps(perturbed_traj_srr, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_d = unit_square_to_gps(perturbed_traj_tracs_d, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_c = unit_square_to_gps(perturbed_traj_tracs_c, x_min, x_max, y_min, y_max)
    # print(f"gps_traj: {gps_traj}")
    # print(f"gps_perturbed_traj_tp: {gps_perturbed_traj_tp}")
    # print(f"gps_perturbed_traj_ngram: {gps_perturbed_traj_ngram}")
    # print(f"gps_perturbed_traj_tracs_d: {gps_perturbed_traj_tracs_d}")
    # compute errors
    error_tp = averaged_l2_distance(gps_traj, gps_perturbed_traj_tp)
    error_ngram = averaged_l2_distance(gps_traj, gps_perturbed_traj_ngram)
    error_srr = averaged_l2_distance(gps_traj, gps_perturbed_traj_srr)
    error_tracs_d = averaged_l2_distance(gps_traj, gps_perturbed_traj_tracs_d)
    error_tracs_c = averaged_l2_distance(gps_traj, gps_perturbed_traj_tracs_c)
    return [error_tp, error_ngram, error_srr, error_tracs_d, error_tracs_c]


if __name__ == '__main__':
    location_space, trajectory = load_nyc()
    # first 100 trajectories
    trajectory = trajectory[:10]
    with Pool() as pool:
        error_list = pool.map(process_trajectory, trajectory)
    error_list = np.array(error_list)
    print(f"epsilon: 2, tp: {np.mean(error_list[:, 0])}, ngram: {np.mean(error_list[:, 1])}, srr: {np.mean(error_list[:,2])}, tracs-d: {np.mean(error_list[:, 3])}, tracs-c: {np.mean(error_list[:, 4])}")
    # write to csv
    with open(f"./results/experiment_2_tky.csv", "a") as f:
        f.write("tp,ngram,lsrr,tracs_d,tracs_c\n")
        f.write(f"{np.mean(error_list[:, 0])},{np.mean(error_list[:, 1])},{np.mean(error_list[:, 2])},{np.mean(error_list[:, 3])},{np.mean(error_list[:, 4])}\n")
