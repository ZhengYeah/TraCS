import numpy as np
import pickle
from pathlib import Path

from src.methods.ngram import ngram_perturb
from src.methods.tp import tp_perturb
from src.methods.srr_3_groups import srr_perturb
from src.methods.wrapped_tracs import tracs_d, tracs_c
from src.utilities.trajectory_distance import hotpot_count
from src.utilities.gps_unit_transformation import unit_square_to_gps
from multiprocessing import Pool
from itertools import repeat

pi = np.pi

def load_chi():
    chi_dir = Path(__file__).parent.parent / "experiments" / "discrete_space" / "CHI"
    with open(chi_dir / "chi_location_space.pkl", "rb") as f:
        location_space = pickle.load(f)
    with open(chi_dir / "chi_trajectory.pkl", "rb") as f:
        trajectory = pickle.load(f)
    return location_space, trajectory

def find_hotspot(traj_set, top_percent=0.2):
    """
    Find the hotspot in the trajectory set, defined as the location that appears most frequently across all trajectories.
    """
    location_count = {}
    for traj in traj_set:
        for loc in traj:
            loc_tuple = tuple(loc)
            if loc_tuple not in location_count:
                location_count[loc_tuple] = 0
            location_count[loc_tuple] += 1
    # sort locations by count and select the top percent as hotspot
    sorted_locations = sorted(location_count.items(), key=lambda x: x[1], reverse=True)
    top_k = int(len(sorted_locations) * top_percent)
    hotspot = [loc for loc, count in sorted_locations[:top_k]]
    return hotspot

def process_trajectory_2(traj, hotspot_set):
    location_space, epsilon = load_chi()[0], 2
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
    # CHI: x_min: 41.60015, x_max: 41.99822, y_min: -87.9952, y_max: -87.50765
    x_min, x_max, y_min, y_max = 41.60015, 41.99822, -87.9952, -87.50765
    hotspot_set = unit_square_to_gps(hotspot_set, x_min, x_max, y_min, y_max)
    gps_traj = unit_square_to_gps(traj, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tp = unit_square_to_gps(perturbed_traj_tp, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_ngram = unit_square_to_gps(perturbed_traj_ngram, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_srr = unit_square_to_gps(perturbed_traj_srr, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_d = unit_square_to_gps(perturbed_traj_tracs_d, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_c = unit_square_to_gps(perturbed_traj_tracs_c, x_min, x_max, y_min, y_max)
    # compute errors
    ground_truth = hotpot_count(gps_traj, hotspot_set)
    error_tp = np.abs(hotpot_count(gps_perturbed_traj_tp, hotspot_set) - ground_truth)
    error_ngram = np.abs(hotpot_count(gps_perturbed_traj_ngram, hotspot_set) - ground_truth)
    error_srr = np.abs(hotpot_count(gps_perturbed_traj_srr, hotspot_set) - ground_truth)
    error_tracs_d = np.abs(hotpot_count(gps_perturbed_traj_tracs_d, hotspot_set) - ground_truth)
    error_tracs_c = np.abs(hotpot_count(gps_perturbed_traj_tracs_c, hotspot_set) - ground_truth)
    return [error_tp, error_ngram, error_srr, error_tracs_d, error_tracs_c]

def process_trajectory_4(traj, hotspot_set):
    location_space, epsilon = load_chi()[0], 4
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
    # CHI: x_min: 41.60015, x_max: 41.99822, y_min: -87.9952, y_max: -87.50765
    x_min, x_max, y_min, y_max = 41.60015, 41.99822, -87.9952, -87.50765
    hotspot_set = unit_square_to_gps(hotspot_set, x_min, x_max, y_min, y_max)
    gps_traj = unit_square_to_gps(traj, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tp = unit_square_to_gps(perturbed_traj_tp, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_ngram = unit_square_to_gps(perturbed_traj_ngram, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_srr = unit_square_to_gps(perturbed_traj_srr, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_d = unit_square_to_gps(perturbed_traj_tracs_d, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_c = unit_square_to_gps(perturbed_traj_tracs_c, x_min, x_max, y_min, y_max)
    # compute errors
    ground_truth = hotpot_count(gps_traj, hotspot_set)
    error_tp = np.abs(hotpot_count(gps_perturbed_traj_tp, hotspot_set) - ground_truth)
    error_ngram = np.abs(hotpot_count(gps_perturbed_traj_ngram, hotspot_set) - ground_truth)
    error_srr = np.abs(hotpot_count(gps_perturbed_traj_srr, hotspot_set) - ground_truth)
    error_tracs_d = np.abs(hotpot_count(gps_perturbed_traj_tracs_d, hotspot_set) - ground_truth)
    error_tracs_c = np.abs(hotpot_count(gps_perturbed_traj_tracs_c, hotspot_set) - ground_truth)
    return [error_tp, error_ngram, error_srr, error_tracs_d, error_tracs_c]

def process_trajectory_6(traj, hotspot_set):
    location_space, epsilon = load_chi()[0], 6
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
    # CHI: x_min: 41.60015, x_max: 41.99822, y_min: -87.9952, y_max: -87.50765
    x_min, x_max, y_min, y_max = 41.60015, 41.99822, -87.9952, -87.50765
    hotspot_set = unit_square_to_gps(hotspot_set, x_min, x_max, y_min, y_max)
    gps_traj = unit_square_to_gps(traj, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tp = unit_square_to_gps(perturbed_traj_tp, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_ngram = unit_square_to_gps(perturbed_traj_ngram, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_srr = unit_square_to_gps(perturbed_traj_srr, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_d = unit_square_to_gps(perturbed_traj_tracs_d, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_c = unit_square_to_gps(perturbed_traj_tracs_c, x_min, x_max, y_min, y_max)
    # compute errors
    ground_truth = hotpot_count(gps_traj, hotspot_set)
    error_tp = np.abs(hotpot_count(gps_perturbed_traj_tp, hotspot_set) - ground_truth)
    error_ngram = np.abs(hotpot_count(gps_perturbed_traj_ngram, hotspot_set) - ground_truth)
    error_srr = np.abs(hotpot_count(gps_perturbed_traj_srr, hotspot_set) - ground_truth)
    error_tracs_d = np.abs(hotpot_count(gps_perturbed_traj_tracs_d, hotspot_set) - ground_truth)
    error_tracs_c = np.abs(hotpot_count(gps_perturbed_traj_tracs_c, hotspot_set) - ground_truth)
    return [error_tp, error_ngram, error_srr, error_tracs_d, error_tracs_c]

def process_trajectory_8(traj, hotspot_set):
    location_space, epsilon = load_chi()[0], 8
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
    # CHI: x_min: 41.60015, x_max: 41.99822, y_min: -87.9952, y_max: -87.50765
    x_min, x_max, y_min, y_max = 41.60015, 41.99822, -87.9952, -87.50765
    hotspot_set = unit_square_to_gps(hotspot_set, x_min, x_max, y_min, y_max)
    gps_traj = unit_square_to_gps(traj, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tp = unit_square_to_gps(perturbed_traj_tp, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_ngram = unit_square_to_gps(perturbed_traj_ngram, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_srr = unit_square_to_gps(perturbed_traj_srr, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_d = unit_square_to_gps(perturbed_traj_tracs_d, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_c = unit_square_to_gps(perturbed_traj_tracs_c, x_min, x_max, y_min, y_max)
    # compute errors
    ground_truth = hotpot_count(gps_traj, hotspot_set)
    error_tp = np.abs(hotpot_count(gps_perturbed_traj_tp, hotspot_set) - ground_truth)
    error_ngram = np.abs(hotpot_count(gps_perturbed_traj_ngram, hotspot_set) - ground_truth)
    error_srr = np.abs(hotpot_count(gps_perturbed_traj_srr, hotspot_set) - ground_truth)
    error_tracs_d = np.abs(hotpot_count(gps_perturbed_traj_tracs_d, hotspot_set) - ground_truth)
    error_tracs_c = np.abs(hotpot_count(gps_perturbed_traj_tracs_c, hotspot_set) - ground_truth)
    return [error_tp, error_ngram, error_srr, error_tracs_d, error_tracs_c]

def process_trajectory_10(traj, hotspot_set):
    location_space, epsilon = load_chi()[0], 10
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
    # CHI: x_min: 41.60015, x_max: 41.99822, y_min: -87.9952, y_max: -87.50765
    x_min, x_max, y_min, y_max = 41.60015, 41.99822, -87.9952, -87.50765
    hotspot_set = unit_square_to_gps(hotspot_set, x_min, x_max, y_min, y_max)
    gps_traj = unit_square_to_gps(traj, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tp = unit_square_to_gps(perturbed_traj_tp, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_ngram = unit_square_to_gps(perturbed_traj_ngram, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_srr = unit_square_to_gps(perturbed_traj_srr, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_d = unit_square_to_gps(perturbed_traj_tracs_d, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_c = unit_square_to_gps(perturbed_traj_tracs_c, x_min, x_max, y_min, y_max)
    # compute errors
    ground_truth = hotpot_count(gps_traj, hotspot_set)
    error_tp = np.abs(hotpot_count(gps_perturbed_traj_tp, hotspot_set) - ground_truth)
    error_ngram = np.abs(hotpot_count(gps_perturbed_traj_ngram, hotspot_set) - ground_truth)
    error_srr = np.abs(hotpot_count(gps_perturbed_traj_srr, hotspot_set) - ground_truth)
    error_tracs_d = np.abs(hotpot_count(gps_perturbed_traj_tracs_d, hotspot_set) - ground_truth)
    error_tracs_c = np.abs(hotpot_count(gps_perturbed_traj_tracs_c, hotspot_set) - ground_truth)
    return [error_tp, error_ngram, error_srr, error_tracs_d, error_tracs_c]

if __name__ == '__main__':
    location_space, trajectory = load_chi()
    # first 100 and last 100 trajectories
    trajectory = trajectory[100:200]
    hotspot_set = find_hotspot(trajectory)
    rows = []
    with Pool() as pool:
        error_list = pool.starmap(process_trajectory_2, zip(trajectory, repeat(hotspot_set)))
    error_list = np.array(error_list)
    row = [2, np.mean(error_list[:, 0]), np.mean(error_list[:, 1]), np.mean(error_list[:, 2]), np.mean(error_list[:, 3]), np.mean(error_list[:, 4])]
    rows.append(row)
    print(f"epsilon: 2, tp: {np.mean(error_list[:, 0])}, ngram: {np.mean(error_list[:, 1])}, srr: {np.mean(error_list[:, 2])}, tracs-d: {np.mean(error_list[:, 3])}, tracs-c: {np.mean(error_list[:, 4])}")
    with Pool() as pool:
        error_list = pool.starmap(process_trajectory_4, zip(trajectory, repeat(hotspot_set)))
    error_list = np.array(error_list)
    row = [4, np.mean(error_list[:, 0]), np.mean(error_list[:, 1]), np.mean(error_list[:, 2]), np.mean(error_list[:, 3]), np.mean(error_list[:, 4])]
    rows.append(row)
    print(f"epsilon: 4, tp: {np.mean(error_list[:, 0])}, ngram: {np.mean(error_list[:, 1])}, srr: {np.mean(error_list[:, 2])}, tracs-d: {np.mean(error_list[:, 3])}, tracs-c: {np.mean(error_list[:, 4])}")
    with Pool() as pool:
        error_list = pool.starmap(process_trajectory_6, zip(trajectory, repeat(hotspot_set)))
    error_list = np.array(error_list)
    row = [6, np.mean(error_list[:, 0]), np.mean(error_list[:, 1]), np.mean(error_list[:, 2]), np.mean(error_list[:, 3]), np.mean(error_list[:, 4])]
    rows.append(row)
    print(f"epsilon: 6, tp: {np.mean(error_list[:, 0])}, ngram: {np.mean(error_list[:, 1])}, srr: {np.mean(error_list[:, 2])}, tracs-d: {np.mean(error_list[:, 3])}, tracs-c: {np.mean(error_list[:, 4])}")
    with Pool() as pool:
        error_list = pool.starmap(process_trajectory_8, zip(trajectory, repeat(hotspot_set)))
    error_list = np.array(error_list)
    row = [8, np.mean(error_list[:, 0]), np.mean(error_list[:, 1]), np.mean(error_list[:, 2]), np.mean(error_list[:, 3]), np.mean(error_list[:, 4])]
    rows.append(row)
    print(f"epsilon: 8, tp: {np.mean(error_list[:, 0])}, ngram: {np.mean(error_list[:, 1])}, srr: {np.mean(error_list[:, 2])}, tracs-d: {np.mean(error_list[:, 3])}, tracs-c: {np.mean(error_list[:, 4])}")
    with Pool() as pool:
        error_list = pool.starmap(process_trajectory_10, zip(trajectory, repeat(hotspot_set)))
    error_list = np.array(error_list)
    row = [10, np.mean(error_list[:, 0]), np.mean(error_list[:, 1]), np.mean(error_list[:, 2]), np.mean(error_list[:, 3]), np.mean(error_list[:, 4])]
    rows.append(row)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 5))
    plt.rcParams["font.size"] = 20
    plt.ylim(0, 6.5)
    plt.xticks([2, 4, 6, 8, 10])
    plt.plot([2, 4, 6, 8, 10], [row[3] for row in rows], label="L-SRR", linestyle="--", color="orange", marker="*")
    plt.plot([2, 4, 6, 8, 10], [row[1] for row in rows], label="ATP", linestyle="--", color="black", marker="x")
    plt.plot([2, 4, 6, 8, 10], [row[2] for row in rows], label="NGram", linestyle="--", color="green", marker="^")
    plt.plot([2, 4, 6, 8, 10], [row[4] for row in rows], label="TraCS-D", linestyle="-.", color="red", marker="o")
    plt.plot([2, 4, 6, 8, 10], [row[5] for row in rows], label="TraCS-C", linestyle=":", color="blue", marker="s")
    plt.xlabel(r'Privacy parameter $\varepsilon$')
    plt.ylabel(r"Error in hotpot preservation")
    plt.legend()
    plt.title("Figure 10b")
    plt.show()
