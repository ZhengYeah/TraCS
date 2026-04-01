import pickle
import numpy as np
from pathlib import Path

from src.methods.strawman import strawman_perturbation
from src.methods.two_d_laplace import ldp_2d_laplace_batch
from src.methods.wrapped_tracs import tracs_d, tracs_c
from src.utilities.gps_unit_transformation import unit_square_to_gps
from src.utilities.trajectory_distance import averaged_l2_distance

pi = np.pi

def load_trajectory():
    """"
    Already normalized to [0,1]x[0,1]
    """
    tky_dataset = Path(__file__).parent.parent / "experiments" / "discrete_space" / "TKY" / "tky_trajectory.pkl"
    with open(tky_dataset, "rb") as f:
        trajectory = pickle.load(f)
    # print(f"Length of trajectory: {len(trajectory)}")
    # print(f"Length of location space: {len(location_space)}") # there is the first 100 trajectories
    return trajectory

def strawman(traj, epsilon):
    """
    Apply the Strawman perturbation to a trajectory.
    """
    traj_add_dummy = np.vstack((np.array([0, 0]), traj))
    perturbed_trajectory_strawman = traj_add_dummy.copy()
    num_locations = len(traj)
    for i in range(num_locations):
        ref_location = (perturbed_trajectory_strawman[i][0], perturbed_trajectory_strawman[i][1])
        location = (perturbed_trajectory_strawman[i + 1][0], perturbed_trajectory_strawman[i + 1][1])
        perturbed_trajectory_strawman[i + 1] = strawman_perturbation(ref_location, location, epsilon)
    return perturbed_trajectory_strawman

def tracs_vs_strawman_gps_tky(traj, epsilon):
    perturbed_trajectory_strawman = strawman(traj, epsilon)
    perturbed_trajectory_tracs_d = tracs_d(traj, epsilon, pi / (pi + 1) * epsilon)
    perturbed_trajectory_tracs_c = tracs_c(traj, epsilon, epsilon / 2)
    perturbed_trajectory_laplace = ldp_2d_laplace_batch(traj, epsilon)
    # transform back to gps coordinates
    # TKY: x_min: 35.51076909, x_max: 35.86407116, y_min: 139.47087765, y_max: 139.908359
    x_min, x_max, y_min, y_max = 35.51076909, 35.86407116, 139.47087765, 139.908359
    gps_traj = unit_square_to_gps(traj, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_strawman = unit_square_to_gps(perturbed_trajectory_strawman, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_d = unit_square_to_gps(perturbed_trajectory_tracs_d, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_c = unit_square_to_gps(perturbed_trajectory_tracs_c, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_laplace = unit_square_to_gps(perturbed_trajectory_laplace, x_min, x_max, y_min, y_max)
    # print(f"gps_traj: {gps_traj}")
    return (averaged_l2_distance(gps_traj, gps_perturbed_traj_strawman[1:]),
            averaged_l2_distance(gps_traj, gps_perturbed_traj_tracs_d),
            averaged_l2_distance(gps_traj, gps_perturbed_traj_tracs_c),
            averaged_l2_distance(gps_traj, gps_perturbed_traj_laplace))

rows = []
for epsilon in [2, 4, 6, 8, 10]:
    distance_tracs_d, distance_strawman, distance_tracs_c, distance_2d_laplace = 0, 0, 0, 0
    set_of_traj = load_trajectory()[:100]  # first 100 trajectories
    for traj in set_of_traj:
        for _ in range(20):
            # average distance
            distance_strawman += tracs_vs_strawman_gps_tky(traj, epsilon)[0]
            distance_tracs_d += tracs_vs_strawman_gps_tky(traj, epsilon)[1]
            distance_tracs_c += tracs_vs_strawman_gps_tky(traj, epsilon)[2]
            distance_2d_laplace += tracs_vs_strawman_gps_tky(traj, epsilon)[3]
    row = {
        "epsilon": epsilon,
        "tracs_d": distance_tracs_d / 2000,
        "strawman": distance_strawman / 2000,
        "tracs_c": distance_tracs_c / 2000,
        "2d_laplace": distance_2d_laplace / 2000
    }
    rows.append(row)
    print(f"epsilon: {epsilon}, tracs-d: {row['tracs_d']}, strawman: {row['strawman']}, tracs-c: {row['tracs_c']}, 2d-laplace: {row['2d_laplace']}")
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.rcParams["font.size"] = 20
plt.plot([row["epsilon"] for row in rows], [row["2d_laplace"] for row in rows], label="T-Laplace", linestyle="--", color="gray", marker="^")
plt.plot([row["epsilon"] for row in rows], [row["strawman"] for row in rows], label="Strawman", linestyle="--", color="black", marker="x")
plt.plot([row["epsilon"] for row in rows], [row["tracs_d"] for row in rows], label="TraCS-D", linestyle="-.", color="red", marker="o")
plt.plot([row["epsilon"] for row in rows], [row["tracs_c"] for row in rows], label="TraCS-C", linestyle=":", color="blue", marker="s")
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Average error")
plt.legend()
plt.title("Figure 7a")

########
# Figure 7b
########

def load_trajectory():
    """"
    Already normalized to [0,1]x[0,1]
    """
    chi_dataset = Path(__file__).parent.parent / "experiments" / "discrete_space" / "CHI" / "chi_trajectory.pkl"
    with open(chi_dataset, "rb") as f:
        trajectory = pickle.load(f)
    # print(f"Length of trajectory: {len(trajectory)}")
    # print(f"Length of location space: {len(location_space)}") # there is the first 100 trajectories
    return trajectory

def tracs_vs_strawman_gps_chi(traj, epsilon):
    perturbed_trajectory_strawman = strawman(traj, epsilon)
    perturbed_trajectory_tracs_d = tracs_d(traj, epsilon, pi / (pi + 1) * epsilon)
    perturbed_trajectory_tracs_c = tracs_c(traj, epsilon, epsilon / 2)
    perturbed_trajectory_laplace = ldp_2d_laplace_batch(traj, epsilon)
    # transform back to gps coordinates
    # CHI: x_min: 41.60015, x_max: 41.99822, y_min: -87.9952, y_max: -87.50765
    x_min, x_max, y_min, y_max = 41.60015, 41.99822, -87.9952, -87.50765
    gps_traj = unit_square_to_gps(traj, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_strawman = unit_square_to_gps(perturbed_trajectory_strawman, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_d = unit_square_to_gps(perturbed_trajectory_tracs_d, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_tracs_c = unit_square_to_gps(perturbed_trajectory_tracs_c, x_min, x_max, y_min, y_max)
    gps_perturbed_traj_laplace = unit_square_to_gps(perturbed_trajectory_laplace, x_min, x_max, y_min, y_max)

    # print(f"gps_traj: {gps_traj}")
    return (averaged_l2_distance(gps_traj, gps_perturbed_traj_strawman[1:]), # Remove dummy head
            averaged_l2_distance(gps_traj, gps_perturbed_traj_tracs_d),
            averaged_l2_distance(gps_traj, gps_perturbed_traj_tracs_c),
            averaged_l2_distance(gps_traj, gps_perturbed_traj_laplace))

rows = []
for epsilon in [2, 4, 6, 8, 10]:
    distance_tracs_d, distance_strawman, distance_tracs_c, distance_2d_laplace = 0, 0, 0, 0
    set_of_traj = load_trajectory()[:100]  # first 100 trajectories
    for traj in set_of_traj:
        for _ in range(20):
            # average distance
            distance_strawman += tracs_vs_strawman_gps_chi(traj, epsilon)[0]
            distance_tracs_d += tracs_vs_strawman_gps_chi(traj, epsilon)[1]
            distance_tracs_c += tracs_vs_strawman_gps_chi(traj, epsilon)[2]
            distance_2d_laplace += tracs_vs_strawman_gps_chi(traj, epsilon)[3]
    row = {
        "epsilon": epsilon,
        "tracs_d": distance_tracs_d / 2000,
        "strawman": distance_strawman / 2000,
        "tracs_c": distance_tracs_c / 2000,
        "2d_laplace": distance_2d_laplace / 2000
    }
    rows.append(row)
    print(f"epsilon: {epsilon}, tracs-d: {row['tracs_d']}, strawman: {row['strawman']}, tracs-c: {row['tracs_c']}, 2d-laplace: {row['2d_laplace']}")
plt.figure(figsize=(5,5))
plt.rcParams["font.size"] = 20
plt.plot([row["epsilon"] for row in rows], [row["2d_laplace"] for row in rows], label="T-Laplace", linestyle="--", color="gray", marker="^")
plt.plot([row["epsilon"] for row in rows], [row["strawman"] for row in rows], label="Strawman", linestyle="--", color="black", marker="x")
plt.plot([row["epsilon"] for row in rows], [row["tracs_d"] for row in rows], label="TraCS-D", linestyle="-.", color="red", marker="o")
plt.plot([row["epsilon"] for row in rows], [row["tracs_c"] for row in rows], label="TraCS-C", linestyle=":", color="blue", marker="s")
plt.xlabel(r'Privacy parameter $\varepsilon$')
plt.ylabel(r"Average error")
plt.legend()
plt.title("Figure 7b")
plt.show()
