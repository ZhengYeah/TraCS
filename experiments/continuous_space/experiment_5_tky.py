import pickle
import numpy as np
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
    with open(f"../discrete_space/TKY/tky_trajectory.pkl", "rb") as f:
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


if __name__ == '__main__':
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
        print(f"epsilon: {epsilon}, tracs-d: {distance_tracs_d / 2000}, strawman: {distance_strawman / 2000}, tracs-c: {distance_tracs_c / 2000}, 2d-laplace: {distance_2d_laplace / 2000}")
        # save to csv
        with open(f"./results/experiment_5_tky.csv", "a") as f:
            # header
            if epsilon == 2:
                # # clear the file
                # f.seek(0)
                # f.truncate()
                f.write("epsilon,tracs_d,strawman,tracs_c,2d_laplace\n")
            f.write(f"{epsilon},{distance_tracs_d / 2000},{distance_strawman / 2000},{distance_tracs_c / 2000},{distance_2d_laplace / 2000}\n")
