import numpy as np
from src.utilities.discrete_location_space import discrete_location_grid
from src.methods.ngram import ngram_perturb
from src.methods.tp import tp_perturb
from src.methods.wrapped_tracs import tracs_d, tracs_c
from src.utilities.generate_random_traj import generate_random_traj_discrete
from src.utilities.trajectory_distance import averaged_l2_distance

pi = np.pi


def error_tp_ngram(private_traj, location_space, epsilon, theta):
    perturbed_traj_tp = tp_perturb(private_traj, location_space, epsilon)
    perturbed_traj_ngram = ngram_perturb(private_traj, location_space, epsilon, theta)
    error_tp = averaged_l2_distance(private_traj, perturbed_traj_tp)
    error_ngram = averaged_l2_distance(private_traj, perturbed_traj_ngram)
    return error_tp, error_ngram


def error_tracs(private_traj, epsilon, epsilon_d, epsilon_1, x_max, y_max, granularity):
    perturbed_traj_tracs_d = tracs_d(private_traj, epsilon, epsilon_d, x_max, y_max)
    perturbed_traj_tracs_c = tracs_c(private_traj, epsilon, epsilon_1, x_max, y_max)
    # floor the perturbed locations
    perturbed_traj_tracs_d = [(int(x * granularity) / granularity, int(y * granularity) / granularity) for x, y in perturbed_traj_tracs_d]
    perturbed_traj_tracs_c = [(int(x * granularity) / granularity, int(y * granularity) / granularity) for x, y in perturbed_traj_tracs_c]
    # assert perturbed_traj_tracs_d[0] in discrete_location_grid(granularity, x_max, y_max)
    error_tracs_d = averaged_l2_distance(private_traj, perturbed_traj_tracs_d)
    error_tracs_c = averaged_l2_distance(private_traj, perturbed_traj_tracs_c)
    return error_tracs_d, error_tracs_c


if __name__ == '__main__':
    x_max, y_max = 1, 1
    granularity = 60
    # generate random trajectory
    location_space = discrete_location_grid(granularity, x_max, y_max)
    private_traj = generate_random_traj_discrete(100, granularity, x_max, y_max)
    # write to csv
    with open(f"./results/experiment_1_2.csv", "a") as f:
        # header
        f.write("epsilon,tp,ngram,tracs_d,tracs_c\n")
        for epsilon in [2, 4, 6, 8, 10]:
            epsilon_d, epsilon_1 = pi / (pi + 1) * epsilon, epsilon / 2
            error_tp, error_ngram = 0, 0
            error_tracs_d, error_tracs_c = 0, 0
            for _ in range(10):
                error_tp += error_tp_ngram(private_traj, location_space, epsilon, theta=x_max * 0.75)[0]
                error_ngram += error_tp_ngram(private_traj, location_space, epsilon, theta=x_max * 0.75)[1]
                error_tracs_d += error_tracs(private_traj, epsilon, epsilon_d, epsilon_1, x_max, y_max, granularity)[0]
                error_tracs_c += error_tracs(private_traj, epsilon, epsilon_d, epsilon_1, x_max, y_max, granularity)[1]
            error_tp, error_ngram = error_tp / 10, error_ngram / 10
            error_tracs_d, error_tracs_c = error_tracs_d / 10, error_tracs_c / 10
            print(f"epsilon: {epsilon}, tp: {error_tp}, ngram: {error_ngram}, tracs-d: {error_tracs_d}, tracs-c: {error_tracs_c}")
            f.write(f"{epsilon},{error_tp},{error_ngram},{error_tracs_d},{error_tracs_c}\n")
