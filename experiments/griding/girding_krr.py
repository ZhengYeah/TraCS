import numpy as np

from src.ldp_mechanisms import DiscreteMechanism
from src.utilities.discrete_location_space import GridLocationSpace
from src.perturbation_tracs import CoordinatePerturbation


def griding_em(true_loc, epsilon, granularity) -> tuple:
    cells = GridLocationSpace(granularity, x_max=1, y_max=1)
    cell_space = cells.get_location_space()
    perturbed_cell = DiscreteMechanism(true_loc, epsilon, len(cell_space)).exp_mechanism_loc(cell_space)
    uniform_in_cell = cells.uniform_sample_from_a_cell(perturbed_cell[0], perturbed_cell[1])
    return uniform_in_cell


def euclidean_distance(loc1, loc2):
    return np.linalg.norm(np.array(loc1) - np.array(loc2))


if __name__ == '__main__':
    true_location = (0.1, 0.1)

    # Write to csv
    with open(f"./griding_em.csv", "a") as f:
        # header
        f.write("epsilon,results_10,results_100,results_tracs_c\n")
        for epsilon in np.linspace(2, 10, 5, endpoint=True):
            results_10 = []
            result_100 = []
            results_tracs_c = []
            for _ in range(200):
                perturbed_loc_10 = griding_em(true_location, epsilon, 10)
                perturbed_loc_100 = griding_em(true_location, epsilon, 100)
                tracs_c = CoordinatePerturbation(true_location, epsilon, epsilon / 2, x_max=1, y_max=1)
                tracs_c.perturb()
                perturbed_loc_tracs_c = tracs_c.perturbed_location
                error_10 = euclidean_distance(true_location, perturbed_loc_10)
                error_100 = euclidean_distance(true_location, perturbed_loc_100)
                error_tracs_c = euclidean_distance(true_location, perturbed_loc_tracs_c)
                results_10.append(error_10)
                result_100.append(error_100)
                results_tracs_c.append(error_tracs_c)
            avg_error_10 = sum(results_10) / len(results_10)
            avg_error_100 = sum(result_100) / len(result_100)
            avg_error_tracs_c = sum(results_tracs_c) / len(results_tracs_c)
            f.write(f"{epsilon},{avg_error_10},{avg_error_100},{avg_error_tracs_c}\n")
            print(f"epsilon: {epsilon}, granularity: 10, average error: {avg_error_10}")
            print(f"epsilon: {epsilon}, granularity: 100, average error: {avg_error_100}")
            print(f"epsilon: {epsilon}, tracs-c average error: {avg_error_tracs_c}")
