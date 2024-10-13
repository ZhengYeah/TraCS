import numpy as np
from src.ldp_mechanisms import DiscreteMechanism
from src.perturbation_tracs import DirectionDistancePerturbation

pi = np.pi


def strawman_perturbation(ref_location: tuple, location: tuple, epsilon):
    epsilon_d = pi / (pi + 1) * epsilon
    direction_distance = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d)
    direction_distance._direction_perturbation()
    private_sector = direction_distance.private_direction // (pi / 3)
    assert 0 <= private_sector <= 5
    # direction perturbation
    mechanism = DiscreteMechanism(private_sector, epsilon_d, 6)
    perturbed_sector = mechanism.krr()
    uniform_sample = np.random.uniform(0, 1)
    perturbed_direction = perturbed_sector * (pi / 3) + uniform_sample * (pi / 3)
    # distance perturbation
    direction_distance.perturbed_direction = perturbed_direction
    direction_distance._distance_perturbation()
    perturbed_distance = direction_distance.perturbed_distance

    x = ref_location[0] + perturbed_distance * np.cos(perturbed_direction)
    y = ref_location[1] + perturbed_distance * np.sin(perturbed_direction)
    assert 0 <= x <= 1 and 0 <= y <= 1
    return x, y

