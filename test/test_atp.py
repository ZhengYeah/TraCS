import pytest
import numpy as np
from src.perturbation_tracs import CoordinatePerturbation
from src.utilities.discrete_location_space import discrete_location_grid
from src.other_methods.tp import tp_bi_direction, merge_traj, tp_perturb

pi = np.pi


@pytest.mark.parametrize("loc_1, loc_2, private_loc, location_space, epsilon", [((0, 0), (1, 1), (0.5, 0.5), discrete_location_grid(10), 5)])
def test_tp_bi_direction(loc_1, loc_2, private_loc, location_space, epsilon):
    perturbed_loc, reduced_loc_space = tp_bi_direction(loc_1, loc_2, private_loc, location_space, epsilon)
    assert perturbed_loc in location_space
    import matplotlib.pyplot as plt
    plt.scatter(*zip(*reduced_loc_space), c='b')
    plt.scatter(*zip(*location_space), c='r')
    plt.scatter(*private_loc, c='g')
    plt.scatter(*perturbed_loc, c='y')
    plt.show()



