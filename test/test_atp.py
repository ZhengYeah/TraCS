import pytest
import numpy as np
from src.utilities.discrete_location_space import discrete_location_grid
from src.other_methods.tp import tp_bi_direction, merge_traj, tp_perturb

pi = np.pi


@pytest.mark.parametrize("loc_1, loc_2, private_loc, location_space, epsilon", [((0, 0), (1, 1), (0.5, 0.5), discrete_location_grid(10), 2)])
def test_tp_bi_direction(loc_1, loc_2, private_loc, location_space, epsilon):
    perturbed_loc, reduced_loc_space = tp_bi_direction(loc_1, loc_2, private_loc, location_space, epsilon)
    assert perturbed_loc in location_space
    import matplotlib.pyplot as plt
    plt.scatter(*zip(*location_space), c='r')
    plt.scatter(*zip(*reduced_loc_space), c='b')
    plt.scatter(*private_loc, c='g')
    plt.scatter(*perturbed_loc, c='y')
    plt.show()


def test_merge_traj():
    traj_1 = [(0, 2), (1, 3), (2, 4)]
    traj_2 = [(1, 1), (2, 2), (3, 3)]
    location_space = discrete_location_grid(4, x_max=4, y_max=4)
    merged_traj = merge_traj(traj_1, traj_2, location_space)
    assert len(merged_traj) == 3
    assert all([loc in location_space for loc in merged_traj])
    import matplotlib.pyplot as plt
    plt.scatter(*zip(*location_space), c='r')
    plt.plot(*zip(*traj_1), c='b')
    plt.plot(*zip(*traj_2), c='g')
    plt.plot(*zip(*merged_traj), c='y')
    plt.show()


@pytest.mark.parametrize("traj, location_space, epsilon", [([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)], discrete_location_grid(40, x_max=4, y_max=4), 8)])
def test_tp_perturb(traj, location_space, epsilon):
    perturbed_traj = tp_perturb(traj, location_space, epsilon)
    import matplotlib.pyplot as plt
    plt.scatter(*zip(*location_space), c='r')
    plt.plot(*zip(*traj), c='b')
    plt.plot(*zip(*perturbed_traj), c='y')
    plt.show()

