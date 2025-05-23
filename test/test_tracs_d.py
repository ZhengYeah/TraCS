import pytest
from src.perturbation_tracs import DirectionDistancePerturbation
import numpy as np

pi = np.pi


def test_direction_perturbation():
    epsilon = 2
    epsilon_d = pi / (pi + 1) * epsilon
    # test case 1
    ref_location, location = (0.5, 0.5), (0.6, 0.6)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d)
    perturbation._direction_perturbation()
    print(perturbation.private_direction)
    assert abs(perturbation.private_direction - pi / 4) < 1e-3
    # test case 2
    ref_location, location = (0.5, 0.5), (0.4, 0.7)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d)
    perturbation._direction_perturbation()
    assert abs(perturbation.private_direction - (pi - np.arctan(2))) < 1e-3
    # test case 3
    ref_location, location = (0.5, 0.5), (0.4, 0.2)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d)
    perturbation._direction_perturbation()
    assert abs(perturbation.private_direction - (pi + np.arctan(3))) < 1e-3
    # test case 4
    ref_location, location = (0.5, 0.5), (0.6, 0.2)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d)
    perturbation._direction_perturbation()
    assert abs(perturbation.private_direction - (2 * pi - np.arctan(3))) < 1e-3


def test_distance_perturbation():
    epsilon = 2
    epsilon_d = pi / (pi + 1) * epsilon
    # test case 1
    ref_location, location = (0.5, 0.5), (0.6, 0.6)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d)
    perturbation._direction_perturbation()
    perturbation._distance_perturbation()
    assert abs(perturbation.private_distance_space - 0.5 * np.sqrt(2)) < 1e-3
    # test case 2
    ref_location, location = (0.5, 0.5), (0.4, 0.7)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d)
    perturbation._direction_perturbation()
    perturbation._distance_perturbation()
    assert abs(perturbation.private_distance_space - np.sqrt(5) / 4) < 1e-3
    # test case 3
    ref_location, location = (0.5, 0.5), (0.4, 0.2)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d)
    perturbation._direction_perturbation()
    perturbation._distance_perturbation()
    assert abs(perturbation.private_distance_space - np.sqrt(10) / 6) < 1e-3
    # test case 4
    ref_location, location = (0.5, 0.5), (0.6, 0.2)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d)
    perturbation._direction_perturbation()
    perturbation._distance_perturbation()
    assert abs(perturbation.private_distance_space - np.sqrt(10) / 6) < 1e-3
    # test case 5
    ref_location, location = (1, 1), (2, 2)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d, 2, 3)
    perturbation._direction_perturbation()
    perturbation._distance_perturbation()
    assert abs(perturbation.private_distance_space - np.sqrt(2)) < 1e-3

    # random test
    for _ in range(1000):
        ref_location = (np.random.rand(), np.random.rand())
        location = (np.random.rand(), np.random.rand())
        perturbation = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d, 2, 3)
        perturbation._direction_perturbation()
        perturbation._distance_perturbation()
        # print(ref_location, location, perturbation.private_distance_space)
        assert 0 <= perturbation.private_distance_space <= np.sqrt(2 ** 2 + 3 ** 2)


@pytest.mark.parametrize("ref_location, location, epsilon", [((0.5, 0.5), (1.9, 2.5), 6),])
def test_perturb(ref_location, location, epsilon):
    # collect the perturbed locations
    set_perturbed_location = set()
    epsilon_d = pi / (pi + 1) * epsilon
    for _ in range(100):
        perturbation = DirectionDistancePerturbation(ref_location, location, epsilon, epsilon_d, 2, 3)
        perturbation.perturb()
        set_perturbed_location.add(perturbation.perturbed_location)
    # draw the perturbed locations
    import matplotlib.pyplot as plt
    plot = plt.figure()
    ax = plot.add_subplot(111)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 3)
    for perturbed_location in set_perturbed_location:
        ax.plot(*perturbed_location, 'ro')  # packing the tuple
    ax.plot(*location, 'bo')
    ax.plot(*ref_location, 'go')
    plt.show()
