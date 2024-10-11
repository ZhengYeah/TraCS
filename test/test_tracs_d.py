import pytest
from src.perturbation_tracs import DirectionDistancePerturbation
import numpy as np

pi = np.pi


def test_direction_perturbation():
    epsilon = 2
    # test case 1
    ref_location, location = (0.5, 0.5), (0.6, 0.6)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon)
    perturbation._direction_perturbation()
    print(perturbation.private_direction)
    assert abs(perturbation.private_direction - pi / 4) < 1e-3
    # test case 2
    ref_location, location = (0.5, 0.5), (0.4, 0.7)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon)
    perturbation._direction_perturbation()
    assert abs(perturbation.private_direction - (pi - np.arctan(2))) < 1e-3
    # test case 3
    ref_location, location = (0.5, 0.5), (0.4, 0.2)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon)
    perturbation._direction_perturbation()
    assert abs(perturbation.private_direction - (pi + np.arctan(3))) < 1e-3
    # test case 4
    ref_location, location = (0.5, 0.5), (0.6, 0.2)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon)
    perturbation._direction_perturbation()
    assert abs(perturbation.private_direction - (2 * pi - np.arctan(3))) < 1e-3


def test_distance_perturbation():
    epsilon = 2
    # test case 1
    ref_location, location = (0.5, 0.5), (0.6, 0.6)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon)
    perturbation._distance_perturbation()
    assert abs(perturbation.perturbed_distance - 0.1 * np.sqrt(2)) < 1e-3
    # test case 2
    ref_location, location = (0.5, 0.5), (0.4, 0.7)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon)
    perturbation._distance_perturbation()
    assert abs(perturbation.perturbed_distance - 0.1 * np.sqrt(5)) < 1e-3
    # test case 3
    ref_location, location = (0.5, 0.5), (0.4, 0.2)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon)
    perturbation._distance_perturbation()
    assert abs(perturbation.perturbed_distance - 0.3 * np.sqrt(5)) < 1e-3
    # test case 4
    ref_location, location = (0.5, 0.5), (0.6, 0.2)
    perturbation = DirectionDistancePerturbation(ref_location, location, epsilon)
    perturbation._distance_perturbation()
    assert abs(perturbation.perturbed_distance - 0.3 * np.sqrt(5)) < 1e-3
