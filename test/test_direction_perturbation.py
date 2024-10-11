import pytest
from src.perturbation_tracs import DirectionDistancePerturbation
import numpy as np

pi = np.pi


def test_direction_perturbation():
    epsilon = 2
    # test case 1
    ref_location, location = (0.5, 0.6), (0.5, 0.6)
    perturbation = DirectionDistancePerturbation(location, ref_location, epsilon)
    perturbation._direction_perturbation()
    print(perturbation.private_direction)
    assert abs(perturbation.private_direction - pi / 4) < 1e-3
    # test case 2
    ref_location, location = (0.5, 0.5), (0.4, 0.7)
    perturbation = DirectionDistancePerturbation(location, ref_location, epsilon)
    perturbation._direction_perturbation()
    assert abs(perturbation.private_direction - 3 * pi / 4) < 1e-3
    # test case 3
    ref_location, location = (0.5, 0.5), (0.4, 0.2)
    perturbation = DirectionDistancePerturbation(location, ref_location, epsilon)
    perturbation._direction_perturbation()
    assert abs(perturbation.private_direction - 5 * pi / 4) < 1e-3
    # test case 4
    ref_location, location = (0.5, 0.5), (0.6, 0.2)
    perturbation = DirectionDistancePerturbation(location, ref_location, epsilon)
    perturbation._direction_perturbation()
    assert abs(perturbation.private_direction - 7 * pi / 4) < 1e-3

