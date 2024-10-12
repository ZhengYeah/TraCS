from src.perturbation_tracs import DirectionDistancePerturbation
import numpy as np

pi = np.pi


def test_direction_perturbation():
    location = (0.5, 0.5)
    ref_location = (0.5, 0.5)
    epsilon = 1
    perturbation = DirectionDistancePerturbation(location, ref_location, epsilon)
    perturbation._direction_perturbation()
    assert -np.pi <= perturbation.private_direction <= np.pi
    assert 0 <= perturbation.perturbed_direction <= 2 * np.pi


def test_distance_perturbation():
    pass

