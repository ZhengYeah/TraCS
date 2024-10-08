import numpy as np
import random

from src.ldp_mechanisms import PiecewiseMechanism


class DirectionDistancePerturbation:
    def __init__(self, location: tuple, ref_location: tuple, epsilon):
        """
        :param location: represented by a pair of coordinates in [0, 1]^2
        """
        assert len(location) == 2 and len(ref_location) == 2
        self.location = location
        self.ref_location = ref_location
        self.epsilon = epsilon
        self.perturbed_direction = None
        self.perturbed_distance = None

    def direction_perturbation(self):
        """
        perturb the location in the direction
        :return: perturbed location
        """
        # private direction
        phi = np.arctan2(self.location[1] - self.ref_location[1], self.location[0] - self.ref_location[0])
        assert -np.pi <= phi <= np.pi
        # perturb the direction
        perturbation = PiecewiseMechanism(phi, self.epsilon)
        self.perturbed_direction = perturbation.circular_perturbation()




    def distance_at_phi(self, phi):
        """
        calculate the distance at the direction of phi
        :param phi: the angle
        :return: the distance
        """
        return np.dot(self.pertubed_direction, self.location) / np.cos(phi)

    def distance_perturbation(self, epsilon):
        """
        perturb the distance
        :param epsilon: the privacy budget
        :return: the perturbed distance
        """
        return kRR(self.distance, [0, 1], epsilon)



class CoordinatePerturbation:
    def __init__(self, location, epsilon):
        self.location = location
        self.epsilon = epsilon

    def perturb(self):
        """
        perturb the location
        :return: the perturbed location
        """
        perturbation = DiscreteMechanism(self.location, self.epsilon, 6)
        perturbed = perturbation.kRR(self.location, [0, 1, 2, 3, 4, 5], self.epsilon)
        return kRR(self.location, [0, 1], self.epsilon)


