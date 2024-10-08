import numpy as np
import random

from src.ldp_mechanisms import DiscreteMechanism


class DirectionDistancePerturbation:
    def __init__(self, location, ref_location):
        self.location = location
        self.ref_location = ref_location
        self.direction = None
        self.distance = None

    def direction_perturbation(self):
        """
        perturb the location in the direction
        :return: perturbed location
        """
        self.location = self.location + self.direction * self.distance
        return self.location

    def distance_at_phi(self, phi):
        """
        calculate the distance at the direction of phi
        :param phi: the angle
        :return: the distance
        """
        return np.dot(self.direction, self.location) / np.cos(phi)

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


