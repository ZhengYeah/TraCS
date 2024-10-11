import numpy as np
from src.ldp_mechanisms import PiecewiseMechanism

pi = np.pi


class DirectionDistancePerturbation:
    def __init__(self, ref_location: tuple, location: tuple, epsilon):
        """
        :param location: represented by a pair of coordinates in [0, 1]^2
        """
        assert len(location) == 2 and len(ref_location) == 2
        self.location = location
        self.perturbed_location = None
        self.ref_location = ref_location
        self.epsilon = epsilon
        self.private_direction = None
        self.private_distance = None
        self.perturbed_direction = None
        self.perturbed_distance = None

    # private methods
    def _direction_perturbation(self):
        """
        perturb the direction (\varphi to \varphi' in the paper)
        """
        # conor case
        if self.location == self.ref_location:
            self.perturbed_direction = 0
            return
        # private direction
        # arc tangent has a valid range of [0, \pi / 2]
        # so there are four cases, depending on the location of the reference and the private location
        if self.location[0] >= self.ref_location[0] and self.location[1] >= self.ref_location[1]:
            self.private_direction = np.arctan2(self.location[1] - self.ref_location[1], self.location[0] - self.ref_location[0])
        elif self.location[0] < self.ref_location[0] and self.location[1] >= self.ref_location[1]:
            self.private_direction = pi - np.arctan2(self.location[1] - self.ref_location[1], self.ref_location[0] - self.location[0])
        elif self.location[0] < self.ref_location[0] and self.location[1] < self.ref_location[1]:
            self.private_direction = pi + np.arctan2(self.ref_location[1] - self.location[1], self.ref_location[0] - self.location[0])
        elif self.location[0] >= self.ref_location[0] and self.location[1] < self.ref_location[1]:
            self.private_direction = 2 * pi - np.arctan2(self.ref_location[1] - self.location[1], self.location[0] - self.ref_location[0])
        else:
            raise ValueError("Invalid private direction")
        assert 0 <= self.private_direction <= 2 * pi
        # perturb the direction
        perturbation = PiecewiseMechanism(self.private_direction, self.epsilon / 2)
        self.perturbed_direction = perturbation.circular_perturbation()

    def _distance_perturbation(self):
        """
        perturb the distance (r(\varphi) to r'(\varphi') in the paper)
        """
        # arc tangent of the private direction, necessary to use only the [0, 2\pi) range
        phi_1 = np.arctan2(1 - self.ref_location[1], 1 - self.ref_location[0])
        phi_2 = pi - np.arctan2(1 - self.ref_location[1], self.location[0])
        phi_3 = pi + np.arctan2(self.ref_location[1], self.ref_location[0])
        phi_4 = 2 * pi - np.arctan2(self.location[1], 1 - self.location[0])
        if 0 <= self.private_direction < phi_1:
            private_distance_space = (1 - self.ref_location[0]) / np.cos(self.private_direction)
        elif phi_1 <= self.private_direction < phi_2:
            private_distance_space = (1 - self.ref_location[1]) / np.sin(self.private_direction)
        elif phi_2 <= self.private_direction < phi_3:
            private_distance_space = self.ref_location[0] / np.cos(self.private_direction)
        elif phi_3 <= self.private_direction < phi_4:
            private_distance_space = self.ref_location[1] / np.sin(self.private_direction)
        else:
            raise ValueError("The private direction is out of range")

        # normalize the private distance
        normalized_private_distance = self.private_distance / private_distance_space
        assert 0 <= normalized_private_distance <= 1
        # perturb the distance
        perturbation = PiecewiseMechanism(normalized_private_distance, self.epsilon / 2)
        normalized_perturbed_distance = perturbation.linear_perturbation()

        # denormalize the perturbed distance according to the perturbed direction
        if 0 <= self.perturbed_direction < phi_1:
            perturbed_distance_space = (1 - self.ref_location[0]) / np.cos(self.perturbed_direction)
        elif phi_1 <= self.perturbed_direction < phi_2:
            perturbed_distance_space = (1 - self.ref_location[1]) / np.sin(self.perturbed_direction)
        elif phi_2 <= self.perturbed_direction < phi_3:
            perturbed_distance_space = self.ref_location[0] / np.cos(self.perturbed_direction)
        elif phi_3 <= self.perturbed_direction < phi_4:
            perturbed_distance_space = self.ref_location[1] / np.sin(self.perturbed_direction)
        else:
            raise ValueError("The perturbed direction is out of range")
        self.perturbed_distance = normalized_perturbed_distance * perturbed_distance_space

    def perturb(self):
        """
        encapsulate the perturbation process
        """
        self._direction_perturbation()
        self._distance_perturbation()
        x = self.ref_location[0] + self.perturbed_distance * np.cos(self.perturbed_direction)
        y = self.ref_location[1] + self.perturbed_distance * np.sin(self.perturbed_direction)
        assert 0 <= x <= 1 and 0 <= y <= 1
        self.perturbed_location = (x, y)


class CoordinatePerturbation:
    def __init__(self, location: tuple, epsilon):
        """
        :param location: represented by a pair of coordinates in [0, 1]^2
        """
        assert len(location) == 2
        self.location = location
        self.perturbed_location = None
        self.epsilon = epsilon

    def perturb(self):
        """
        perturb the coordinates of the location
        """
        perturbation = PiecewiseMechanism(self.location[0], self.epsilon)
        x = perturbation.linear_perturbation()
        perturbation = PiecewiseMechanism(self.location[1], self.epsilon)
        y = perturbation.linear_perturbation()
        self.perturbed_location = (x, y)


if __name__ == "__main__":
    ref_location, location = (0.5, 0.5), (0.6, 0.6)
    epsilon = 2
    perturbation = DirectionDistancePerturbation(location, ref_location, epsilon)
    perturbation._direction_perturbation()
    print(perturbation.private_direction)
