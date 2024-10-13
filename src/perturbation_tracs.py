import numpy as np
from src.ldp_mechanisms import PiecewiseMechanism

pi = np.pi


class DirectionDistancePerturbation:
    def __init__(self, ref_location: tuple, location: tuple, epsilon):
        """
        :param location: represented by a pair of coordinates in [0, 1]^2
        """
        assert len(location) == 2 and len(ref_location) == 2
        self.ref_location = ref_location
        self.location = location
        self.perturbed_location = None
        self.epsilon = epsilon
        self.private_direction = None
        self.private_distance_space = None
        self.perturbed_direction = None

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
        # arctan2 has a range of [-pi, pi], so we need to convert it to [0, 2pi)
        self.private_direction = np.arctan2(self.location[1] - self.ref_location[1], self.location[0] - self.ref_location[0])
        if self.private_direction < 0:
            self.private_direction += 2 * pi
        assert 0 <= self.private_direction <= 2 * pi
        # perturb the direction
        perturbation = PiecewiseMechanism(self.private_direction, self.epsilon / 2)
        self.perturbed_direction = perturbation.circular_perturbation()

    # def _distance_perturbation(self):
    #     """
    #     perturb the distance (r(\varphi) to r'(\varphi') in the paper)
    #     """
    #     # arctan2 has a range of [-pi, pi], so we need to convert it to [0, 2pi)
    #     phi_1 = np.arctan2(1 - self.ref_location[1], 1 - self.ref_location[0])
    #     phi_2 = np.arctan2(1 - self.ref_location[1], -self.location[0])
    #     phi_3 = np.arctan2(-self.ref_location[1], -self.ref_location[0]) + 2 * pi
    #     phi_4 = np.arctan2(-self.location[1], 1 - self.location[0]) + 2 * pi
    #     if 0 <= self.private_direction < phi_1:
    #         self.private_distance_space = (1 - self.ref_location[0]) / np.cos(self.private_direction)
    #     elif phi_1 <= self.private_direction < phi_2:
    #         self.private_distance_space = (1 - self.ref_location[1]) / np.sin(self.private_direction)
    #     elif phi_2 <= self.private_direction < phi_3:
    #         self.private_distance_space = self.ref_location[0] / -np.cos(self.private_direction)
    #     elif phi_3 <= self.private_direction < phi_4:
    #         self.private_distance_space = self.ref_location[1] / -np.sin(self.private_direction)
    #     elif phi_4 <= self.private_direction < 2 * pi:
    #         self.private_distance_space = (1 - self.ref_location[0]) / np.cos(self.private_direction)
    #     else:
    #         raise ValueError("The private direction is out of range")
    #
    #     # normalize the private distance
    #     private_distance = np.sqrt((self.location[0] - self.ref_location[0]) ** 2 + (self.location[1] - self.ref_location[1]) ** 2)
    #     normalized_private_distance = private_distance / self.private_distance_space
    #     assert 0 <= normalized_private_distance <= 1
    #     # perturb the distance
    #     perturbation = PiecewiseMechanism(normalized_private_distance, self.epsilon / 2)
    #     normalized_perturbed_distance = perturbation.linear_perturbation()
    #
    #     # denormalize the perturbed distance according to the perturbed direction
    #     if 0 <= self.perturbed_direction < phi_1:
    #         perturbed_distance_space = (1 - self.ref_location[0]) / np.cos(self.perturbed_direction)
    #     elif phi_1 <= self.perturbed_direction < phi_2:
    #         perturbed_distance_space = (1 - self.ref_location[1]) / np.sin(self.perturbed_direction)
    #     elif phi_2 <= self.perturbed_direction < phi_3:
    #         perturbed_distance_space = self.ref_location[0] / -np.cos(self.perturbed_direction)
    #     elif phi_3 <= self.perturbed_direction < phi_4:
    #         perturbed_distance_space = self.ref_location[1] / -np.sin(self.perturbed_direction)
    #     elif phi_4 <= self.perturbed_direction < 2 * pi:
    #         perturbed_distance_space = (1 - self.ref_location[0]) / np.cos(self.perturbed_direction)
    #     else:
    #         raise ValueError("The perturbed direction is out of range")
    #     self.perturbed_distance = normalized_perturbed_distance * perturbed_distance_space

    @staticmethod
    def get_distance_space(ref_location, direction):
        # use parametric equations of the line to calculate the distance space
        tmp_right, tmp_left, tmp_top, tmp_bottom = np.inf, np.inf, np.inf, np.inf
        vec_direction = (np.cos(direction), np.sin(direction))
        if vec_direction[0] >= 0:
            tmp_right = (1 - ref_location[0]) / vec_direction[0]
        if vec_direction[0] < 0:
            tmp_left = ref_location[0] / -vec_direction[0]
        if vec_direction[1] >= 0:
            tmp_top = (1 - ref_location[1]) / vec_direction[1]
        if vec_direction[1] < 0:
            tmp_bottom = ref_location[1] / -vec_direction[1]
        # select the minimum distance with parameter >= 0 (meaningful distance)
        tmp = [tmp_right, tmp_left, tmp_top, tmp_bottom]
        tmp = [i for i in tmp if i >= 0]
        distance_space = min(tmp)
        assert 0 <= distance_space <= np.sqrt(2)
        return distance_space

    def _distance_perturbation(self):
        """
        perturb the distance (r(\varphi) to r'(\varphi') in the paper)
        """
        self.private_distance_space = self.get_distance_space(self.ref_location, self.private_direction)
        # normalize the private distance
        private_distance = np.sqrt((self.location[0] - self.ref_location[0]) ** 2 + (self.location[1] - self.ref_location[1]) ** 2)
        normalized_private_distance = private_distance / self.private_distance_space
        assert 0 <= normalized_private_distance <= 1
        # perturb the distance
        perturbation = PiecewiseMechanism(normalized_private_distance, self.epsilon / 2)
        normalized_perturbed_distance = perturbation.linear_perturbation()

        # denormalize the perturbed distance according to the perturbed direction
        perturbed_distance_space = self.get_distance_space(self.ref_location, self.perturbed_direction)
        assert 0 <= perturbed_distance_space <= np.sqrt(2)
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


# if __name__ == "__main__":
#     epsilon = 2
#     # test case 1
#     ref_location, location = (0.8736161875439702, 0.31987568053095894), (0.4309125951493439, 0.9731022863296187)
#     perturbation = DirectionDistancePerturbation(ref_location, location, epsilon)
#     perturbation._direction_perturbation()
#     perturbation._distance_perturbation()
#     print(perturbation.private_direction, perturbation.private_distance_space, perturbation.perturbed_direction, perturbation.perturbed_distance)
