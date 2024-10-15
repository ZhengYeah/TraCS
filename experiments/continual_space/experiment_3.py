import numpy as np
from src.ldp_mechanisms import PiecewiseMechanism
from copy import deepcopy
from src.perturbation_tracs import DirectionDistancePerturbation
from src.utilities.generate_random_traj import generate_random_traj
from src.utilities.trajectory_distance import averaged_l2_distance

pi = np.pi


class DirectionDistanceSW:
    """
    Redesigned SW mechanism for direction and distance perturbation
    """
    def __init__(self, ref_location: tuple, location: tuple, epsilon, epsilon_d, x_max=1, y_max=1):
        """
        :param location: represented by a pair of coordinates in [0, 1]^2
        """
        assert len(location) == 2 and len(ref_location) == 2
        self.ref_location = ref_location
        self.location = location
        self.perturbed_location = None
        self.epsilon, self.epsilon_d = epsilon, epsilon_d
        self.private_direction = None
        self.private_distance_space = None
        self.perturbed_direction = None
        self.x_max, self.y_max = x_max, y_max

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
        assert 0 <= self.private_direction <= 2 * pi + 1e-6
        # perturb the direction
        perturbation = PiecewiseMechanism(self.private_direction, self.epsilon_d)
        self.perturbed_direction = perturbation.sw_circular()

    @staticmethod
    def get_distance_space(ref_location, direction, x_max=1, y_max=1):
        # use parametric equations of the line to calculate the distance space
        tmp_right, tmp_left, tmp_top, tmp_bottom = np.inf, np.inf, np.inf, np.inf
        vec_direction = (np.cos(direction), np.sin(direction))
        if vec_direction[0] >= 0:
            tmp_right = (x_max - ref_location[0]) / vec_direction[0]
        if vec_direction[0] < 0:
            tmp_left = ref_location[0] / -vec_direction[0]
        if vec_direction[1] >= 0:
            tmp_top = (y_max - ref_location[1]) / vec_direction[1]
        if vec_direction[1] < 0:
            tmp_bottom = ref_location[1] / -vec_direction[1]
        # select the minimum distance with parameter >= 0 (meaningful distance)
        tmp = [tmp_right, tmp_left, tmp_top, tmp_bottom]
        tmp = [i for i in tmp if i >= 0]
        distance_space = min(tmp)
        assert 0 <= distance_space <= np.sqrt(x_max ** 2 + y_max ** 2)
        return distance_space

    def _distance_perturbation(self):
        """
        perturb the distance (r(\varphi) to r'(\varphi') in the paper)
        """
        self.private_distance_space = self.get_distance_space(self.ref_location, self.private_direction, self.x_max, self.y_max)
        # normalize the private distance
        private_distance = np.sqrt((self.location[0] - self.ref_location[0]) ** 2 + (self.location[1] - self.ref_location[1]) ** 2)
        normalized_private_distance = private_distance / self.private_distance_space
        assert 0 <= normalized_private_distance <= 1 + 1e-6
        # perturb the distance
        perturbation = PiecewiseMechanism(normalized_private_distance, self.epsilon - self.epsilon_d)
        normalized_perturbed_distance = perturbation.sw_linear()

        # denormalize the perturbed distance according to the perturbed direction
        perturbed_distance_space = self.get_distance_space(self.ref_location, self.perturbed_direction, self.x_max, self.y_max)
        assert 0 <= perturbed_distance_space <= np.sqrt(self.x_max ** 2 + self.y_max ** 2)
        self.perturbed_distance = normalized_perturbed_distance * perturbed_distance_space

    def perturb(self):
        """
        encapsulate the perturbation process
        """
        self._direction_perturbation()
        self._distance_perturbation()
        x = self.ref_location[0] + self.perturbed_distance * np.cos(self.perturbed_direction)
        y = self.ref_location[1] + self.perturbed_distance * np.sin(self.perturbed_direction)
        assert 0 <= x <= self.x_max and 0 <= y <= self.y_max
        self.perturbed_location = (x, y)


def tracs_vs_sw(epsilon, num_locations):
    """
    comparison of TRACS-D and re-designed SW on a random trajectory with n_points points
    """
    trajectory = generate_random_traj(num_locations)
    # tracs-d and sw
    # add a dummy ref location to the head of the trajectory
    trajectory = np.vstack((np.array([0, 0]), trajectory))
    perturbed_trajectory_tracs_d = deepcopy(trajectory)
    perturbed_trajectory_sw = deepcopy(trajectory)
    for i in range(num_locations):
        # tracs-d
        ref_location_1 = (perturbed_trajectory_tracs_d[i][0], perturbed_trajectory_tracs_d[i][1])
        location_1 = (perturbed_trajectory_tracs_d[i + 1][0], perturbed_trajectory_tracs_d[i + 1][1])
        epsilon_d = pi / (pi + 1) * epsilon
        tracs_d = DirectionDistancePerturbation(ref_location_1, location_1, epsilon, epsilon_d)
        tracs_d.perturb()
        perturbed_trajectory_tracs_d[i + 1] = tracs_d.perturbed_location
        # strawman
        ref_location_2 = (perturbed_trajectory_sw[i][0], perturbed_trajectory_sw[i][1])
        location_2 = (perturbed_trajectory_sw[i + 1][0], perturbed_trajectory_sw[i + 1][1])
        sw = DirectionDistanceSW(ref_location_2, location_2, epsilon, epsilon_d)
        sw.perturb()
        perturbed_trajectory_sw[i + 1] = sw.perturbed_location
    return averaged_l2_distance(trajectory, perturbed_trajectory_tracs_d), averaged_l2_distance(trajectory, perturbed_trajectory_sw)


if __name__ == '__main__':
    for epsilon in [2, 4, 6, 8, 10]:
        distance_tracs_d, distance_sw = 0, 0
        for _ in range(1000):
            # average distance
            distance_tracs_d += tracs_vs_sw(epsilon, 100)[0]
            distance_sw += tracs_vs_sw(epsilon, 100)[1]
        print(f"epsilon: {epsilon}, tracs-d: {distance_tracs_d / 1000}, sw: {distance_sw / 1000}")
        # # save to csv
        # with open(f"./results/experiment_1_1.csv", "a") as f:
        #     # header
        #     if epsilon == 2:
        #         # # clear the file
        #         # f.seek(0)
        #         # f.truncate()
        #         f.write("epsilon,tracs_d,strawman,tracs_c\n")
        #     f.write(f"{epsilon},{distance_tracs_d / 1000},{distance_strawman / 1000},{distance_tracs_c / 1000}\n")
