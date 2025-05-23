import pytest
from src.methods.srr_3_groups import srr_perturb
import numpy as np


def test_srr():
    epsilon = 2
    # [0,1] grid
    location_space = [(i / 10, j / 10) for i in range(11) for j in range(11)]
    input_x = [(0.5, 0.5) for _ in range(1000)]
    distance_list = [0.3, 0.6]
    perturbed_locations = srr_perturb(input_x, location_space, epsilon, distance_list)
    # draw the histogram of perturbed locations into each group
    num_group_0 = num_group_1 = num_group_2 = 0
    for i in range(len(perturbed_locations)):
        if np.sqrt((input_x[i][0] - perturbed_locations[i][0]) ** 2 + (input_x[i][1] - perturbed_locations[i][1]) ** 2) <= distance_list[0]:
            num_group_0 += 1
        elif np.sqrt((input_x[i][0] - perturbed_locations[i][0]) ** 2 + (input_x[i][1] - perturbed_locations[i][1]) ** 2) <= distance_list[1]:
            num_group_1 += 1
        else:
            num_group_2 += 1
    print("\n")
    print("Group 0: ", num_group_0)
    print("Group 1: ", num_group_1)
    print("Group 2: ", num_group_2)
    # it's true num_group_1 > num_group_0, as there are too much more points in group 1,
    # although the probability of group 0 is higher than group 1
