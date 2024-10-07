import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import random

Path = 'traj_LDP_submission_support/data/NYC/'
# Path='traj_LDP_submission_support/data/CHI/'
# Path='traj_LDP_submission_support/data/CLE/'
# Path='traj_LDP_submission_support/data/CPS/'

with open(Path + 'traj_list.pickle', 'rb') as traj_list_read:
    traj_list = pickle.load(traj_list_read)

with open(Path + 'longitude_dict_file.pkl', 'rb') as longitude_dict_read:
    longitude_dict = pickle.load(longitude_dict_read)

with open(Path + 'latitude_dict_file.pkl', 'rb') as latitude_dict_read:
    latitude_dict = pickle.load(latitude_dict_read)

with open(Path + 'str2index_dict_file.pickle', 'rb') as str2index_dict_read:
    str2index_dict = pickle.load(str2index_dict_read)

with open(Path + 'index2str_dict_file.pickle', 'rb') as index2str_dict_read:
    index2str_dict = pickle.load(index2str_dict_read)

with open(Path + 'np_poi_lat_lon_distance_matrix_file.pickle', 'rb') as poi_distance_matrix_read:
    poi_distance_matrix = pickle.load(poi_distance_matrix_read)

with open(Path + 'poi_angel_num_matrix_6_file.pickle', 'rb') as poi_angel_num_matrix_6_read:
    poi_angel_num_matrix_6 = pickle.load(poi_angel_num_matrix_6_read)

with open(Path + 'poi_angel_num_matrix_4_file.pickle', 'rb') as poi_angel_num_matrix_4_read:
    poi_angel_num_matrix_4 = pickle.load(poi_angel_num_matrix_4_read)

with open(Path + 'poi_angel_num_matrix_2_file.pickle', 'rb') as poi_angel_num_matrix_2_read:
    poi_angel_num_matrix_2 = pickle.load(poi_angel_num_matrix_2_read)

with open(Path + 'poi_angel_num_matrix_12_file.pickle', 'rb') as poi_angel_num_matrix_12_read:
    poi_angel_num_matrix_12 = pickle.load(poi_angel_num_matrix_12_read)

with open(Path + 'traj_center_list_file.pickle', 'rb') as traj_center_list_read:
    traj_center_list = pickle.load(traj_center_list_read)

delta_poi_distance_matrix = poi_distance_matrix.max()


def get_reversed_num_direc_of_traj(perturbed_direc_of_traj, direc_num=4):
    assert direc_num in [2, 4, 6, 12]
    reversed_perturbed_num_direc_of_traj = np.array(perturbed_direc_of_traj)
    if direc_num == 4:
        reversed_perturbed_num_direc_of_traj = (reversed_perturbed_num_direc_of_traj + 2) % 4
    elif direc_num == 2:
        reversed_perturbed_num_direc_of_traj = (reversed_perturbed_num_direc_of_traj + 1) % 2
    elif direc_num == 6:
        reversed_perturbed_num_direc_of_traj = (reversed_perturbed_num_direc_of_traj + 3) % 6
    elif direc_num == 12:
        reversed_perturbed_num_direc_of_traj = (reversed_perturbed_num_direc_of_traj + 6) % 12
    else:
        print('get_reversed_num_direc_of_traj wrong!')
    reversed_perturbed_num_direc_of_traj = reversed_perturbed_num_direc_of_traj.tolist()

    return reversed_perturbed_num_direc_of_traj


def kRR(value, vals, epsilon):
    """
    the k-random response
    :param value: current value
    :param vals: the possible value
    :param epsilon: privacy budget
    :return:
    """
    values = vals.copy()
    p = np.e ** epsilon / (np.e ** epsilon + len(values) - 1)
    if np.random.random() < p:
        return value
    values.remove(value)
    return values[np.random.randint(low=0, high=len(values))]


def square_wave_mechanism(poi_idx, R, epsilon):
    """
    :param poi_idx: the index of the current poi
    :param R: max trajectory radius
    :return: perturbed R
    """
    delta = max(poi_distance_matrix[poi_idx])
    t = R / delta

    b = (epsilon * (np.e ** (epsilon)) - np.e ** epsilon + 1) / (
                2 * np.e ** (epsilon) * (np.e ** epsilon - 1 - epsilon))
    x = random.uniform(0, 1)
    if x < (2 * b * np.e ** epsilon) / (2 * b * np.e ** epsilon + 1):
        perturbed_t = random.uniform(- b + t, b + t)
    else:
        x_1 = random.uniform(0, 1)
        if x_1 <= t:
            perturbed_t = random.uniform(- b, - b + t)
        else:
            perturbed_t = random.uniform(b + t, b + 1)

    perturbed_t = (perturbed_t + b) / (2 * b + 1)
    perturbed_t = perturbed_t * delta
    return perturbed_t

# TODO: what is the meaning of "support"?
def find_this_direc_support_of_this_poi(this_poi, this_poi_direc, poi_angel_num_matrix_adap):
    """

    :param this_poi:
    :param this_poi_direc:
    :param poi_angel_num_matrix_adap:
    :return:
    """
    this_poi_direc_num_line = poi_angel_num_matrix_adap[this_poi]
    this_direc_support_array = np.argwhere(this_poi_direc_num_line == this_poi_direc).reshape(-1)

    return this_direc_support_array


def find_all_support_of_this_poi(this_poi, this_poi_direc, poi_angel_num_matrix_adap):
    this_poi_direc_num_line = poi_angel_num_matrix_adap[this_poi]
    this_direc_support_array = np.argwhere(this_poi_direc_num_line >= 0).reshape(-1)

    return this_direc_support_array


def post_process_for_R(traj_center_nearest_poi, perturbed_R, epsilon_for_d1):
    delta = max(poi_distance_matrix[traj_center_nearest_poi])

    b = (epsilon_for_d1 * (np.e ** (epsilon_for_d1)) - np.e ** epsilon_for_d1 + 1) / (
                2 * np.e ** (epsilon_for_d1) * (np.e ** epsilon_for_d1 - 1 - epsilon_for_d1))
    t_star = (perturbed_R / delta) * (2 * b + 1) - b
    t_list = np.arange(0, 1.1, step=0.1)

    res_1_list = []
    res_2_list = []
    t_idx_list = []

    for t_idx, t in enumerate(t_list):
        d = (2 * b * (np.e ** (epsilon_for_d1) - 1) * t) / (2 * b * np.e ** epsilon_for_d1 + 1) + (1 + 2 * b) / (
                    2 * (2 * b * np.e ** (epsilon_for_d1) + 1)) - t
        v = b ** 2 / 3 + ((2 * b + 1) * (b + 1 - 3 * t ** 2)) / (
                    3 * (2 * b * np.e ** epsilon_for_d1 + 1)) - d ** 2 - 2 * d * t

        l = t - b
        r = t + b

        if t_star <= r and t_star >= l:
            t_idx_list.append(t_idx)

    t_cali_list = []
    for t_idx in t_idx_list:
        t_cali_list.append(t_list[t_idx])

    t_l = min(t_cali_list)
    t_r = max(t_cali_list)
    t_l = (t_l + b) * delta / (2 * b + 1)
    t_r = (t_r + b) * delta / (2 * b + 1)

    for poi_idx, poi_distance in enumerate(poi_distance_matrix[traj_center_nearest_poi]):
        if t_l < 0:
            t_l = 0
        if poi_distance <= t_r and poi_distance >= t_l:
            res_1_list.append(poi_distance * np.e ** epsilon_for_d1 / (2 * b * np.e ** epsilon_for_d1 + 1))
        else:
            res_2_list.append(poi_distance * 1 / (2 * b * np.e ** epsilon_for_d1 + 1))

    if len(res_1_list) > 0:
        t_star_cali = (sum(res_1_list) + sum(res_2_list)) \
                      / (len(res_1_list) * np.e ** epsilon_for_d1 / (2 * b * np.e ** epsilon_for_d1 + 1) + len(
            res_2_list) * 1 / (2 * b * np.e ** epsilon_for_d1 + 1))
    else:
        t_star_cali = sum(poi_distance_matrix[traj_center_nearest_poi]) / (poi_distance_matrix.shape[0] - 1)

    return t_star_cali


def sigmoid(x):
    s = 1 / (1 + np.exp(- x))
    return s


def ATP_mechanism(traj, epsilon, poi_angel_num_matrix_adap, poi_angel_num_adap_list, epsilon_direct_rate=0.75,
                  kRR_exp_kRR=False, traj_idx=None):
    """
    :param traj:
    :param poi_angel_num_matrix_adap:
    :param poi_angel_num_adap_list:
    :param epsilon_direct_rate:
    :param kRR_exp_kRR:
    :param traj_idx:
    :return:
    """
    epsilon_traj_center_rate = 0.25
    assert traj_idx != None
    epsilon_direction = epsilon * epsilon_direct_rate * 0.75 / (len(traj) - 1)
    epsilon_traj_center = epsilon * epsilon_traj_center_rate * 0.25
    epsilon_for_d1 = epsilon * epsilon_traj_center_rate * 0.75
    epsilon_per_poi = (epsilon - epsilon * 0.75 * epsilon_direct_rate - epsilon * epsilon_traj_center_rate) / (len(traj))

    traj_array = []
    for this_poi_str in traj:
        traj_array.append(str2index_dict[this_poi_str])

    traj_center_nearest_poi = str2index_dict[Exp_mechanism([traj_center_list[traj_idx]], epsilon_traj_center)[0]]

    R_2 = max(poi_distance_matrix[traj_center_nearest_poi][traj_array])

    R = square_wave_mechanism(traj_center_nearest_poi, R_2, epsilon_for_d1)

    assert R >= 0

    traj_center_dist_matrix = poi_distance_matrix[traj_center_nearest_poi]
    avg_dist = post_process_for_R(traj_center_nearest_poi, R, epsilon_for_d1)

    if R < avg_dist:
        R = R + (avg_dist - R) * np.e ** (- epsilon_for_d1) * sigmoid((avg_dist - R) / (2 * avg_dist))
    else:
        R = R - (R - avg_dist) * np.e ** (- epsilon_for_d1) * sigmoid(
            (R - avg_dist) / (2 * (max(poi_distance_matrix[traj_center_nearest_poi]) - avg_dist)))

    mask = np.argwhere(traj_center_dist_matrix <= R)

    poi_list_for_dist = []
    poi_list_for_direc_dist = []
    perturbed_traj = []
    if kRR_exp_kRR == False:
        for i, this_poi in enumerate(traj_array):
            if i % 2 == 0:
                poi_list_for_dist.append(this_poi)
            else:
                poi_list_for_direc_dist.append(this_poi)
    else:
        for i, this_poi in enumerate(traj_array):
            if i % 2 == 1:
                poi_list_for_dist.append(this_poi)
            else:
                poi_list_for_direc_dist.append(this_poi)
    perturbed_poi_list_for_dist = []
    for i, this_poi in enumerate(poi_list_for_dist):
        x = find_all_support_of_this_poi(this_poi, 1, poi_angel_num_matrix_adap)
        y = np.intersect1d(mask, x).reshape(-1)

        delta_of_this_poi_direc = delta_poi_distance_matrix
        over = np.exp(epsilon_per_poi * (- poi_distance_matrix[this_poi][y] / (2 * delta_of_this_poi_direc)))

        total = over.sum()

        pdf = over / total
        cdf = pdf.copy()
        for j in range(1, pdf.shape[0]):
            cdf[j] += cdf[j - 1]
        rand_uni_num = np.random.random()

        perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
        perturbed_poi_index = y[perturbed_poi_index_of_direc_support_array]
        perturbed_poi_index = int(perturbed_poi_index)

        perturbed_poi_list_for_dist.append(perturbed_poi_index)

    perturbed_poi_list_for_direc_dist = []
    if len(traj_array) % 2 == 1:
        for i, this_poi in enumerate(poi_list_for_direc_dist):
            if kRR_exp_kRR == False:
                perturbed_poi_pre_direc = kRR(poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[i]][this_poi],
                                              poi_angel_num_adap_list, epsilon_direction)
                perturbed_poi_post_direc = kRR(poi_angel_num_matrix_adap[this_poi][perturbed_poi_list_for_dist[i + 1]],
                                               poi_angel_num_adap_list, epsilon_direction)

                reversed_perturbed_poi_post_direc = \
                get_reversed_num_direc_of_traj([perturbed_poi_post_direc], direc_num=len(poi_angel_num_adap_list))[0]

                this_poi_pre_direc_support_array = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[i],
                                                                                       perturbed_poi_pre_direc,
                                                                                       poi_angel_num_matrix_adap)
                this_poi_post_direc_support_array = find_this_direc_support_of_this_poi(
                    perturbed_poi_list_for_dist[i + 1], reversed_perturbed_poi_post_direc, poi_angel_num_matrix_adap)

                this_poi_pre_direc_support_array = np.intersect1d(mask, this_poi_pre_direc_support_array).reshape(-1)

                this_poi_post_direc_support_array = np.intersect1d(mask, this_poi_post_direc_support_array).reshape(-1)

                this_poi_direc_support_array = np.intersect1d(this_poi_pre_direc_support_array,
                                                              this_poi_post_direc_support_array)
            else:
                if i == 0:

                    perturbed_poi_post_direc = kRR(poi_angel_num_matrix_adap[this_poi][perturbed_poi_list_for_dist[i]],
                                                   poi_angel_num_adap_list, epsilon_direction)

                    reversed_perturbed_poi_post_direc = \
                    get_reversed_num_direc_of_traj([perturbed_poi_post_direc], direc_num=len(poi_angel_num_adap_list))[
                        0]
                    this_poi_direc_support_array = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[i],
                                                                                       reversed_perturbed_poi_post_direc,
                                                                                       poi_angel_num_matrix_adap)

                    this_poi_direc_support_array = np.intersect1d(mask, this_poi_direc_support_array).reshape(-1)


                elif i == len(poi_list_for_direc_dist) - 1:

                    perturbed_poi_pre_direc = kRR(poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[-1]][this_poi],
                                                  poi_angel_num_adap_list, epsilon_direction)

                    this_poi_direc_support_array = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[-1],
                                                                                       perturbed_poi_pre_direc,
                                                                                       poi_angel_num_matrix_adap)

                    this_poi_direc_support_array = np.intersect1d(mask, this_poi_direc_support_array).reshape(-1)


                else:

                    perturbed_poi_pre_direc = kRR(poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[i]][this_poi],
                                                  poi_angel_num_adap_list, epsilon_direction)
                    perturbed_poi_post_direc = kRR(
                        poi_angel_num_matrix_adap[this_poi][perturbed_poi_list_for_dist[i + 1]],
                        poi_angel_num_adap_list, epsilon_direction)

                    reversed_perturbed_poi_post_direc = \
                    get_reversed_num_direc_of_traj([perturbed_poi_post_direc], direc_num=len(poi_angel_num_adap_list))[
                        0]

                    this_poi_pre_direc_support_array = find_this_direc_support_of_this_poi(
                        perturbed_poi_list_for_dist[i], perturbed_poi_pre_direc, poi_angel_num_matrix_adap)
                    this_poi_post_direc_support_array = find_this_direc_support_of_this_poi(
                        perturbed_poi_list_for_dist[i + 1], reversed_perturbed_poi_post_direc,
                        poi_angel_num_matrix_adap)

                    this_poi_pre_direc_support_array = np.intersect1d(mask, this_poi_pre_direc_support_array).reshape(
                        -1)

                    this_poi_post_direc_support_array = np.intersect1d(mask, this_poi_post_direc_support_array).reshape(
                        -1)

                    this_poi_direc_support_array = np.intersect1d(this_poi_pre_direc_support_array,
                                                                  this_poi_post_direc_support_array)

            this_poi_dist_support_array = poi_distance_matrix[this_poi][this_poi_direc_support_array]

            if len(this_poi_dist_support_array) > 0:
                if len(this_poi_direc_support_array) != 1:

                    delta_of_this_poi_direc = delta_poi_distance_matrix
                    over = np.exp(epsilon_per_poi * (- this_poi_dist_support_array / (2 * delta_of_this_poi_direc)))

                    total = over.sum()

                    pdf = over / total
                    cdf = pdf.copy()
                    for j in range(1, pdf.shape[0]):
                        cdf[j] += cdf[j - 1]
                    rand_uni_num = np.random.random()

                    perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
                    perturbed_poi_index = this_poi_direc_support_array[perturbed_poi_index_of_direc_support_array]
                    perturbed_poi_str = index2str_dict[perturbed_poi_index]
                else:
                    perturbed_poi_index = this_poi_direc_support_array[0]
                    perturbed_poi_str = index2str_dict[perturbed_poi_index]
            else:
                x = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[i - 1], perturbed_poi_pre_direc,
                                                        poi_angel_num_matrix_adap)
                y = np.intersect1d(mask, x).reshape(-1)
                if len(y > 0):

                    delta_of_this_poi_direc = delta_poi_distance_matrix

                    over = np.exp(epsilon_per_poi * (- poi_distance_matrix[this_poi][y]).reshape(-1) / (
                                2 * delta_of_this_poi_direc))

                    total = over.sum()

                    pdf = over / total
                    cdf = pdf.copy()
                    for j in range(1, pdf.shape[0]):
                        cdf[j] += cdf[j - 1]
                    rand_uni_num = np.random.random()

                    perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
                    perturbed_poi_index = y[perturbed_poi_index_of_direc_support_array]
                else:
                    x = find_all_support_of_this_poi(this_poi, 1, poi_angel_num_matrix_adap)
                    y = np.intersect1d(mask, x).reshape(-1)

                    delta_of_this_poi_direc = delta_poi_distance_matrix

                    over = np.exp(epsilon_per_poi * (- poi_distance_matrix[this_poi][y]).reshape(-1) / (
                                2 * delta_of_this_poi_direc))

                    total = over.sum()

                    pdf = over / total
                    cdf = pdf.copy()
                    for j in range(1, pdf.shape[0]):
                        cdf[j] += cdf[j - 1]
                    rand_uni_num = np.random.random()

                    perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
                    perturbed_poi_index = y[perturbed_poi_index_of_direc_support_array]
                perturbed_poi_str = index2str_dict[perturbed_poi_index]

            perturbed_poi_list_for_direc_dist.append(perturbed_poi_str)
        if kRR_exp_kRR == False:
            perturbed_traj.append(index2str_dict[perturbed_poi_list_for_dist[0]])
            for i, poi_str in enumerate(perturbed_poi_list_for_direc_dist):
                perturbed_traj.append(poi_str)
                perturbed_traj.append(index2str_dict[perturbed_poi_list_for_dist[i + 1]])
        else:
            perturbed_traj.append(perturbed_poi_list_for_direc_dist[0])
            for i, poi_idx in enumerate(perturbed_poi_list_for_dist):
                perturbed_traj.append(index2str_dict[poi_idx])
                perturbed_traj.append(perturbed_poi_list_for_direc_dist[i + 1])

    else:
        for i, this_poi in enumerate(poi_list_for_direc_dist):
            if kRR_exp_kRR == False:
                if i != len(poi_list_for_direc_dist) - 1:
                    perturbed_poi_pre_direc = kRR(poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[i]][this_poi],
                                                  poi_angel_num_adap_list, epsilon_direction)
                    perturbed_poi_post_direc = kRR(
                        poi_angel_num_matrix_adap[this_poi][perturbed_poi_list_for_dist[i + 1]],
                        poi_angel_num_adap_list, epsilon_direction)

                    reversed_perturbed_poi_post_direc = \
                    get_reversed_num_direc_of_traj([perturbed_poi_post_direc], direc_num=len(poi_angel_num_adap_list))[
                        0]

                    this_poi_pre_direc_support_array = find_this_direc_support_of_this_poi(
                        perturbed_poi_list_for_dist[i], perturbed_poi_pre_direc, poi_angel_num_matrix_adap)
                    this_poi_post_direc_support_array = find_this_direc_support_of_this_poi(
                        perturbed_poi_list_for_dist[i + 1], reversed_perturbed_poi_post_direc,
                        poi_angel_num_matrix_adap)

                    this_poi_pre_direc_support_array = np.intersect1d(mask, this_poi_pre_direc_support_array).reshape(
                        -1)

                    this_poi_post_direc_support_array = np.intersect1d(mask, this_poi_post_direc_support_array).reshape(
                        -1)

                    this_poi_direc_support_array = np.intersect1d(this_poi_pre_direc_support_array,
                                                                  this_poi_post_direc_support_array)
                else:
                    perturbed_poi_pre_direc = kRR(poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[i]][this_poi],
                                                  poi_angel_num_adap_list, epsilon_direction)

                    this_poi_direc_support_array = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[i],
                                                                                       perturbed_poi_pre_direc,
                                                                                       poi_angel_num_matrix_adap)

                    this_poi_direc_support_array = np.intersect1d(mask, this_poi_direc_support_array).reshape(-1)

            else:
                if i == 0:
                    perturbed_poi_post_direc = kRR(poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[i]][this_poi],
                                                   poi_angel_num_adap_list, epsilon_direction)

                    this_poi_direc_support_array = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[i],
                                                                                       perturbed_poi_post_direc,
                                                                                       poi_angel_num_matrix_adap)

                    this_poi_direc_support_array = np.intersect1d(mask, this_poi_direc_support_array).reshape(-1)

                else:
                    perturbed_poi_pre_direc = kRR(
                        poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[i - 1]][this_poi],
                        poi_angel_num_adap_list, epsilon_direction)
                    perturbed_poi_post_direc = kRR(poi_angel_num_matrix_adap[this_poi][perturbed_poi_list_for_dist[i]],
                                                   poi_angel_num_adap_list, epsilon_direction)

                    reversed_perturbed_poi_post_direc = \
                    get_reversed_num_direc_of_traj([perturbed_poi_post_direc], direc_num=len(poi_angel_num_adap_list))[
                        0]

                    this_poi_pre_direc_support_array = find_this_direc_support_of_this_poi(
                        perturbed_poi_list_for_dist[i - 1], perturbed_poi_pre_direc, poi_angel_num_matrix_adap)
                    this_poi_post_direc_support_array = find_this_direc_support_of_this_poi(
                        perturbed_poi_list_for_dist[i], reversed_perturbed_poi_post_direc, poi_angel_num_matrix_adap)

                    this_poi_pre_direc_support_array = this_poi_pre_direc_support_array[mask].reshape(-1)

                    this_poi_post_direc_support_array = this_poi_post_direc_support_array[mask].reshape(-1)

                    this_poi_direc_support_array = np.intersect1d(this_poi_pre_direc_support_array,
                                                                  this_poi_post_direc_support_array)

            this_poi_dist_support_array = poi_distance_matrix[this_poi][this_poi_direc_support_array]

            if len(this_poi_dist_support_array) > 0:
                if len(this_poi_direc_support_array) != 1:
                    delta_of_this_poi_direc = delta_poi_distance_matrix

                    over = np.exp(epsilon_per_poi * (- this_poi_dist_support_array / (2 * delta_of_this_poi_direc)))
                    total = over.sum()

                    pdf = over / total
                    cdf = pdf.copy()
                    for j in range(1, pdf.shape[0]):
                        cdf[j] += cdf[j - 1]
                    rand_uni_num = np.random.random()

                    perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
                    perturbed_poi_index = this_poi_direc_support_array[perturbed_poi_index_of_direc_support_array]
                    perturbed_poi_str = index2str_dict[perturbed_poi_index]
                else:
                    perturbed_poi_index = this_poi_direc_support_array[0]
                    perturbed_poi_str = index2str_dict[perturbed_poi_index]
            else:
                x = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[i - 1], perturbed_poi_pre_direc,
                                                        poi_angel_num_matrix_adap)
                y = np.intersect1d(mask, x).reshape(-1)
                if len(y > 0):
                    delta_of_this_poi_direc = delta_poi_distance_matrix

                    over = np.exp(epsilon_per_poi * (- poi_distance_matrix[this_poi][y]).reshape(-1) / (
                                2 * delta_of_this_poi_direc))
                    total = over.sum()

                    pdf = over / total
                    cdf = pdf.copy()
                    for j in range(1, pdf.shape[0]):
                        cdf[j] += cdf[j - 1]
                    rand_uni_num = np.random.random()

                    perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
                    perturbed_poi_index = y[perturbed_poi_index_of_direc_support_array]
                else:
                    x = find_all_support_of_this_poi(this_poi, 1, poi_angel_num_matrix_adap)
                    y = np.intersect1d(mask, x).reshape(-1)
                    delta_of_this_poi_direc = delta_poi_distance_matrix

                    over = np.exp(epsilon_per_poi * (- poi_distance_matrix[this_poi][y]).reshape(-1) / (
                                2 * delta_of_this_poi_direc))
                    total = over.sum()

                    pdf = over / total
                    cdf = pdf.copy()
                    for j in range(1, pdf.shape[0]):
                        cdf[j] += cdf[j - 1]
                    rand_uni_num = np.random.random()

                    perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
                    perturbed_poi_index = y[perturbed_poi_index_of_direc_support_array]

                perturbed_poi_str = index2str_dict[perturbed_poi_index]

            perturbed_poi_list_for_direc_dist.append(perturbed_poi_str)

        if kRR_exp_kRR == False:
            for i, poi_str in enumerate(perturbed_poi_list_for_direc_dist):
                perturbed_traj.append(index2str_dict[perturbed_poi_list_for_dist[i]])
                perturbed_traj.append(poi_str)
        else:
            for i, poi_str in enumerate(perturbed_poi_list_for_direc_dist):
                perturbed_traj.append(poi_str)
                perturbed_traj.append(index2str_dict[perturbed_poi_list_for_dist[i]])

    return perturbed_traj


def Exp_mechanism(traj, epsilon):
    traj_array = []
    for this_poi_str in traj:
        traj_array.append(str2index_dict[this_poi_str])
    traj_array = np.array(traj_array)
    delta = delta_poi_distance_matrix
    epsilon = epsilon / len(traj)
    epsilon_array = np.array(epsilon)
    over = np.exp(epsilon_array * (- poi_distance_matrix[traj_array]) / (2 * delta))
    total = over.sum(axis=1).reshape((-1, 1))
    cdf = over / total

    pdf = cdf.copy().T
    for j in range(1, cdf.shape[1]):
        pdf[j] += pdf[j - 1]
    pdf = pdf.T

    rand_uni_num = np.random.random((pdf.shape[0], 1))

    perturbed_traj = traj_array.copy()
    for i in range(rand_uni_num.shape[0]):
        perturbed_traj[i] = np.searchsorted(pdf[i], rand_uni_num[i], side='right')

    perturbed_traj_str = []
    for this_perturbed_poi in perturbed_traj:
        perturbed_traj_str.append(index2str_dict[this_perturbed_poi])

    return perturbed_traj_str


def get_errors_of_perturbed_traj(traj, perturbed_traj):
    errors_of_perturbed_traj = 0

    for i, poi in enumerate(traj):
        this_poi_error = poi_distance_matrix[str2index_dict[poi]][str2index_dict[perturbed_traj[i]]]
        errors_of_perturbed_traj += this_poi_error

    normed_errors_of_perturbed_traj = errors_of_perturbed_traj / len(traj)

    return normed_errors_of_perturbed_traj


def get_preservation_range_queries_errors_of_perturbed_traj(traj, perturbed_traj):
    errors_of_perturbed_traj = 0

    dist_list = [0.5, 1, 2, 3, 4]
    # For CPS:
    # dist_list = [0.1,0.25,0.5,1,2]
    normed_errors_of_perturbed_traj = np.zeros((1, len(dist_list)))

    for dist_idx, dist in enumerate(dist_list):

        errors_of_perturbed_traj = 0

        for i, poi in enumerate(traj):
            this_poi_error = 0
            if poi_distance_matrix[str2index_dict[poi]][str2index_dict[perturbed_traj[i]]] <= dist:
                this_poi_error = 1
            errors_of_perturbed_traj += this_poi_error

        normed_errors_of_perturbed_traj[0][dist_idx] = errors_of_perturbed_traj / len(traj)

    return normed_errors_of_perturbed_traj


def perturb_traj(traj, epsilon, poi_angel_num_matrix_adap, poi_angel_num_adap_list, mechanism='Exp_mechanism',
                 kRR_exp_kRR=False, traj_idx=None):
    if mechanism == 'ATP_mechanism':
        perturb_mechanism = ATP_mechanism
    else:
        print('please choose a correct mechanism.')

    perturbed_traj = perturb_mechanism(traj, epsilon, poi_angel_num_matrix_adap, poi_angel_num_adap_list, kRR_exp_kRR,
                                       traj_idx=traj_idx)

    errors_of_perturbed_traj = get_errors_of_perturbed_traj(traj, perturbed_traj)

    return perturbed_traj, errors_of_perturbed_traj


traj_list_without_timestamp = []
traj_list_timestamp = []

for traj in traj_list:
    this_traj = []
    this_traj_timestamp = []
    for this_poi in traj:
        this_traj.append(this_poi[0])
        this_traj_timestamp.append(this_poi[1])
        if this_poi[0] not in latitude_dict.keys():
            print(this_poi)
    traj_list_without_timestamp.append(np.array(this_traj))
    traj_list_timestamp.append(this_traj_timestamp)

epsilon_list = [0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 10]


def find_opt_traj(perturbed_traj, perturbed_reversed_traj):
    opt_traj = []
    perturbed_reversed_traj = list(reversed(perturbed_reversed_traj))
    for i in range(len(perturbed_traj)):
        dist_matrix = poi_distance_matrix[str2index_dict[perturbed_traj[i]]] + \
                      poi_distance_matrix[str2index_dict[perturbed_reversed_traj[i]]]
        min_poi_idx = np.argmin(dist_matrix)
        opt_traj.append(index2str_dict[min_poi_idx])
    return opt_traj


def get_px_errors_of_eps_idx(errors_total_px, eps_idx):
    this_eps_errors = np.zeros((1, 5))
    for idx in range(5):
        this_eps_errors += errors_total_px[idx][eps_idx]
    errors_total_px_ = this_eps_errors / 5

    return errors_total_px_


errors_NE = np.zeros((5, 9))
errors_PQR = np.zeros((5, 9, 5))
for k in range(5):
    errors_list = []
    errors_PQR_matrix = np.zeros((len(epsilon_list), 5))
    for eps_idx, epsilon in enumerate(epsilon_list):
        errors = 0
        errors_px = np.zeros((1, 5))

        poi_angel_num_matrix_adap = None
        poi_angel_num_adap_list = None
        if epsilon == 10:
            poi_angel_num_matrix_adap = poi_angel_num_matrix_12
            poi_angel_num_adap_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        elif epsilon <= 2:
            poi_angel_num_matrix_adap = poi_angel_num_matrix_2
            poi_angel_num_adap_list = [0, 1]
        elif epsilon == 4:
            poi_angel_num_matrix_adap = poi_angel_num_matrix_4
            poi_angel_num_adap_list = [0, 1, 2, 3]
        else:
            poi_angel_num_matrix_adap = poi_angel_num_matrix_6
            poi_angel_num_adap_list = [0, 1, 2, 3, 4, 5]

        for i, trajectory in enumerate(traj_list_without_timestamp):
            j = 0
            reversed_trajectory = list(reversed(trajectory))
            perturbed_traj, _ = perturb_traj(trajectory, epsilon / 2, poi_angel_num_matrix_adap,
                                             poi_angel_num_adap_list, mechanism='ATP_mechanism', traj_idx=i)
            perturbed_reversed_traj, _ = perturb_traj(reversed_trajectory, epsilon / 2, poi_angel_num_matrix_adap,
                                                      poi_angel_num_adap_list, mechanism='ATP_mechanism',
                                                      kRR_exp_kRR=True, traj_idx=i)
            perturbed_traj = find_opt_traj(perturbed_traj, perturbed_reversed_traj)
            errors_of_perturbed_traj = get_errors_of_perturbed_traj(trajectory, perturbed_traj)
            errors_px_of_perturbed_traj = get_preservation_range_queries_errors_of_perturbed_traj(trajectory,
                                                                                                  perturbed_traj)

            errors += errors_of_perturbed_traj
            errors_px += errors_px_of_perturbed_traj
            if (i + 1) % 1000 == 0:
                print('epsilon : ', epsilon, i + 1, '/', len(traj_list_without_timestamp), 'done.')
                tm = time.localtime(time.time())
                print('time:', tm.tm_hour, ':', tm.tm_min, ':', tm.tm_sec)

        errors = errors / len(traj_list)
        errors_px = errors_px / len(traj_list)
        errors_list.append(errors)
        errors_PQR_matrix[eps_idx] = errors_px
        print('errors_NE k=', k + 1, 'epsilon = ', epsilon, 'done.', errors, 'px errors', errors_px)
        tm1 = time.localtime(time.time())
        print('time:', tm1.tm_hour, ':', tm1.tm_min, ':', tm1.tm_sec)

    errors_NE[k] = errors_NE[k] + np.array(errors_list)
    errors_PQR[k] = errors_PQR_matrix

errors_NE = errors_NE.sum(axis=0) / 5
errors_PQR_list = []
for i in range(9):
    errors_PQR_ = get_px_errors_of_eps_idx(errors_PQR, eps_idx=i)
    errors_PQR_list.append(errors_PQR_)
print('errors_PQR', errors_PQR_)
print('errors_NE', errors_NE)

errors_NE = errors_NE.tolist()
errors_PQR_ = errors_PQR_.tolist()

with open(Path + 'errors_ATP_file.pickle', 'wb') as errors_NE_read:
    pickle.dump(errors_NE, errors_NE_read)

with open(Path + 'errors_px_ATP_file.pickle', 'wb') as errors_PQR_read:
    pickle.dump(errors_PQR_list, errors_PQR_read)

plt.plot(np.arange(len(epsilon_list)), errors_NE, marker='x', color='r')
plt.xticks(np.arange(len(epsilon_list)), epsilon_list)
plt.show()
